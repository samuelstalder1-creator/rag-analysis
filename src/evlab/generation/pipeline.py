from __future__ import annotations

from pathlib import Path
from typing import Iterable
from difflib import SequenceMatcher

from evlab.generation.false_answer import FALSE_ANSWER_PROMPT
from evlab.generation.false_answers import valid_saved_answer
from evlab.generation.minimal_edit import edit_answer_spans, edit_all_answer_spans, person_name_parts, MINIMAL_EDIT_PROMPT
from evlab.generation.paraphrase import (
    paraphrase, PARAPHRASE_PROMPT, paraphrase_false, FALSE_DOCUMENT_PROMPT,
    verify_false_answerability,
)
from evlab.generation.output import GenerationOutput, source_identity
from evlab.generation.validate import (
    contains_alias,
    validate_correct_copy,
    validate_false_copy,
    validate_nonempty,
    expand_number_aliases,
)
from evlab.generation.semantic import (
    build_verifier,
    verify_document,
    validator_signature,
    deterministic_receipt,
    validation_mode,
)
from evlab.generation.minimal_edit import occurrences
from evlab.io import append_jsonl, iter_jsonl, stable_hash, write_jsonl
from evlab.llm.cache import CachedLLMClient
from evlab.llm.client import build_llm_client


def run_generation_pipeline(
    config: dict, candidates: Iterable[dict], *, limit: int | None = None
) -> dict[str, int]:
    conditions = _conditions(config)
    if int(config.get("samples_per_h", 1)) < 1:
        raise ValueError("samples_per_h must be positive")
    if not 0 < float(config.get("validation", {}).get("max_similarity", 0.95)) <= 1:
        raise ValueError("validation.max_similarity must be in (0, 1]")
    max_attempts = config.get("false_document_max_attempts", 1)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("false_document_max_attempts must be a positive integer")
    retry_seed = config.get("retry_seed", 0)
    if type(retry_seed) is not int or retry_seed < 0:
        raise ValueError("retry_seed must be a non-negative integer")
    excluded = config.get("exclude_doc_ids", [])
    if not isinstance(excluded, list) or any(not isinstance(did, str) or not did for did in excluded):
        raise ValueError("exclude_doc_ids must be a list of nonempty document IDs")
    excluded_ids = set(excluded)
    excluded_seen: set[str] = set()
    semantic = validation_mode(config) == "semantic"
    span_selection = str(config.get("span_selection", "llm"))
    if span_selection not in {"llm", "all"}:
        raise ValueError("span_selection must be 'llm' or 'all'")
    selected = list(candidates)
    if limit is not None:
        selected = selected[:limit]
    skipped: list[str] = []
    if "false" in conditions:
        invalid = []
        for candidate in selected:
            if not valid_saved_answer(
                candidate, require_semantic=semantic,
                require_source_constraints=config.get("require_source_constraints", config.get("false_answer_source") == "nqopen_import"),
            ):
                invalid.append(str(candidate["query_id"]))
        if invalid and config.get("skip_missing_false_answers", False):
            # Questions whose answer stage failed are left out, not silently filled.
            skipped = invalid
            selected = [c for c in selected if str(c["query_id"]) not in set(invalid)]
            print(
                f"generation: skipping {len(skipped)} question(s) without a valid saved "
                f"false answer: {', '.join(skipped[:10])}",
                flush=True,
            )
        elif invalid:
            remedy = (
                "Rebuild nqopen+ and review false_answer_review.jsonl"
                if config.get("false_answer_source") == "nqopen_import"
                else "Run evlab generate-false-answers --config <config> first"
            )
            raise ValueError(
                f"Missing, stale or unverified saved false answers: {', '.join(invalid[:10])}. {remedy}"
            )
    client = CachedLLMClient(
        build_llm_client(config["llm"]), config.get("cache_path", "cache/llm.sqlite")
    )
    # Without a judge, span selection is a generation step run by the generator.
    verifier = build_verifier(config) if semantic else client
    signature = stable_hash(
        {
            "config": {
                k: v
                for k, v in config.items()
                if k
                not in {
                    "resume",
                    "progress_every",
                    "output_dir",
                    "cache_path",
                    "conditions",
                    "samples_per_h",
                    "skip_missing_false_answers",
                    "retry_seed",  # Operational retry control; completed fingerprints stay valid.
                    "exclude_doc_ids",  # Review hold; does not change generated content.
                }
            },
            "prompts": [FALSE_ANSWER_PROMPT, MINIMAL_EDIT_PROMPT, PARAPHRASE_PROMPT, FALSE_DOCUMENT_PROMPT],
            "false_document_validation_goal": "query_target_answerability_v1",
            "validator": validator_signature(),
        }
    )
    output = GenerationOutput(
        Path(config.get("output_dir", "data/generated/pilot")),
        {**config, "generation_signature": signature,
         **({} if semantic else {"validation_mode": "deterministic"})},
        client.model,
        conditions,
    )
    rejects_path = output.output_dir / f"{output.label}--rejects.jsonl"
    reject_keys = (
        _existing_reject_keys(rejects_path) if config.get("resume", True) else set()
    )
    if not config.get("resume", True):
        write_jsonl(rejects_path, [])
    generated_count = reject_count = 0
    temperature = float(config.get("temperature", 0.3))
    every = int(config.get("progress_every", 25))
    try:
        for idx, candidate in enumerate(selected):
            question = (
                ""
                if config.get("source") == "human_corpus"
                else str(candidate["query"])
            )
            gold = str(candidate["answer"])
            aliases = list(candidate.get("answer_aliases", [gold]))
            for doc in candidate.get("h_docs", []):
                did, title, text = (
                    str(doc["doc_id"]),
                    str(doc.get("title", "") or ""),
                    str(doc["text"]),
                )
                if did in excluded_ids:
                    excluded_seen.add(did)
                    continue
                for condition in sorted(conditions):
                    target = (
                        candidate.get("false_answer") if condition == "false" else None
                    )
                    fingerprint = stable_hash(
                        {
                            "signature": signature,
                            "question": question,
                            "gold": gold,
                            "aliases": aliases,
                            "target": target,
                            "answer_record": (
                                candidate.get("false_answer_record")
                                if condition == "false"
                                else None
                            ),
                            "source": source_identity(doc, aliases),
                            "condition": condition,
                        }
                    )
                    for sample in range(int(config.get("samples_per_h", 1))):
                        if output.contains(condition, sample, did, fingerprint):
                            continue
                        seed = idx * 1000 + sample * 2 + int(condition == "false") + retry_seed * 10000000
                        edited_title, edited_text, edits = title, text, {}
                        generated = None
                        try:
                            seed_receipt = None
                            if condition == "false":
                                try:
                                    edited_text, edits["body"] = _edit_spans(
                                        span_selection, verifier, text=text,
                                        context_title=title, gold=gold, aliases=aliases,
                                        target_answer=target, question=question, seed=idx,
                                    )
                                except ValueError as exc:
                                    if (not str(exc).startswith("No editable gold-answer spans in source")
                                            or occurrences(text, aliases)
                                            or not occurrences(title, aliases)):
                                        raise
                                    # The title supplies the subject; keep the original body
                                    # and expand the edited title in the generated passage.
                                    edits["body"] = []
                                    edits["answer_from_title"] = True
                                if any(
                                    contains_alias(title, alias) for alias in aliases
                                ):
                                    edited_title, edits["title"] = _edit_spans(
                                        span_selection,
                                        verifier,
                                        text=title,
                                        gold=gold,
                                        aliases=aliases,
                                        target_answer=target,
                                        question=question,
                                        seed=idx,
                                    )
                                elif span_selection == "all" and person_name_parts(gold):
                                    # Titles often use a shortened name ("Christopher
                                    # Lloyd" for "Christopher Allen Lloyd").
                                    try:
                                        edited_title, edits["title"] = edit_all_answer_spans(
                                            title, aliases, target, gold=gold)
                                    except ValueError:
                                        pass
                                seed_receipt = _check_document(
                                    semantic,
                                    verifier,
                                    question=question,
                                    gold=gold,
                                    aliases=aliases,
                                    target=target,
                                    source=title + "\n" + text,
                                    seed=edited_title + "\n" + edited_text,
                                    generated=edited_title + "\n" + edited_text,
                                    condition=condition,
                                    stage="false_seed",
                                )
                            if condition == "false":
                                generated = paraphrase_false(
                                    client,
                                    question=question,
                                    target_answer=target,
                                    title=edited_title,
                                    text=edited_text,
                                    temperature=temperature,
                                    seed=seed,
                                    max_attempts=config.get("false_document_max_attempts", 1),
                                    max_similarity=float(config.get("validation", {}).get("max_similarity", 0.95)),
                                    source_text=text,
                                    answer_aliases=aliases,
                                )
                            else:
                                generated = paraphrase(
                                    client,
                                    text=edited_text,
                                    temperature=temperature,
                                    seed=seed,
                                )
                            lexical = (
                                validate_false_copy(
                                    generated,
                                    target_answer=target,
                                    answer_aliases=aliases,
                                    allow_unrelated_gold=True,
                                )
                                if condition == "false"
                                else (
                                    validate_correct_copy(
                                        generated, answer_aliases=aliases
                                    )
                                    if question and aliases
                                    else validate_nonempty(generated)
                                )
                            )
                            if not lexical.ok:
                                raise ValueError(",".join(lexical.reasons))
                            similarity = SequenceMatcher(
                                None, edited_text.casefold(), generated.casefold()
                            ).ratio()
                            if similarity >= float(
                                config.get("validation", {}).get("max_similarity", 0.95)
                            ):
                                raise ValueError("near_duplicate_of_seed")
                            if condition == "false" and SequenceMatcher(
                                None, text.casefold(), generated.casefold()
                            ).ratio() >= float(
                                config.get("validation", {}).get("max_similarity", 0.95)
                            ):
                                raise ValueError("near_duplicate_of_source")
                            if any(
                                SequenceMatcher(
                                    None, previous.casefold(), generated.casefold()
                                ).ratio()
                                >= float(
                                    config.get("validation", {}).get(
                                        "max_similarity", 0.95
                                    )
                                )
                                for previous in output.previous_texts(condition, did)
                            ):
                                raise ValueError("near_duplicate_of_other_sample")
                            # Gold mentions may survive only where the edit left
                            # unrelated occurrences; anything more is residue.
                            # Number words count too: "three" is residue for gold "3".
                            residue_aliases = expand_number_aliases(aliases)
                            if condition == "false" and len(
                                occurrences(generated, residue_aliases)
                            ) > len(occurrences(edited_text, residue_aliases)):
                                raise ValueError("gold_answer_residue")
                            receipt = _check_document(
                                semantic,
                                verifier,
                                question=question,
                                gold=gold,
                                aliases=aliases,
                                target=target,
                                source=title + "\n" + text,
                                seed=edited_title + "\n" + edited_text,
                                generated=edited_title + "\n" + generated,
                                condition=condition,
                                stage="paraphrase",
                            )
                            generated_count += output.append(
                                {
                                    "condition": condition,
                                    "sample_idx": sample,
                                    "provenance_root_id": did,
                                    "title": edited_title,
                                    "text": generated,
                                    "validation_receipt": {
                                        "retry_seed": retry_seed,
                                        "fingerprint": fingerprint,
                                        "query_id": candidate["query_id"],
                                        "gold_answer": gold,
                                        "target_answer": target,
                                        "false_answer_origin": (
                                            candidate.get("false_answer_record", {}).get("origin", "generated")
                                            if condition == "false" else None
                                        ),
                                        "false_answer_record_hash": (
                                            stable_hash(candidate["false_answer_record"])
                                            if condition == "false" else None
                                        ),
                                        "source_hash": stable_hash(source_identity(doc, aliases)),
                                        "source_content_hash": stable_hash(
                                            {
                                                "doc_id": did,
                                                "title": title,
                                                "text": text,
                                            }
                                        ),
                                        "question_hash": (
                                            stable_hash(
                                                {
                                                    "question": question,
                                                    "gold": gold,
                                                    "aliases": aliases,
                                                }
                                            )
                                            if question
                                            else None
                                        ),
                                        "edits": edits,
                                        "seed_text": edited_text,
                                        "seed_title": edited_title,
                                        "seed_validation": seed_receipt,
                                        "validation": receipt,
                                        "similarity": similarity,
                                    },
                                }
                            )
                        except Exception as exc:
                            attempts = getattr(exc, "attempts", [])
                            if generated is None and attempts and attempts[-1].get("stage") == "paraphrase":
                                generated = attempts[-1].get("response")
                            reject_count += _append_reject(
                                rejects_path,
                                {
                                    "query_id": candidate["query_id"],
                                    "doc_id": did,
                                    "condition": condition,
                                    "sample_idx": sample,
                                    "stage": f"{condition}_validation",
                                    "reason": str(exc),
                                    "attempts": getattr(exc, "attempts", []),
                                    "source_text": text,
                                    "false_seed_text": edited_text,
                                    "target_answer": target,
                                    "generated_text": generated,
                                    "generated_title": edited_title,
                                    "edits": edits,
                                    "answer": gold,
                                    "answer_aliases": aliases,
                                },
                                reject_keys,
                            )
            if every > 0 and (idx + 1) % every == 0:
                output.update_counts()
                print(
                    f"generation: candidates={idx + 1} generated={generated_count} rejects={reject_count}",
                    flush=True,
                )
    finally:
        output.update_counts()
    result = {"generated": generated_count, "rejects": reject_count}
    if excluded_seen:
        result["excluded_documents"] = len(excluded_seen)
    if config.get("skip_missing_false_answers", False):
        result["skipped_questions"] = len(skipped)
    return result


def _edit_spans(mode: str, client, *, text, aliases, target_answer, question, seed, gold=None, context_title=""):
    if mode == "all":
        return edit_all_answer_spans(text, aliases, target_answer, gold=gold, context_title=context_title)
    return edit_answer_spans(client, text=text, aliases=aliases, target_answer=target_answer,
                             question=question, seed=seed)


def _check_document(semantic: bool, verifier, **kwargs) -> dict:
    target_goal = kwargs["condition"] == "false" and kwargs["stage"] == "paraphrase"
    if semantic:
        if target_goal:
            return verify_false_answerability(verifier, **kwargs)
        return verify_document(verifier, **kwargs)
    evidence = {key: kwargs[key] for key in
                ("question", "gold", "aliases", "target", "source", "seed", "generated", "condition")}
    # The lexical, similarity and residue checks ran before this point.
    checks = {"nonempty": bool(kwargs["generated"].strip())}
    receipt = deterministic_receipt(stage=kwargs["stage"], evidence=evidence, checks=checks)
    if target_goal:
        receipt.update(validation_goal="query_target_answerability", answerability_verified=False)
    return receipt


def _conditions(config: dict) -> set[str]:
    raw = config.get("conditions", ["correct", "false"])
    values = (
        {_condition_value(raw)}
        if isinstance(raw, str)
        else {_condition_value(item) for item in raw}
    )
    if values - {"correct", "false"}:
        raise ValueError(f"Unsupported generation conditions: {sorted(values)}")
    return values or {"correct", "false"}


def _condition_value(value: object) -> str:
    return "false" if value is False else str(value).lower()


def _existing_reject_keys(path: Path) -> set[str]:
    return (
        {str(row.get("reject_key") or stable_hash(row)) for row in iter_jsonl(path)}
        if path.exists()
        else set()
    )


def _append_reject(path: Path, row: dict, reject_keys: set[str]) -> int:
    key = stable_hash(row)
    if key in reject_keys:
        return 0
    append_jsonl(path, [{**row, "reject_key": key}])
    reject_keys.add(key)
    return 1
