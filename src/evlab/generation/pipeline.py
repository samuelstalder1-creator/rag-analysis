from __future__ import annotations

from pathlib import Path
from typing import Iterable

from evlab.generation.false_answer import is_valid_false_answer
from evlab.generation.minimal_edit import minimal_edit_with_fallback
from evlab.generation.paraphrase import paraphrase
from evlab.generation.output import GenerationOutput
from evlab.generation.validate import contains_alias, validate_correct_copy, validate_false_copy, validate_nonempty
from evlab.io import append_jsonl, iter_jsonl, stable_hash, write_jsonl
from evlab.llm.cache import CachedLLMClient
from evlab.llm.client import build_llm_client


def run_generation_pipeline(config: dict, candidates: Iterable[dict], *, limit: int | None = None) -> dict[str, int]:
    conditions = _conditions(config)
    candidates = list(candidates)
    selected = candidates if limit is None else candidates[:limit]
    if "false" in conditions:
        invalid = []
        for candidate in selected:
            answer = candidate.get("false_answer")
            if not isinstance(answer, str) or not is_valid_false_answer(
                answer, str(candidate["answer"]), list(candidate.get("answer_aliases", [candidate["answer"]]))
            ):
                invalid.append(str(candidate["query_id"]))
        if invalid:
            raise ValueError(
                f"Missing or invalid saved false answers for {len(invalid)} questions: {', '.join(invalid[:10])}. "
                "Run evlab generate-false-answers --config <config> first; answers are read from generated_false_answers.jsonl in the output directory."
            )
    llm_config = config["llm"]
    client = CachedLLMClient(build_llm_client(llm_config), config.get("cache_path", "cache/llm.sqlite"))
    output_dir = Path(config.get("output_dir", "data/generated/pilot"))
    temperature = float(config.get("temperature", 0.3))
    per_h_samples = int(config.get("samples_per_h", 1))
    progress_every = int(config.get("progress_every", 25))
    require_answer_in_correct = bool(config.get("require_answer_in_correct", True))
    resume = bool(config.get("resume", True))
    output = GenerationOutput(output_dir, config, client.model, conditions)
    for path in sorted(output.paths.values()):
        print(f"generation corpus: {path}", flush=True)
    rejects_path = output_dir / f"{output.label}--rejects.jsonl"
    reject_keys = _existing_reject_keys(rejects_path) if resume else set()
    if not resume:
        write_jsonl(rejects_path, [])
    generated_count = 0
    reject_count = 0

    try:
        for idx, candidate in enumerate(candidates):
            if limit is not None and idx >= limit:
                break
            if progress_every > 0 and idx % progress_every == 0:
                print(
                    f"generation progress: candidates={idx} generated={generated_count} rejects={reject_count}",
                    flush=True,
                )
            query_id = str(candidate["query_id"])
            question = str(candidate["query"])
            answer = str(candidate["answer"])
            aliases = [str(item) for item in candidate.get("answer_aliases", [answer])]
            pending_conditions = {
                condition for condition in conditions
                if any(not output.contains(condition, sample, str(doc["doc_id"]))
                       for doc in candidate.get("h_docs", []) for sample in range(per_h_samples))
            }
            if not pending_conditions:
                continue
            target_answer = str(candidate["false_answer"]) if "false" in pending_conditions else None

            for h_doc in candidate.get("h_docs", []):
                h_id = str(h_doc["doc_id"])
                h_title = str(h_doc.get("title", "") or "")
                h_text = str(h_doc["text"])
                aliases_in_doc = [str(item) for item in h_doc.get("aliases_in_doc", aliases)]
                false_seed = None
                false_title = h_title
                title_validation = None
                if "false" in conditions and target_answer is not None:
                    if any(contains_alias(h_title, alias) for alias in aliases):
                        false_title = minimal_edit_with_fallback(
                            client,
                            text=h_title,
                            aliases=aliases,
                            target_answer=target_answer,
                            seed=idx,
                        )
                        title_validation = validate_false_copy(
                            false_title, target_answer=target_answer, answer_aliases=aliases
                        )
                    false_seed = minimal_edit_with_fallback(
                        client,
                        text=h_text,
                        aliases=aliases_in_doc,
                        target_answer=target_answer,
                        seed=idx,
                    )
                for sample_idx in range(per_h_samples):
                    sample_seed = idx * 1000 + sample_idx
                    if "correct" in conditions and not output.contains("correct", sample_idx, h_id):
                        correct_text = paraphrase(client, text=h_text, temperature=temperature, seed=sample_seed)
                        correct_validation = (
                            validate_correct_copy(correct_text, answer_aliases=aliases)
                            if require_answer_in_correct and aliases
                            else validate_nonempty(correct_text)
                        )
                        if correct_validation.ok:
                            generated_count += output.append(
                                _row(
                                    prefix="AC",
                                    title=h_title,
                                    query_id=query_id,
                                    h_id=h_id,
                                    text=correct_text,
                                    condition="correct",
                                    answer=answer,
                                    target_answer=None,
                                    model=client.model,
                                    sample_idx=sample_idx,
                                ),
                            )
                        else:
                            reject_count += _append_reject(
                                rejects_path,
                                {
                                    "candidate_idx": idx,
                                    "query_id": query_id,
                                    "question": question,
                                    "doc_id": h_id,
                                    "title": h_title,
                                    "condition": "correct",
                                    "stage": "correct_validation",
                                    "sample_idx": sample_idx,
                                    "reasons": list(correct_validation.reasons),
                                    "answer": answer,
                                    "answer_aliases": aliases,
                                    "aliases_in_doc": aliases_in_doc,
                                    "source_text": h_text,
                                    "generated_text": correct_text,
                                },
                                reject_keys,
                            )

                    if ("false" in conditions and false_seed is not None and target_answer is not None
                            and not output.contains("false", sample_idx, h_id)):
                        false_text = paraphrase(client, text=false_seed, temperature=temperature, seed=sample_seed + 1)
                        false_validation = validate_false_copy(false_text, target_answer=target_answer, answer_aliases=aliases)
                        reasons = list(false_validation.reasons)
                        if title_validation is not None:
                            reasons.extend(f"title_{reason}" for reason in title_validation.reasons)
                        if not reasons:
                            generated_count += output.append(
                                _row(
                                    prefix="AF",
                                    query_id=query_id,
                                    h_id=h_id,
                                    title=false_title,
                                    text=false_text,
                                    condition="false",
                                    answer=answer,
                                    target_answer=target_answer,
                                    model=client.model,
                                    sample_idx=sample_idx,
                                ),
                            )
                        else:
                            reject_count += _append_reject(
                                rejects_path,
                                {
                                    "candidate_idx": idx,
                                    "query_id": query_id,
                                    "question": question,
                                    "doc_id": h_id,
                                    "title": h_title,
                                    "condition": "false",
                                    "stage": "false_validation",
                                    "sample_idx": sample_idx,
                                    "reasons": reasons,
                                    "answer": answer,
                                    "target_answer": target_answer,
                                    "answer_aliases": aliases,
                                    "aliases_in_doc": aliases_in_doc,
                                    "source_text": h_text,
                                    "false_seed_text": false_seed,
                                    "generated_title": false_title,
                                    "generated_text": false_text,
                                },
                                reject_keys,
                            )
            if progress_every > 0 and (idx + 1) % progress_every == 0:
                output.update_counts()
    finally:
        output.update_counts()


    print(
        f"generation complete: generated={generated_count} rejects={reject_count}",
        flush=True,
    )
    return {"generated": generated_count, "rejects": reject_count}


def _conditions(config: dict) -> set[str]:
    raw = config.get("conditions", ["correct", "false"])
    if isinstance(raw, str):
        values = {_condition_value(raw)}
    else:
        values = {_condition_value(item) for item in raw}
    unsupported = values - {"correct", "false"}
    if unsupported:
        raise ValueError(f"Unsupported generation conditions: {sorted(unsupported)}")
    return values or {"correct", "false"}


def _condition_value(value: object) -> str:
    if value is False:
        return "false"
    return str(value).lower()


def _row(
    *,
    prefix: str,
    query_id: str,
    h_id: str,
    text: str,
    condition: str,
    answer: str,
    target_answer: str | None,
    model: str,
    sample_idx: int,
    title: str = "",
) -> dict:
    return {
        "doc_id": f"gen::{prefix}_{query_id}_{h_id}_{model.replace('/', '-')}_s{sample_idx:02d}_d1",
        "title": title,
        "text": text,
        "origin": "generated",
        "query_id": query_id,
        "condition": condition,
        "gold_answer": answer,
        "target_answer": target_answer,
        "parent_id": h_id if condition == "correct" else f"{h_id}*",
        "seed_id": h_id if condition == "correct" else f"{h_id}*",
        "provenance_root_id": h_id,
        "generation_depth": 1,
        "generator_model": model,
        "prompt_version": "v1",
        "sample_idx": sample_idx,
    }


def _existing_reject_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("reject_key") or _reject_key(row)) for row in iter_jsonl(path)}


def _append_reject(path: Path, row: dict, reject_keys: set[str]) -> int:
    key = _reject_key(row)
    if key in reject_keys:
        return 0
    row = dict(row)
    row["reject_key"] = key
    append_jsonl(path, [row])
    reject_keys.add(key)
    return 1


def _reject_key(row: dict) -> str:
    comparable = {key: value for key, value in row.items() if key != "reject_key"}
    return stable_hash(comparable)
