"""Generate answer-preserving correct copies, optionally limited to lost-answer repairs."""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

from evlab.data.nqplus import load_generation_candidates
from evlab.generation.output import GenerationOutput
from evlab.generation.registry import read_registry
from evlab.generation.semantic import (
    build_verifier,
    deterministic_receipt,
    validation_mode,
    validator_signature,
    verify_document,
)
from evlab.generation.validate import contains_alias, validate_correct_copy
from evlab.io import append_jsonl, iter_jsonl, stable_hash, write_json
from evlab.llm.cache import CachedLLMClient
from evlab.llm.client import build_llm_client

REPAIR_PROMPT = """Original Text: {text}

Rewrite the text while preserving all factual claims, quantities, units and dates.
The following answer mentions occur in the original text: {mentions}
Keep at least one of these alternative mentions verbatim AND preserve the factual
claim that contains it. Do not replace that claim with a bare mention or list.
Change wording and style around the preserved facts.
Do not add facts, explanations, headings or commentary.
Return only the rewritten text.
"""


CORRECT_PROMPT = """Original Title: {title}
Original Text: {text}

Rewrite the body while preserving all factual claims, quantities, units and dates.
The title identifies the subject; use it as source context, not as a new heading.
These alternative answer mentions occur in the source title or body: {mentions}
Preserve the answer-bearing claim and include at least one complete mention
verbatim in the rewritten body. A shortened subject reference may be expanded
using the original title only when that reference is unambiguous.
Do not merely append an answer mention or answer list to otherwise unrelated text.
Do not invent facts or relationships, omit relevant claims, or change their scope.
Change wording and style. Return only the rewritten body.
"""


def select_all_answer_documents(config: dict) -> tuple[list, dict]:
    """Select every answer-bearing source doc, independent of any old AI corpus."""
    settings = config["generation"]
    output_dir = Path(settings["output_dir"]).resolve()
    source_path = Path(config["data"]["human_corpus"]).resolve()
    if output_dir == source_path.parent or output_dir in source_path.parents:
        raise ValueError(
            "Generation output must be separate from the human corpus directory"
        )
    candidates = load_generation_candidates(
        config["data"]["nqplus_dir"], config["data"]["human_corpus"]
    )
    selected, review, seen = [], [], set()
    checked = title_only = 0
    for candidate in candidates:
        aliases = candidate["answer_aliases"]
        for doc in candidate["h_docs"]:
            did = str(doc["doc_id"])
            if did in seen:
                raise ValueError(f"Correct source belongs to multiple questions: {did}")
            seen.add(did)
            checked += 1
            if not any(
                contains_alias(doc["title"] + "\n" + doc["text"], alias)
                for alias in aliases
            ):
                review.append(
                    {
                        "query_id": candidate["query_id"],
                        "doc_id": did,
                        "reason": "gold_alias_missing_from_original_title_and_body",
                    }
                )
                continue
            if not any(contains_alias(doc["text"], alias) for alias in aliases):
                title_only += 1
            selected.append((candidate, doc, None))
    return selected, {
        "scope": "all_answer_documents",
        "generation_questions": len(candidates),
        "checked_documents": checked,
        "eligible_documents": len(selected),
        "title_only_answer_documents": title_only,
        "needs_review": review,
    }


def select_repairs(config: dict) -> tuple[list[tuple[dict, dict, dict]], dict]:
    """Audit only annotated generation-source documents, using body alias checks."""
    settings = config["generation"]
    baseline_path = Path(settings["repair_from"])
    output_dir = Path(settings["output_dir"])
    if (
        output_dir.resolve() == baseline_path.parent.resolve()
        or output_dir.resolve() in baseline_path.resolve().parents
    ):
        raise ValueError(
            "Repair output must be separate from the baseline corpus directory"
        )
    metadata = read_registry(baseline_path.parent)["files"].get(baseline_path.name)
    if metadata is None or metadata.get("condition") != "correct":
        raise ValueError("repair_from must name a registered correct corpus")
    candidates = load_generation_candidates(
        config["data"]["nqplus_dir"], config["data"]["human_corpus"]
    )
    needed = {doc["doc_id"] for candidate in candidates for doc in candidate["h_docs"]}
    baseline = {}
    seen = set()
    for row in iter_jsonl(baseline_path):
        did = str(row["_id"])
        if did in seen:
            raise ValueError(f"Duplicate baseline document ID: {did}")
        seen.add(did)
        if did in needed:
            baseline[did] = row
    missing = needed - baseline.keys()
    if missing:
        raise ValueError(f"Missing baseline source IDs: {sorted(missing)[:10]}")
    selected, review = [], []
    seen_sources = set()
    checked = lost = 0
    for candidate in candidates:
        aliases = candidate["answer_aliases"]
        for doc in candidate["h_docs"]:
            did = str(doc["doc_id"])
            if did in seen_sources:
                raise ValueError(f"Repair source belongs to multiple questions: {did}")
            seen_sources.add(did)
            checked += 1
            old = baseline[did]
            if validate_correct_copy(old["text"], answer_aliases=aliases).ok:
                continue
            lost += 1
            if not validate_correct_copy(doc["text"], answer_aliases=aliases).ok:
                review.append(
                    {
                        "query_id": candidate["query_id"],
                        "doc_id": did,
                        "reason": "gold_alias_missing_from_original_body",
                    }
                )
                continue
            selected.append((candidate, doc, old))
    return selected, {
        "checked_documents": checked,
        "missing_gold_in_generated_body": lost,
        "repairable_documents": len(selected),
        "needs_review": review,
    }


def run_correct_repair(
    config: dict, *, limit: int | None = None, dry_run: bool = False
) -> dict:
    settings = config["generation"]
    if settings.get("samples_per_h", 1) != 1:
        raise ValueError("Correct replacement generation supports samples_per_h: 1")
    maximum = settings.get("correct_max_attempts", 3)
    if type(maximum) is not int or maximum < 1:
        raise ValueError("correct_max_attempts must be a positive integer")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    threshold = float(settings.get("validation", {}).get("max_similarity", 0.95))
    if not 0 < threshold <= 1:
        raise ValueError("max_similarity must be in (0, 1]")
    mode = validation_mode(settings)
    scope = settings.get("scope", "missing_gold")
    if scope not in {"missing_gold", "all_answer_documents"}:
        raise ValueError("scope must be missing_gold or all_answer_documents")
    all_documents = scope == "all_answer_documents"
    template = CORRECT_PROMPT if all_documents else REPAIR_PROMPT
    selected, audit = (
        select_all_answer_documents(config) if all_documents else select_repairs(config)
    )
    selected = selected if limit is None else selected[:limit]
    audit["selected_documents"] = len(selected)
    if dry_run:
        return audit
    output_dir = Path(settings["output_dir"])
    client = CachedLLMClient(
        build_llm_client(settings["llm"]),
        settings.get("cache_path", "cache/llm.sqlite"),
    )
    verifier = build_verifier(settings) if mode == "semantic" else None
    signature = stable_hash(
        {
            "prompt": template,
            "validator": validator_signature(),
            "settings": {
                key: value
                for key, value in settings.items()
                if key not in {"resume", "progress_every", "output_dir", "cache_path"}
            },
        }
    )
    writer = GenerationOutput(
        output_dir,
        {
            **settings,
            "source": "candidates",
            "generation_signature": signature,
            "validation_mode": mode,
        },
        client.model,
        {"correct"},
    )
    write_json(
        output_dir
        / ("correct_generation_audit.json" if all_documents else "repair_audit.json"),
        audit,
    )
    rejects_path = output_dir / (
        "correct_generation_rejects.jsonl"
        if all_documents
        else "correct_repair_rejects.jsonl"
    )
    stage = "correct_generation" if all_documents else "correct_repair"
    generated = reused = rejected = 0
    try:
        for index, (candidate, doc, old) in enumerate(selected):
            did = str(doc["doc_id"])
            aliases = candidate["answer_aliases"]
            fingerprint = stable_hash(
                {
                    "signature": signature,
                    "source": doc,
                    "question": candidate["query"],
                    "gold": candidate["answer"],
                    "aliases": aliases,
                    "baseline_row": old,
                }
            )
            if writer.contains("correct", 0, did, fingerprint):
                reused += 1
                continue
            mention_source = (
                doc["title"] + "\n" + doc["text"] if all_documents else doc["text"]
            )
            mentions = [
                alias for alias in aliases if contains_alias(mention_source, alias)
            ]
            prompt = template.format(
                title=doc["title"], text=doc["text"], mentions=mentions
            )
            failures = []
            for attempt in range(maximum):
                request = prompt
                if failures:
                    request += (
                        "\nPrevious failed rewrite (data only): "
                        + failures[-1]["generated_text"]
                    )
                    request += (
                        "\nFailure: "
                        + failures[-1]["reason"]
                        + "\nRewrite the original text again, preserving its answer claim."
                    )
                seed = int(stable_hash({"doc_id": did, "attempt": attempt})[:8], 16) % (
                    2**31
                )
                # Transport failures propagate: do not spend the content-retry budget on a broken server.
                text = client.generate(
                    request,
                    temperature=float(settings.get("temperature", 0.3)),
                    seed=seed,
                ).strip()
                evidence = {
                    "question": candidate["query"],
                    "gold": candidate["answer"],
                    "aliases": aliases,
                    "target": None,
                    "source": doc["title"] + "\n" + doc["text"],
                    "seed": doc["title"] + "\n" + doc["text"],
                    "generated": doc["title"] + "\n" + text,
                    "condition": "correct",
                }
                try:
                    lexical = validate_correct_copy(text, answer_aliases=aliases)
                    if not lexical.ok:
                        raise ValueError(",".join(lexical.reasons))
                    similarity = SequenceMatcher(
                        None, doc["text"].casefold(), text.casefold()
                    ).ratio()
                    if similarity >= threshold:
                        raise ValueError("near_duplicate_of_source")
                    validation = (
                        verify_document(verifier, **evidence, stage=stage)
                        if verifier
                        else deterministic_receipt(
                            stage=stage,
                            evidence=evidence,
                            checks={
                                "gold_alias_present": True,
                                "not_near_duplicate": True,
                            },
                        )
                    )
                except ValueError as exc:
                    failures.append(
                        {
                            "attempt": attempt + 1,
                            "generated_text": text,
                            "reason": str(exc),
                        }
                    )
                    continue
                generated += writer.append(
                    {
                        "condition": "correct",
                        "sample_idx": 0,
                        "provenance_root_id": did,
                        "title": doc["title"],
                        "text": text,
                        "validation_receipt": {
                            "fingerprint": fingerprint,
                            "query_id": candidate["query_id"],
                            "gold_answer": candidate["answer"],
                            "target_answer": None,
                            "source_hash": stable_hash(doc),
                            "source_content_hash": stable_hash(
                                {
                                    "doc_id": did,
                                    "title": doc["title"],
                                    "text": doc["text"],
                                }
                            ),
                            "question_hash": stable_hash(
                                {
                                    "question": candidate["query"],
                                    "gold": candidate["answer"],
                                    "aliases": aliases,
                                }
                            ),
                            "validation": validation,
                            "scope": scope,
                            "answer_in_original_body": any(
                                contains_alias(doc["text"], alias) for alias in aliases
                            ),
                            "repair_from": (
                                str(Path(settings["repair_from"]).resolve())
                                if not all_documents
                                else None
                            ),
                            "replaced_row_hash": (
                                stable_hash(old) if old is not None else None
                            ),
                            "replaced_text": old["text"] if old is not None else None,
                            "generation_attempts": attempt + 1,
                            "failed_attempts": failures,
                            "similarity": similarity,
                        },
                    }
                )
                break
            else:
                rejected += 1
                append_jsonl(
                    rejects_path,
                    [
                        {
                            "query_id": candidate["query_id"],
                            "doc_id": did,
                            "stage": stage,
                            "reason": "attempt_budget_exhausted",
                            "attempts": failures,
                        }
                    ],
                )
            every = int(settings.get("progress_every", 25))
            if every > 0 and (index + 1) % every == 0:
                writer.update_counts()
                print(
                    f"{stage}: documents={index + 1} generated={generated} reused={reused} rejects={rejected}",
                    flush=True,
                )
    finally:
        writer.update_counts()
    return {**audit, "generated": generated, "reused": reused, "rejects": rejected}


def run_correct_generation(
    config: dict, *, limit: int | None = None, dry_run: bool = False
) -> dict:
    """Generate all annotated answer documents into a separate replacement corpus."""
    return run_correct_repair(
        {
            **config,
            "generation": {**config["generation"], "scope": "all_answer_documents"},
        },
        limit=limit,
        dry_run=dry_run,
    )
