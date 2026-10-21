"""Step 1: persist false answers separately from immutable source queries."""

from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path

from evlab.data.nqplus import load_generation_candidates, resolve_nqplus_paths
from evlab.generation.false_answer import (
    generate_false_answer,
    canonical,
    is_valid_false_answer,
)
from evlab.generation.output import output_label
from evlab.generation.semantic import (
    build_verifier,
    verify_answer,
    VALIDATOR_VERSION,
    validator_signature,
    ModelOutputError,
    deterministic_receipt,
    receipt_mode,
    validation_mode,
)
from evlab.generation.false_answer import answer_type
from evlab.generation.registry import update_registry
from evlab.io import append_jsonl, iter_jsonl, stable_hash, write_jsonl
from evlab.llm.cache import CachedLLMClient
from evlab.llm.client import build_llm_client


def answers_path(config: dict) -> Path:
    return (
        Path(config.get("output_dir", "data/generated/pilot"))
        / "generated_false_answers.jsonl"
    )


def answer_source_hash(question: str, gold: str, aliases: list[str]) -> str:
    return stable_hash(
        {"question": question, "answer": gold, "answer_aliases": aliases}
    )


def attach_saved_answers(candidates: list[dict], config: dict) -> list[dict]:
    path = answers_path(config)
    # Deterministic runs may use any receipt; semantic runs require judged answers.
    require_semantic = validation_mode(config) == "semantic"
    saved = {}
    if path.exists():
        for row in iter_jsonl(path):
            qid = str(row["_id"])
            if qid in saved:
                raise ValueError(f"Duplicate generated false answer ID: {qid}")
            saved[qid] = row
    for candidate in candidates:
        candidate.pop("false_answer", None)
        candidate.pop("false_answer_record", None)
        row = saved.get(str(candidate["query_id"]))
        if row is not None and row.get("source_hash") == answer_source_hash(
            candidate["query"], candidate["answer"], candidate["answer_aliases"]
        ):
            if (
                row.get("verification", {}).get("version") == VALIDATOR_VERSION
                and row.get("verification", {}).get("validator_hash")
                == validator_signature()
                and (
                    not require_semantic
                    or receipt_mode(row.get("verification", {})) == "semantic"
                )
            ):
                candidate["false_answer"] = row.get("false_answer")
                candidate["false_answer_record"] = row
    return candidates


def _save_answers(
    path: Path,
    updates: dict[str, str],
    *,
    expected: dict,
    source_path: Path,
    verifications: dict | None = None,
    extras: dict | None = None,
) -> dict[str, str]:
    """Checkpoint generated answers atomically; never write source queries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = list(iter_jsonl(path)) if path.exists() else []
        by_id = {str(row["_id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError("Duplicate generated false answer IDs")
        current = {str(row["_id"]): row for row in iter_jsonl(source_path)}
        saved = {}
        for qid, answer in updates.items():
            question, gold, aliases = expected[qid]
            source = current[qid]
            metadata = source["metadata"]
            if (
                source["text"],
                metadata["answer"],
                metadata["answer_aliases"],
            ) != expected[qid]:
                raise ValueError(
                    f"Question or gold answer changed while generating {qid}; retry with current data"
                )
            if not is_valid_false_answer(answer, gold, aliases):
                raise ValueError(f"Invalid false answer for {qid}: {answer!r}")
            fingerprint = answer_source_hash(question, gold, aliases)
            existing = by_id.get(qid, {})
            if (
                existing.get("verification", {}).get("version") == VALIDATOR_VERSION
                and existing.get("verification", {}).get("validator_hash")
                == validator_signature()
                and existing.get("verification", {}).get("sources_hash")
                == (verifications or {}).get(qid, {}).get("sources_hash")
                and existing.get("source_hash") == fingerprint
                and is_valid_false_answer(
                    existing.get("false_answer", ""), gold, aliases
                )
            ):
                continue
            if not verifications or qid not in verifications:
                raise ValueError(
                    "A verification receipt is required to save a false answer"
                )
            by_id[qid] = {
                "_id": qid,
                "false_answer": answer,
                "source_hash": fingerprint,
                "answer_type": answer_type(gold, question),
                "verification": verifications[qid],
                **((extras or {}).get(qid, {})),
            }
            saved[qid] = answer
        if saved:
            fd, temporary_name = tempfile.mkstemp(
                prefix=".generated-answers-", suffix=".jsonl", dir=path.parent
            )
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                write_jsonl(temporary, by_id.values())
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return saved


def _record_answers(
    output_dir: Path,
    queries_path: Path,
    answers: dict[str, str],
    *,
    model: str,
    temperature: float | None,
    origin: str,
) -> None:
    if not answers:
        return
    relative_path = os.path.relpath(queries_path.resolve(), output_dir.resolve())
    spec = {
        "path": relative_path,
        "field": "false_answer",
        "model": model,
        "temperature": temperature,
        "prompt_version": "v1",
        "origin": origin,
    }
    key = "answers_" + stable_hash(spec)[:12]

    def change(registry: dict) -> None:
        jobs = registry.setdefault("answer_generations", {}) or {}
        registry["answer_generations"] = jobs
        previous = jobs.get(key, {})
        ids = sorted(set(previous.get("query_ids", [])) | answers.keys())
        jobs[key] = {**spec, "answer_count": len(ids), "query_ids": ids}

    update_registry(output_dir, change)


def deterministic_answer_receipt(candidate: dict, target: str) -> dict:
    """Receipt for runs without an LLM judge; records the rule-based checks only."""
    evidence = {
        "question": candidate["query"],
        "gold": candidate["answer"],
        "aliases": candidate["answer_aliases"],
        "target": target,
        "sources": candidate["h_docs"],
    }
    checks = {
        "structurally_valid": is_valid_false_answer(
            target, candidate["answer"], candidate["answer_aliases"], question=candidate["query"]
        ),
        "gold_in_sources": any(doc.get("aliases_in_doc") for doc in candidate["h_docs"]),
    }
    receipt = deterministic_receipt(stage="false_answer", evidence=evidence, checks=checks)
    receipt["sources_hash"] = stable_hash(candidate["h_docs"])
    return receipt


def run_false_answer_generation(
    config: dict,
    nqplus_dir: str | Path,
    *,
    human_corpus: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    max_attempts = config.get("false_answer_max_attempts", 3)
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("false_answer_max_attempts must be a positive integer")
    output_dir = Path(config.get("output_dir", "data/generated/pilot"))
    source_path = resolve_nqplus_paths(nqplus_dir).queries_file
    path = answers_path(config)
    candidates = attach_saved_answers(
        load_generation_candidates(nqplus_dir, human_corpus), config
    )
    if limit is not None:
        candidates = candidates[:limit]
    temperature = float(config.get("temperature", 0.3))
    # Raise temperature on retries so the generator does not repeat itself.
    temperature_step = float(config.get("false_answer_retry_temperature_step", 0.0))
    n_candidates = config.get("false_answer_candidates_per_call", 1)
    if type(n_candidates) is not int or n_candidates < 1:
        raise ValueError("false_answer_candidates_per_call must be a positive integer")
    semantic = validation_mode(config) == "semantic"
    model = str(config["llm"]["model"])
    label = output_label(model, temperature, "candidates")
    rejects_path = output_dir / f"{label}--false-answer-rejects.jsonl"
    reject_keys = (
        {str(row["reject_key"]) for row in iter_jsonl(rejects_path)}
        if rejects_path.exists()
        else set()
    )
    client = verifier = None
    generated = reused = rejects = 0
    for idx, candidate in enumerate(candidates):
        qid = str(candidate["query_id"])
        gold = candidate["answer"]
        aliases = candidate["answer_aliases"]
        existing = candidate.get("false_answer")
        record = candidate.get("false_answer_record", {})
        evidence = {
            "question": candidate["query"],
            "gold": gold,
            "aliases": aliases,
            "target": existing,
            "sources": candidate["h_docs"],
        }
        if (
            isinstance(existing, str)
            and is_valid_false_answer(existing, gold, aliases)
            and record.get("verification", {}).get("evidence_hash")
            == stable_hash(evidence)
        ):
            reused += 1
            continue
        if client is None:
            client = CachedLLMClient(
                build_llm_client(config["llm"]),
                config.get("cache_path", "cache/llm.sqlite"),
            )
            verifier = build_verifier(config) if semantic else None
        try:
            rejected_attempts = []
            rejected_targets = set()
            # Every proposal from every call, with its rule-based status, so a later
            # selection step can choose among all valid candidates.
            candidate_pool: list[dict] = []
            for attempt in range(max_attempts):
                details: dict = {}
                try:
                    answer = generate_false_answer(
                        client,
                        question=candidate["query"],
                        answer=gold,
                        aliases=aliases,
                        temperature=min(1.0, temperature + temperature_step * attempt),
                        seed=idx + attempt * 1000000,
                        rejected_attempts=rejected_attempts,
                        n_candidates=n_candidates,
                        exclude=rejected_targets,
                        details=details,
                        sources=[
                            f"{doc.get('title', '')}\n{doc.get('text', '')}"
                            for doc in candidate["h_docs"]
                        ],
                    )
                except ModelOutputError as exc:
                    candidate_pool.extend(
                        {**item, "attempt": attempt + 1} for item in details.get("proposals", [])
                    )
                    rejected_attempts.append(
                        {
                            "attempt": attempt + 1,
                            "stage": "generation",
                            **exc.attempts[0],
                        }
                    )
                    continue
                if canonical(answer) in rejected_targets:
                    rejected_attempts.append(
                        {
                            "attempt": attempt + 1,
                            "stage": "generation",
                            "target": answer,
                            "error": "Repeated previously rejected target",
                        }
                    )
                    continue
                candidate_pool.extend(
                    {**item, "attempt": attempt + 1} for item in details.get("proposals", [])
                )
                rejected_targets.add(canonical(answer))
                try:
                    verification = (
                        verify_answer(verifier, candidate, answer)
                        if semantic
                        else deterministic_answer_receipt(candidate, answer)
                    )
                    break
                except ModelOutputError as exc:
                    for item in candidate_pool:
                        if canonical(item["text"]) == canonical(answer):
                            item["status"] = "rejected_by_judge"
                    rejected_attempts.append(
                        {
                            "attempt": attempt + 1,
                            "stage": "validation",
                            "target": answer,
                            "reason": str(exc),
                            "attempts": exc.attempts,
                        }
                    )
            else:
                raise ModelOutputError(
                    f"No accepted false answer after {max_attempts} generation attempts",
                    rejected_attempts,
                )
            valid_pool = list(dict.fromkeys(
                item["text"] for item in candidate_pool if item["status"] == "valid"
            ))
            extras = {
                "candidates": candidate_pool,
                "valid_candidates": valid_pool,
                "selection": {
                    # Placeholder choice until a selector picks the best candidate.
                    "method": "first_valid_in_model_order",
                    "selected": answer,
                    "selector_model": None,
                },
            }
            verification["rejected_candidates"] = rejected_attempts
            verification["generation_attempts"] = attempt + 1
            verification["candidates_per_call"] = n_candidates
            saved = _save_answers(
                path,
                {qid: answer},
                expected={qid: (candidate["query"], gold, aliases)},
                source_path=source_path,
                verifications={qid: verification},
                extras={qid: extras},
            )
            _record_answers(
                output_dir,
                path,
                saved,
                model=client.model,
                temperature=temperature,
                origin="generated",
            )
            generated += len(saved)
            reused += int(not saved)
        except (
            Exception
        ) as exc:  # Preserve successful answers if another question fails.
            rejects += 1
            reject = {
                "query_id": qid,
                "stage": "false_answer",
                "reason": str(exc),
                "attempts": getattr(exc, "attempts", []),
            }
            key = stable_hash(reject)
            if key not in reject_keys:
                append_jsonl(rejects_path, [{**reject, "reject_key": key}])
                reject_keys.add(key)
        every = int(config.get("progress_every", 25))
        if every > 0 and (idx + 1) % every == 0:
            print(
                f"false answers: questions={idx + 1} generated={generated} reused={reused} rejects={rejects}",
                flush=True,
            )
    return {"generated": generated, "reused": reused, "rejects": rejects}
