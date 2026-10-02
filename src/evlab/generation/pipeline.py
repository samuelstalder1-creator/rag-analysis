from __future__ import annotations

from pathlib import Path
from typing import Iterable

from evlab.generation.false_answer import generate_false_answer
from evlab.generation.minimal_edit import minimal_edit_with_fallback
from evlab.generation.paraphrase import paraphrase
from evlab.generation.validate import validate_correct_copy, validate_false_copy
from evlab.io import append_jsonl, iter_jsonl, stable_hash, write_json, write_jsonl
from evlab.llm.cache import CachedLLMClient
from evlab.llm.client import build_llm_client


def run_generation_pipeline(config: dict, candidates: Iterable[dict], *, limit: int | None = None) -> dict[str, int]:
    llm_config = config["llm"]
    client = CachedLLMClient(build_llm_client(llm_config), config.get("cache_path", "cache/llm.sqlite"))
    output_dir = Path(config.get("output_dir", "data/generated/pilot"))
    generated_path = output_dir / "generated_docs.jsonl"
    correct_path = output_dir / "correct_docs.jsonl"
    false_path = output_dir / "false_docs.jsonl"
    rejects_path = output_dir / "rejects.jsonl"
    progress_path = output_dir / "progress.json"
    temperature = float(config.get("temperature", 0.3))
    per_h_samples = int(config.get("samples_per_h", 1))
    progress_every = int(config.get("progress_every", 25))
    conditions = _conditions(config)
    resume = bool(config.get("resume", True))
    generated_ids = _existing_generated_ids([generated_path, correct_path, false_path]) if resume else set()
    reject_keys = _existing_reject_keys(rejects_path) if resume else set()
    if not resume:
        write_jsonl(generated_path, [])
        write_jsonl(correct_path, [])
        write_jsonl(false_path, [])
        write_jsonl(rejects_path, [])
    elif generated_path.exists():
        _ensure_split_outputs(generated_path, correct_path, false_path)
    generated_count = 0
    reject_count = 0
    processed_count = 0
    _write_progress(progress_path, status="running", candidates_seen=0, generated=0, rejects=0, conditions=conditions)

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
            target_answer = None
            if "false" in conditions:
                try:
                    target_answer = generate_false_answer(
                        client,
                        question=question,
                        answer=answer,
                        aliases=aliases,
                        temperature=temperature,
                        seed=idx,
                    )
                except Exception as exc:  # noqa: BLE001 - captured as experiment reject data
                    reject_count += _append_reject(
                        rejects_path,
                        {
                            "candidate_idx": idx,
                            "query_id": query_id,
                            "question": question,
                            "answer": answer,
                            "answer_aliases": aliases,
                            "stage": "false_answer",
                            "reason": str(exc),
                        },
                        reject_keys,
                    )
                    processed_count = idx + 1
                    _write_progress(
                        progress_path,
                        status="running",
                        candidates_seen=processed_count,
                        generated=generated_count,
                        rejects=reject_count,
                        conditions=conditions,
                        last_query_id=query_id,
                    )
                    if conditions == {"false"}:
                        continue

            for h_doc in candidate.get("h_docs", []):
                h_id = str(h_doc["doc_id"])
                h_title = str(h_doc.get("title", "") or "")
                h_text = str(h_doc["text"])
                aliases_in_doc = [str(item) for item in h_doc.get("aliases_in_doc", aliases)]
                false_seed = None
                if "false" in conditions and target_answer is not None:
                    false_seed = minimal_edit_with_fallback(
                        client,
                        text=h_text,
                        aliases=aliases_in_doc,
                        target_answer=target_answer,
                        seed=idx,
                    )
                for sample_idx in range(per_h_samples):
                    sample_seed = idx * 1000 + sample_idx
                    if "correct" in conditions:
                        correct_text = paraphrase(client, text=h_text, temperature=temperature, seed=sample_seed)
                        correct_validation = validate_correct_copy(correct_text, answer_aliases=aliases)
                        if correct_validation.ok:
                            generated_count += _append_generated(
                                generated_path,
                                correct_path,
                                false_path,
                                _row(
                                    prefix="AC",
                                    query_id=query_id,
                                    h_id=h_id,
                                    text=correct_text,
                                    condition="correct",
                                    answer=answer,
                                    target_answer=None,
                                    model=client.model,
                                    sample_idx=sample_idx,
                                ),
                                generated_ids,
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

                    if "false" in conditions and false_seed is not None and target_answer is not None:
                        false_text = paraphrase(client, text=false_seed, temperature=temperature, seed=sample_seed + 1)
                        false_validation = validate_false_copy(false_text, target_answer=target_answer, answer_aliases=aliases)
                        if false_validation.ok:
                            generated_count += _append_generated(
                                generated_path,
                                correct_path,
                                false_path,
                                _row(
                                    prefix="AF",
                                    query_id=query_id,
                                    h_id=h_id,
                                    text=false_text,
                                    condition="false",
                                    answer=answer,
                                    target_answer=target_answer,
                                    model=client.model,
                                    sample_idx=sample_idx,
                                ),
                                generated_ids,
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
                                    "reasons": list(false_validation.reasons),
                                    "answer": answer,
                                    "target_answer": target_answer,
                                    "answer_aliases": aliases,
                                    "aliases_in_doc": aliases_in_doc,
                                    "source_text": h_text,
                                    "false_seed_text": false_seed,
                                    "generated_text": false_text,
                                },
                                reject_keys,
                            )
            processed_count = idx + 1
            if progress_every > 0 and processed_count % progress_every == 0:
                _write_progress(
                    progress_path,
                    status="running",
                    candidates_seen=processed_count,
                    generated=generated_count,
                    rejects=reject_count,
                    conditions=conditions,
                    last_query_id=query_id,
                )
    except BaseException as exc:
        _write_progress(
            progress_path,
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            candidates_seen=processed_count,
            generated=generated_count,
            rejects=reject_count,
            conditions=conditions,
            error=repr(exc),
        )
        raise

    _write_progress(
        progress_path,
        status="complete",
        candidates_seen=processed_count,
        generated=generated_count,
        rejects=reject_count,
        conditions=conditions,
    )

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


def _write_progress(
    path: Path,
    *,
    status: str,
    candidates_seen: int,
    generated: int,
    rejects: int,
    conditions: set[str],
    last_query_id: str | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "status": status,
        "candidates_seen": candidates_seen,
        "generated": generated,
        "rejects": rejects,
        "conditions": sorted(conditions),
    }
    if last_query_id is not None:
        payload["last_query_id"] = last_query_id
    if error is not None:
        payload["error"] = error
    write_json(path, payload)


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
) -> dict:
    return {
        "doc_id": f"gen::{prefix}_{query_id}_{h_id}_{model.replace('/', '-')}_s{sample_idx:02d}_d1",
        "title": "",
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


def _existing_generated_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if path.exists():
            ids.update(str(row["doc_id"]) for row in iter_jsonl(path) if row.get("doc_id"))
    return ids


def _existing_reject_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("reject_key") or _reject_key(row)) for row in iter_jsonl(path)}


def _append_generated(generated_path: Path, correct_path: Path, false_path: Path, row: dict, generated_ids: set[str]) -> int:
    doc_id = str(row["doc_id"])
    if doc_id in generated_ids:
        return 0
    append_jsonl(generated_path, [row])
    append_jsonl(correct_path if row.get("condition") == "correct" else false_path, [row])
    generated_ids.add(doc_id)
    return 1


def _ensure_split_outputs(generated_path: Path, correct_path: Path, false_path: Path) -> None:
    existing = _existing_generated_ids([correct_path, false_path])
    correct_rows: list[dict] = []
    false_rows: list[dict] = []
    for row in iter_jsonl(generated_path):
        doc_id = str(row.get("doc_id", ""))
        if not doc_id or doc_id in existing:
            continue
        if row.get("condition") == "correct":
            correct_rows.append(row)
        elif row.get("condition") == "false":
            false_rows.append(row)
        existing.add(doc_id)
    if correct_rows:
        append_jsonl(correct_path, correct_rows)
    if false_rows:
        append_jsonl(false_path, false_rows)


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
