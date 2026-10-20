"""Step 1: persist one false answer per eligible question in queries.jsonl."""

from __future__ import annotations

import fcntl
import os
import tempfile
from pathlib import Path

from evlab.data.nqplus import load_generation_candidates, resolve_nqplus_paths
from evlab.generation.false_answer import generate_false_answer, is_valid_false_answer
from evlab.generation.output import output_label
from evlab.generation.registry import update_registry
from evlab.io import append_jsonl, iter_jsonl, stable_hash, write_jsonl
from evlab.llm.cache import CachedLLMClient
from evlab.llm.client import build_llm_client


def _save_answers(path: Path, updates: dict[str, str], *, expected: dict | None = None, strict: bool = False) -> dict[str, str]:
    """Merge answers into current data; checkpoint with an atomic replacement."""
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = list(iter_jsonl(path))
        by_id = {str(row["_id"]): row for row in rows}
        saved = {}
        for qid, answer in updates.items():
            row = by_id[qid]
            metadata = row["metadata"]
            if expected is not None and (
                row["text"], metadata["answer"], metadata["answer_aliases"]
            ) != expected[qid]:
                raise ValueError(f"Question or gold answer changed while generating {qid}; retry with current data")
            if not is_valid_false_answer(answer, str(metadata["answer"]), list(metadata["answer_aliases"])):
                raise ValueError(f"Invalid false answer for {qid}: {answer!r}")
            existing = metadata.get("false_answer")
            if isinstance(existing, str) and is_valid_false_answer(existing, str(metadata["answer"]), list(metadata["answer_aliases"])):
                if strict and existing != answer:
                    raise ValueError(f"Conflicting existing and legacy false answers for {qid}")
                continue
            metadata["false_answer"] = answer
            saved[qid] = answer
        if saved:
            fd, temporary_name = tempfile.mkstemp(prefix=".queries-", suffix=".jsonl", dir=path.parent)
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                write_jsonl(temporary, rows)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        return saved


def _record_answers(output_dir: Path, queries_path: Path, answers: dict[str, str], *, model: str, temperature: float | None, origin: str) -> None:
    if not answers:
        return
    relative_path = os.path.relpath(queries_path.resolve(), output_dir.resolve())
    spec = {"path": relative_path, "field": "metadata.false_answer", "model": model,
            "temperature": temperature, "prompt_version": "v1", "origin": origin}
    key = "answers_" + stable_hash(spec)[:12]
    def change(registry: dict) -> None:
        jobs = registry.setdefault("answer_generations", {}) or {}
        registry["answer_generations"] = jobs
        previous = jobs.get(key, {})
        ids = sorted(set(previous.get("query_ids", [])) | answers.keys())
        jobs[key] = {**spec, "answer_count": len(ids), "query_ids": ids}
    update_registry(output_dir, change)


def import_legacy_false_answers(nqplus_dir: str | Path, output_dir: str | Path) -> int:
    """Recover saved targets from old metadata-rich docs, including backups."""
    queries_path = resolve_nqplus_paths(nqplus_dir).queries_file
    known = {str(row["_id"]): row for row in iter_jsonl(queries_path)}
    answers = {}
    by_model: dict[str, dict[str, str]] = {}
    for path in sorted(Path(output_dir).rglob("false_docs.jsonl")):
        for row in iter_jsonl(path):
            qid = str(row.get("query_id", ""))
            answer = row.get("target_answer")
            if qid not in known or not isinstance(answer, str) or not answer.strip():
                continue
            if qid in answers and answers[qid] != answer:
                raise ValueError(f"Legacy runs contain different false answers for {qid}; resolve before generating")
            answers[qid] = answer
            by_model.setdefault(str(row.get("generator_model", "unknown")), {})[qid] = answer
    saved = _save_answers(queries_path, answers, strict=True)
    for model, targets in by_model.items():
        _record_answers(Path(output_dir), queries_path, targets, model=model, temperature=None, origin="legacy_documents")
    return len(saved)


def run_false_answer_generation(config: dict, nqplus_dir: str | Path, *, limit: int | None = None) -> dict[str, int]:
    output_dir = Path(config.get("output_dir", "data/generated/pilot"))
    imported = import_legacy_false_answers(nqplus_dir, output_dir)
    path = resolve_nqplus_paths(nqplus_dir).queries_file
    candidates = load_generation_candidates(nqplus_dir)
    if limit is not None:
        candidates = candidates[:limit]
    queries = {str(row["_id"]): row for row in iter_jsonl(path)}
    temperature = float(config.get("temperature", 0.3))
    model = str(config["llm"]["model"])
    label = output_label(model, temperature, "candidates")
    rejects_path = output_dir / f"{label}--false-answer-rejects.jsonl"
    reject_keys = {str(row["reject_key"]) for row in iter_jsonl(rejects_path)} if rejects_path.exists() else set()
    client = None
    generated = reused = rejects = 0
    for idx, candidate in enumerate(candidates):
        qid = str(candidate["query_id"])
        row = queries[qid]
        metadata = row["metadata"]
        gold = str(metadata["answer"])
        aliases = list(metadata["answer_aliases"])
        existing = metadata.get("false_answer")
        if isinstance(existing, str) and is_valid_false_answer(existing, gold, aliases):
            reused += 1
            continue
        if client is None:
            client = CachedLLMClient(build_llm_client(config["llm"]), config.get("cache_path", "cache/llm.sqlite"))
        try:
            answer = generate_false_answer(client, question=row["text"], answer=gold, aliases=aliases,
                                           temperature=temperature, seed=idx)
            saved = _save_answers(path, {qid: answer}, expected={qid: (row["text"], metadata["answer"], metadata["answer_aliases"])})
            _record_answers(output_dir, path, saved, model=client.model, temperature=temperature, origin="generated")
            generated += len(saved)
            reused += int(not saved)
        except Exception as exc:  # Preserve successful answers if another question fails.
            rejects += 1
            reject = {"query_id": qid, "stage": "false_answer", "reason": str(exc)}
            key = stable_hash(reject)
            if key not in reject_keys:
                append_jsonl(rejects_path, [{**reject, "reject_key": key}])
                reject_keys.add(key)
        every = int(config.get("progress_every", 25))
        if every > 0 and (idx + 1) % every == 0:
            print(f"false answers: questions={idx + 1} generated={generated} reused={reused} rejects={rejects}", flush=True)
    return {"generated": generated, "reused": reused, "imported": imported, "rejects": rejects}
