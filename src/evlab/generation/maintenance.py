"""Reset false generation and inspect coverage against the original dataset."""

from __future__ import annotations

from collections import Counter

import os
import tempfile
from pathlib import Path
from evlab.data.nqplus import load_generation_candidates, load_query_rows
from evlab.generation.false_answers import attach_saved_answers

from evlab.generation.false_answer import is_valid_false_answer
from evlab.generation.output import CORPUS_FILENAME
from evlab.generation.registry import read_registry, update_registry
from evlab.io import iter_jsonl, read_qrels, write_jsonl


def _rewrite(path: Path, rows: list[dict]) -> None:
    fd, name = tempfile.mkstemp(prefix=".cleanup-", suffix=".jsonl", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        write_jsonl(temporary, rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _strip_false_answers(value: object) -> int:
    removed = 0
    if isinstance(value, dict):
        for key in list(value):
            if key in {"false_answer", "false_answers", "target_answer"} or key.startswith("false_answer_"):
                del value[key]
                removed += 1
            else:
                removed += _strip_false_answers(value[key])
    elif isinstance(value, list):
        removed += sum(_strip_false_answers(item) for item in value)
    return removed


def cleanup_false_generation(config: dict) -> dict:
    """Delete only derived false artifacts; retain original Cocktail data."""
    generation = config.get("generation", {})
    output_dir = Path(generation.get("output_dir", "data/generated/pilot"))
    nqplus_dir = Path(config["data"]["nqplus_dir"])
    cache_path = Path(generation.get("cache_path", "cache/llm.sqlite"))
    if cache_path.suffix not in {".sqlite", ".db"}:
        raise ValueError(f"Refusing to delete a non-SQLite generation cache: {cache_path}")
    human_path = config["data"].get("human_corpus")
    if human_path:
        original_dir = Path(human_path).resolve().parent.parent
        if output_dir.resolve() == original_dir or output_dir.resolve() in original_dir.parents:
            raise ValueError("Generation output directory must not contain the original Cocktail dataset")
        if nqplus_dir.resolve() == original_dir or original_dir in nqplus_dir.resolve().parents:
            raise ValueError("Prepared NQ+ data must be separate from the original Cocktail dataset")
    registry = read_registry(output_dir)
    removed_fields = 0
    changed_files = []
    deleted_files = []
    removed_false_document_rows = 0
    false_files = {name for name, metadata in registry["files"].items() if metadata["condition"] == "false"}
    for path in sorted(output_dir.rglob("*.jsonl")):
        match = CORPUS_FILENAME.fullmatch(path.name)
        is_false_corpus = (path.parent == output_dir and path.name in false_files) or (
            match is not None and match["condition"] == "false"
        ) or path.name == "false_docs.jsonl"
        if is_false_corpus or path.name == "generated_false_answers.jsonl" or path.name.endswith("false-answer-rejects.jsonl"):
            if is_false_corpus:
                removed_false_document_rows += sum(1 for _ in iter_jsonl(path))
            path.unlink()
            deleted_files.append(str(path))
            continue
        if path.name not in {"generated_docs.jsonl", "correct_docs.jsonl"} and not path.name.endswith("rejects.jsonl"):
            continue
        rows = list(iter_jsonl(path))
        kept = []
        stripped = 0
        for row in rows:
            false_row = row.get("condition") == "false" or str(row.get("stage", "")).startswith("false")
            if false_row:
                if path.name == "generated_docs.jsonl":
                    removed_false_document_rows += 1
                continue
            stripped += _strip_false_answers(row)
            kept.append(row)
        if len(kept) != len(rows) or stripped:
            if kept:
                _rewrite(path, kept)
                changed_files.append(str(path))
            else:
                path.unlink()
                deleted_files.append(str(path))

    def clean_registry(current: dict) -> None:
        for name in list(current["files"]):
            metadata = current["files"][name]
            if metadata["condition"] == "false" or not (output_dir / name).exists():
                del current["files"][name]
            else:
                metadata["document_count"] = sum(1 for _ in iter_jsonl(output_dir / name))
        current.pop("answer_generations", None)
    update_registry(output_dir, clean_registry)
    # This cache can contain both generated targets and paraphrased false text.
    # Clear the configured generation cache so a fresh run cannot reuse them.
    for path in (cache_path, Path(str(cache_path) + "-wal"), Path(str(cache_path) + "-shm"), Path(str(cache_path) + "-journal")):
        if path.exists():
            path.unlink()
            deleted_files.append(str(path))
    return {"false_answer_fields_removed": removed_fields, "false_document_rows_removed": removed_false_document_rows,
            "deleted_files": deleted_files, "updated_files": changed_files,
            "remaining_registered_files": sorted(read_registry(output_dir)["files"])}


def _inventory(path: Path) -> dict:
    ids = set()
    count = duplicates = empty = 0
    for row in iter_jsonl(path):
        did = str(row.get("_id", row.get("doc_id", row.get("id", ""))))
        count += 1
        duplicates += int(did in ids)
        ids.add(did)
        empty += int(not str(row.get("text", "")).strip())
    return {"ids": ids, "rows": count, "duplicate_ids": duplicates, "empty_text": empty}


def generation_status(config: dict) -> dict:
    data = config["data"]
    human_path = Path(data["human_corpus"])
    original_dir = human_path.parent.parent
    human = _inventory(human_path)
    original_queries = _inventory(original_dir / "queries.jsonl") if (original_dir / "queries.jsonl").exists() else None
    llm_path = Path(data.get("cocktail_llm_corpus", human_path.parent / "llama-2-7b-chat-tmp0.2.jsonl"))
    llm = _inventory(llm_path) if llm_path.exists() else None
    nqplus_dir = Path(data["nqplus_dir"])
    queries = load_query_rows(nqplus_dir)
    by_query = {str(row["_id"]): row for row in queries}
    candidates = attach_saved_answers(load_generation_candidates(nqplus_dir, include_text=False), config.get("generation", {}))
    candidate_ids = {str(doc["doc_id"]) for row in candidates for doc in row.get("h_docs", [])}
    valid_answers = set()
    for candidate in candidates:
        answer = candidate.get("false_answer")
        if isinstance(answer, str) and is_valid_false_answer(answer, candidate["answer"], candidate["answer_aliases"]):
            valid_answers.add(str(candidate["query_id"]))
    output_dir = Path(config.get("generation", {}).get("output_dir", "data/generated/pilot"))
    registry = read_registry(output_dir)
    files = []
    correct_ids = set()
    false_ids = set()
    for name, metadata in registry["files"].items():
        path = output_dir / name
        if not path.exists():
            files.append({"file": name, "error": "registered_file_missing"})
            continue
        inventory = _inventory(path)
        target = human["ids"] if metadata["source"] == "human_corpus" else candidate_ids
        ids = inventory.pop("ids")
        (correct_ids if metadata["condition"] == "correct" else false_ids).update(ids)
        files.append({"file": name, "model": metadata["model"], "condition": metadata["condition"],
                      "source": metadata["source"], **inventory, "expected": len(target),
                      "missing": len(target - ids), "unexpected": len(ids - target),
                      "metadata_count_matches": metadata["document_count"] == inventory["rows"]})
    original = {"directory": str(original_dir), "human_documents": human["rows"],
                "human_duplicate_ids": human["duplicate_ids"],
                "queries": original_queries["rows"] if original_queries else None,
                "provided_llm_documents": llm["rows"] if llm else None,
                "provided_llm_missing_human_ids": len(human["ids"] - llm["ids"]) if llm else None}
    if (original_dir / "qrels/test.tsv").exists():
        original["qrels"] = len(read_qrels(original_dir / "qrels/test.tsv"))
    return {"original_cocktail": original,
            "prepared_nqplus": {"queries": len(queries), "generation_questions": len(candidates),
                                "answer_kinds": dict(Counter(row.get("metadata", {}).get("answer_kind", "unknown") for row in queries)),
                                "flags": dict(Counter(flag for row in queries for flag, value in row.get("metadata", {}).get("flags", {}).items() if value)),
                                "query_ids_missing_from_original": len(set(by_query) - original_queries["ids"]) if original_queries else None,
                                "generation_source_documents": len(candidate_ids),
                                "source_ids_missing_from_original": len(candidate_ids - human["ids"]),
                                "saved_false_answers": len(valid_answers),
                                "missing_false_answers": len(candidates) - len(valid_answers)},
            "generated_files": files,
            "coverage": {"correct_human_generated": len(correct_ids & human["ids"]),
                         "correct_human_missing": len(human["ids"] - correct_ids),
                         "candidate_ids_with_correct_copy": len(correct_ids & candidate_ids),
                         "false_candidate_generated": len(false_ids & candidate_ids),
                         "false_candidate_missing": len(candidate_ids - false_ids)},
            "unregistered_corpus_files": sorted(path.name for path in output_dir.glob("corpus_*.jsonl") if path.name not in registry["files"]),
            "note": "Coverage is based on original document IDs. It does not certify the factual quality or provenance of generated text."}
