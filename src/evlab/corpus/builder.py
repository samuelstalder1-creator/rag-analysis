from __future__ import annotations

from pathlib import Path
from glob import glob

from evlab.data.cocktail import load_corpus
from evlab.io import iter_jsonl, read_qrels
from evlab.generation.output import CORPUS_FILENAME
from evlab.generation.registry import read_registry
from evlab.schema import Doc


def load_generated_docs(path: str | Path, *, query_ids_by_doc: dict[str, str] | None = None) -> list[Doc]:
    candidate = Path(path)
    if candidate.is_dir():
        entries = read_registry(candidate)["files"]
        paths = sorted(candidate / name for name in entries)
        paths.extend(sorted(p for p in candidate.glob("*.jsonl") if CORPUS_FILENAME.fullmatch(p.name)))
    else:
        paths = [Path(p) for p in sorted(glob(str(path)))]
    if not paths:
        raise FileNotFoundError(f"No generated corpus files found: {path}")
    docs = []
    for file in paths:
        docs.extend(_load_generated_file(file, query_ids_by_doc or {}))
    return docs


def _load_generated_file(path: Path, query_ids_by_doc: dict[str, str]) -> list[Doc]:
    docs: list[Doc] = []
    file_metadata = read_registry(path.parent)["files"].get(path.name)
    legacy_filename = CORPUS_FILENAME.fullmatch(path.name)
    if file_metadata is None and legacy_filename is not None:
        file_metadata = {"model": legacy_filename["model"], "condition": legacy_filename["condition"],
                         "generation_depth": int(legacy_filename["depth"]),
                         "sample_idx": int(legacy_filename["sample"]), "prompt_version": "v1"}
    for row in iter_jsonl(path):
        if "_id" in row:
            if file_metadata is None:
                raise ValueError(f"Compact generated corpus is missing from metadata.yaml: {path}")
            root = str(row["_id"])
            condition = file_metadata["condition"]
            docs.append(Doc(
                doc_id=f"gen::{file_metadata.get('id_namespace', path.stem)}::{root}", title=str(row.get("title", "") or ""),
                text=str(row["text"]), origin="generated",
                query_id=query_ids_by_doc.get(root, root), condition=condition,
                parent_id=root if condition == "correct" else f"{root}*",
                seed_id=root if condition == "correct" else f"{root}*",
                provenance_root_id=root, generation_depth=int(file_metadata["generation_depth"]),
                generator_model=file_metadata["model"], prompt_version=file_metadata["prompt_version"],
                sample_idx=int(file_metadata["sample_idx"]),
            ))
            continue
        docs.append(
            Doc(
                doc_id=str(row["doc_id"]),
                title=str(row.get("title", "") or ""),
                text=str(row["text"]),
                origin="generated",
                query_id=_optional_str(row.get("query_id")),
                condition=row.get("condition"),  # type: ignore[arg-type]
                gold_answer=_optional_str(row.get("gold_answer")),
                target_answer=_optional_str(row.get("target_answer")),
                parent_id=_optional_str(row.get("parent_id")),
                seed_id=_optional_str(row.get("seed_id")),
                provenance_root_id=_optional_str(row.get("provenance_root_id")),
                generation_depth=int(row.get("generation_depth", 1)),
                generator_model=_optional_str(row.get("generator_model")),
                prompt_version=_optional_str(row.get("prompt_version")),
                sample_idx=int(row["sample_idx"]) if row.get("sample_idx") is not None else None,
                metadata=dict(row),
            )
        )
    return docs


def build_corpus(config: dict) -> list[Doc]:
    data = config["data"]
    corpus = config.get("corpus", {})
    base = str(corpus.get("base", "human"))
    docs = load_corpus(data["human_corpus"], origin="human", namespace="human")
    if base == "cocktail_mixed":
        docs.extend(load_corpus(data["cocktail_llm_corpus"], origin="cocktail_llm", namespace="cocktail_llm"))
    elif base != "human":
        raise ValueError(f"Unsupported base corpus: {base}")

    generated_file = corpus.get("generated_docs")
    if generated_file:
        query_ids_by_doc: dict[str, str] = {}
        for qrel in read_qrels(Path(data["nqplus_dir"]) / "qrels" / "test.tsv"):
            if qrel.relevance > 0:
                previous = query_ids_by_doc.setdefault(qrel.doc_id, qrel.query_id)
                if previous != qrel.query_id:
                    raise ValueError(f"Generated document {qrel.doc_id} belongs to multiple questions")
        docs.extend(select_generated_docs(
            load_generated_docs(generated_file, query_ids_by_doc=query_ids_by_doc), corpus
        ))
    return docs


def select_generated_docs(generated_docs: list[Doc], corpus_config: dict) -> list[Doc]:
    k_correct = int(corpus_config.get("k_correct", 0))
    k_false = int(corpus_config.get("k_false", 0))
    depth = int(corpus_config.get("generation_depth", 1))
    selected: list[Doc] = []
    grouped: dict[tuple[str, str], list[Doc]] = {}
    for doc in generated_docs:
        if doc.generation_depth != depth:
            continue
        if not doc.query_id or not doc.condition:
            continue
        grouped.setdefault((doc.query_id, doc.condition), []).append(doc)
    for (query_id, condition), docs in grouped.items():
        limit = k_correct if condition == "correct" else k_false
        selected.extend(sorted(docs, key=lambda doc: (doc.sample_idx or 0, doc.doc_id))[:limit])
    return selected


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
