from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evlab.io import iter_jsonl, read_qrels
from evlab.schema import Query, Qrel


@dataclass(frozen=True)
class NQPlusPaths:
    nqplus_dir: Path
    queries_file: Path
    rag_queries_file: Path
    generation_candidates_file: Path
    qrels_file: Path


def resolve_nqplus_paths(nqplus_dir: str | Path) -> NQPlusPaths:
    root = Path(nqplus_dir)
    return NQPlusPaths(
        nqplus_dir=root,
        queries_file=root / "queries.jsonl",
        rag_queries_file=root / "rag_queries.jsonl",
        generation_candidates_file=root / "generation_candidates.jsonl",
        qrels_file=root / "qrels" / "test.tsv",
    )


def load_retrieval_queries(nqplus_dir: str | Path) -> list[Query]:
    paths = resolve_nqplus_paths(nqplus_dir)
    qrels = read_qrels(paths.qrels_file)
    relevant = _relevant_by_query(qrels)
    queries: list[Query] = []
    for row in iter_jsonl(paths.queries_file):
        metadata = dict(row.get("metadata", {}) or {})
        aliases = tuple(str(item) for item in metadata.get("answer_aliases", []) or [])
        answer = metadata.get("answer")
        qid = str(row["_id"])
        queries.append(
            Query(
                query_id=qid,
                text=str(row["text"]),
                answer=str(answer) if answer is not None else None,
                aliases=aliases,
                relevant_doc_ids=tuple(relevant.get(qid, ())),
                metadata=metadata,
            )
        )
    return queries


def load_rag_queries(nqplus_dir: str | Path) -> list[Query]:
    paths = resolve_nqplus_paths(nqplus_dir)
    queries: list[Query] = []
    for row in iter_jsonl(paths.rag_queries_file):
        aliases = tuple(str(item) for item in row.get("correct_answer_aliases", []) or [])
        answer = str(row["correct_answer"])
        queries.append(
            Query(
                query_id=str(row["query_id"]),
                text=str(row["query"]),
                answer=answer,
                aliases=aliases or (answer,),
                relevant_doc_ids=tuple(str(item) for item in row.get("relevant_docs", []) or []),
                metadata=dict(row),
            )
        )
    return queries


def load_generation_candidates(nqplus_dir: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(resolve_nqplus_paths(nqplus_dir).generation_candidates_file))


def load_qrels(nqplus_dir: str | Path) -> list[Qrel]:
    return read_qrels(resolve_nqplus_paths(nqplus_dir).qrels_file)


def validate_nqplus(nqplus_dir: str | Path) -> dict[str, int]:
    paths = resolve_nqplus_paths(nqplus_dir)
    required = [
        paths.queries_file,
        paths.rag_queries_file,
        paths.generation_candidates_file,
        paths.qrels_file,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing NQ+ files: " + ", ".join(missing))
    retrieval = load_retrieval_queries(nqplus_dir)
    rag = load_rag_queries(nqplus_dir)
    generation = load_generation_candidates(nqplus_dir)
    qrels = load_qrels(nqplus_dir)
    qids = {query.query_id for query in retrieval}
    qrels_missing_queries = [qrel.query_id for qrel in qrels if qrel.query_id not in qids]
    if qrels_missing_queries:
        raise ValueError(f"Qrels reference missing queries: {qrels_missing_queries[:10]}")
    return {
        "retrieval_queries": len(retrieval),
        "rag_queries": len(rag),
        "generation_candidates": len(generation),
        "qrels": len(qrels),
    }


def _relevant_by_query(qrels: list[Qrel]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for qrel in qrels:
        if qrel.relevance > 0:
            grouped.setdefault(qrel.query_id, []).append(qrel.doc_id)
    return grouped
