from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evlab.generation.validate import contains_alias
from evlab.io import iter_jsonl, stable_hash
from evlab.schema import Query, Qrel


@dataclass(frozen=True)
class NQPlusPaths:
    nqplus_dir: Path
    queries_file: Path


def resolve_nqplus_paths(nqplus_dir: str | Path) -> NQPlusPaths:
    root = Path(nqplus_dir)
    return NQPlusPaths(nqplus_dir=root, queries_file=root / "queries.jsonl")


def load_query_rows(nqplus_dir: str | Path) -> list[dict]:
    """Join generated labels without changing the source query records on disk."""
    root = Path(nqplus_dir)
    labels_path = root / "generated_labels.jsonl"
    labels = {}
    if labels_path.exists():
        for label in iter_jsonl(labels_path):
            qid = str(label["_id"])
            if qid in labels:
                raise ValueError(f"Duplicate generated label ID: {qid}")
            labels[qid] = label
    rows = list(iter_jsonl(root / "queries.jsonl"))
    for row in rows:
        label = labels.get(str(row["_id"]))
        if label is not None:
            if label["source_hash"] != stable_hash(row):
                raise ValueError(f"Stale generated labels for {row['_id']}; rerun data classify-answers")
            row["metadata"] = {**row.get("metadata", {}), **label["metadata"]}
    return rows


def _load_queries(nqplus_dir: str | Path, usage: str) -> list[Query]:
    queries = []
    seen = set()
    for row in load_query_rows(nqplus_dir):
        qid = str(row["_id"])
        if qid in seen:
            raise ValueError(f"Duplicate query ID: {qid}")
        seen.add(qid)
        metadata = dict(row.get("metadata", {}) or {})
        answer = metadata.get("answer")
        default = True if usage == "retrieval" else bool(answer) if usage == "reader" else False
        if not metadata.get("usable_for", {}).get(usage, default):
            continue
        aliases = tuple(str(item) for item in metadata.get("answer_aliases", []) or [])
        queries.append(Query(
            query_id=qid, text=str(row["text"]), answer=str(answer) if answer is not None else None,
            aliases=aliases or ((str(answer),) if answer is not None else ()),
            relevant_doc_ids=tuple(dict.fromkeys(str(item) for item in metadata.get("relevant_docs", []))),
            metadata=metadata,
        ))
    return queries


def load_retrieval_queries(nqplus_dir: str | Path) -> list[Query]:
    return _load_queries(nqplus_dir, "retrieval")


def load_rag_queries(nqplus_dir: str | Path) -> list[Query]:
    return _load_queries(nqplus_dir, "reader")


def load_generation_queries(nqplus_dir: str | Path) -> list[Query]:
    return _load_queries(nqplus_dir, "generation")


def load_generation_candidates(
    nqplus_dir: str | Path, human_corpus: str | Path | None = None, *, include_text: bool = True,
) -> list[dict[str, Any]]:
    """Build candidates in memory from query annotations and original documents."""
    queries = load_generation_queries(nqplus_dir)
    candidates = []
    for query in queries:
        docs = list(dict.fromkeys(str(did) for did in query.metadata.get("docs_with_answer", [])))
        if not query.answer or not docs:
            raise ValueError(f"Generation query {query.query_id} needs an answer and docs_with_answer")
        if not set(docs) <= set(query.relevant_doc_ids):
            raise ValueError(f"Generation query {query.query_id} references answer documents outside relevant_docs")
        candidate = {"query_id": query.query_id, "query": query.text, "answer": query.answer,
                     "answer_aliases": list(query.aliases), "answer_kind": query.metadata.get("answer_kind", "single"),
                     "h_docs": [{"doc_id": did} for did in docs],
                     "other_relevant_docs": [did for did in query.relevant_doc_ids if did not in docs]}
        candidates.append(candidate)
    if include_text and candidates:
        root = Path(nqplus_dir)
        source_path = Path(human_corpus) if human_corpus is not None else root.parent.parent / "cocktail/nq/corpus/human.jsonl"
        needed = {doc["doc_id"] for candidate in candidates for doc in candidate["h_docs"]}
        sources = {}
        for row in iter_jsonl(source_path):
            did = str(row.get("_id", row.get("id", row.get("doc_id", ""))))
            if did in needed:
                if did in sources:
                    raise ValueError(f"Duplicate source document ID: {did}")
                sources[did] = row
        missing = needed - sources.keys()
        if missing:
            raise ValueError(f"Missing source documents in {source_path}: {sorted(missing)[:10]}")
        for candidate in candidates:
            for doc in candidate["h_docs"]:
                source = sources[doc["doc_id"]]
                doc.update(title=str(source.get("title", "") or ""), text=str(source["text"]))
                content = doc["title"] + "\n" + doc["text"]
                doc["aliases_in_doc"] = [alias for alias in candidate["answer_aliases"] if contains_alias(content, alias)]
    return candidates


def load_qrels(nqplus_dir: str | Path) -> list[Qrel]:
    """Prepared relevance annotations are stored only in queries.jsonl."""
    return [Qrel(query.query_id, did, 1) for query in load_retrieval_queries(nqplus_dir)
            for did in query.relevant_doc_ids]


def validate_nqplus(nqplus_dir: str | Path, *, human_corpus: str | Path | None = None) -> dict[str, int]:
    retrieval = load_retrieval_queries(nqplus_dir)
    rag = load_rag_queries(nqplus_dir)
    generation = load_generation_candidates(nqplus_dir, human_corpus, include_text=human_corpus is not None)
    return {"retrieval_queries": len(retrieval), "rag_queries": len(rag),
            "generation_candidates": len(generation), "qrels": len(load_qrels(nqplus_dir))}
