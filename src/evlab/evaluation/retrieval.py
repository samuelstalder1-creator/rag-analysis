from __future__ import annotations

import math

from evlab.schema import Doc, Hit, Qrel


def evaluate_retrieval(
    run: dict[str, list[Hit]],
    qrels: list[Qrel],
    *,
    docs_by_id: dict[str, Doc] | None = None,
    relevance_mode: str = "strict",
    cutoffs: tuple[int, ...] = (10, 100),
) -> dict[str, float]:
    relevant = relevant_by_query(qrels)
    query_ids = sorted(relevant)
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        metrics[f"recall@{cutoff}"] = _mean(
            [recall_at_k(relevant[qid], run.get(qid, []), cutoff, docs_by_id, relevance_mode) for qid in query_ids]
        )
        metrics[f"ndcg@{cutoff}"] = _mean(
            [ndcg_at_k(relevant[qid], run.get(qid, []), cutoff, docs_by_id, relevance_mode) for qid in query_ids]
        )
    metrics["mrr"] = _mean(
        [reciprocal_rank(relevant[qid], run.get(qid, []), docs_by_id, relevance_mode) for qid in query_ids]
    )
    return metrics


def relevant_by_query(qrels: list[Qrel]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for qrel in qrels:
        if qrel.relevance > 0:
            grouped.setdefault(qrel.query_id, {})[qrel.doc_id] = max(
                qrel.relevance,
                grouped.get(qrel.query_id, {}).get(qrel.doc_id, 0),
            )
    return grouped


def recall_at_k(
    relevant_docs: dict[str, int],
    hits: list[Hit],
    k: int,
    docs_by_id: dict[str, Doc] | None = None,
    relevance_mode: str = "strict",
) -> float:
    relevant_ids = {doc_id for doc_id, score in relevant_docs.items() if score > 0}
    if not relevant_ids:
        return 0.0
    retrieved = set(_collapsed_qrel_ids(hits, docs_by_id, relevance_mode)[:k])
    return len(relevant_ids & retrieved) / len(relevant_ids)


def ndcg_at_k(
    relevant_docs: dict[str, int],
    hits: list[Hit],
    k: int,
    docs_by_id: dict[str, Doc] | None = None,
    relevance_mode: str = "strict",
) -> float:
    if not relevant_docs:
        return 0.0
    dcg = 0.0
    for rank, mapped_doc_id in enumerate(_collapsed_qrel_ids(hits, docs_by_id, relevance_mode)[:k], start=1):
        rel = relevant_docs.get(mapped_doc_id, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(rank + 1)
    ideal = sorted((score for score in relevant_docs.values() if score > 0), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def reciprocal_rank(
    relevant_docs: dict[str, int],
    hits: list[Hit],
    docs_by_id: dict[str, Doc] | None = None,
    relevance_mode: str = "strict",
) -> float:
    for rank, mapped_doc_id in enumerate(_collapsed_qrel_ids(hits, docs_by_id, relevance_mode), start=1):
        if relevant_docs.get(mapped_doc_id, 0) > 0:
            return 1 / rank
    return 0.0


def _collapsed_qrel_ids(hits: list[Hit], docs_by_id: dict[str, Doc] | None, relevance_mode: str) -> list[str]:
    mapped: list[str] = []
    seen: set[str] = set()
    for hit in sorted(hits, key=lambda item: item.rank):
        qrel_id = _qrel_id(hit.doc_id, docs_by_id, relevance_mode)
        if qrel_id in seen:
            continue
        seen.add(qrel_id)
        mapped.append(qrel_id)
    return mapped


def _qrel_id(doc_id: str, docs_by_id: dict[str, Doc] | None, relevance_mode: str) -> str:
    doc = docs_by_id.get(doc_id) if docs_by_id else None
    if relevance_mode == "relaxed":
        return doc.root_id if doc else doc_id.split("::", 1)[-1]
    if doc and doc.origin == "human":
        return doc.root_id
    if not docs_by_id and doc_id.startswith("human::"):
        return doc_id.split("::", 1)[-1]
    return doc_id


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
