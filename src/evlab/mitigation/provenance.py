from __future__ import annotations

from evlab.schema import Doc, Hit


def max_one_per_root(hits: list[Hit], docs_by_id: dict[str, Doc], *, top_k: int) -> list[Hit]:
    seen: set[str] = set()
    selected: list[Hit] = []
    for hit in hits:
        doc = docs_by_id.get(hit.doc_id)
        root = doc.root_id if doc else hit.doc_id
        if root in seen:
            continue
        seen.add(root)
        selected.append(hit)
        if len(selected) == top_k:
            break
    return _rerank(selected)


def _rerank(hits: list[Hit]) -> list[Hit]:
    return [
        Hit(query_id=hit.query_id, doc_id=hit.doc_id, rank=rank, score=hit.score, retriever=hit.retriever)
        for rank, hit in enumerate(hits, start=1)
    ]
