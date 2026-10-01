from __future__ import annotations

from evlab.schema import Doc, Hit


def simple_text_dedup(hits: list[Hit], docs_by_id: dict[str, Doc], *, top_k: int) -> list[Hit]:
    seen_texts: set[str] = set()
    selected: list[Hit] = []
    for hit in hits:
        doc = docs_by_id.get(hit.doc_id)
        fingerprint = " ".join((doc.index_text if doc else hit.doc_id).lower().split())
        if fingerprint in seen_texts:
            continue
        seen_texts.add(fingerprint)
        selected.append(hit)
        if len(selected) == top_k:
            break
    return [
        Hit(query_id=hit.query_id, doc_id=hit.doc_id, rank=rank, score=hit.score, retriever=hit.retriever)
        for rank, hit in enumerate(selected, start=1)
    ]
