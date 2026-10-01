from __future__ import annotations

from evlab.schema import Doc, Hit


def format_context(hits: list[Hit], docs_by_id: dict[str, Doc]) -> tuple[str, list[str]]:
    chunks: list[str] = []
    doc_ids: list[str] = []
    for index, hit in enumerate(hits, start=1):
        doc = docs_by_id[hit.doc_id]
        doc_ids.append(doc.doc_id)
        title = f"{doc.title}\n" if doc.title else ""
        chunks.append(f"[{index}] {title}{doc.text}")
    return "\n\n".join(chunks), doc_ids
