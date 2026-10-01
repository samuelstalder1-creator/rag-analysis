from __future__ import annotations

from pathlib import Path
from typing import Iterable

from evlab.io import iter_jsonl
from evlab.schema import Doc, Origin, namespaced_id


def load_corpus(path: str | Path, *, origin: Origin, namespace: str | None = None) -> list[Doc]:
    ns = namespace or origin
    docs: list[Doc] = []
    for row in iter_jsonl(path):
        original = str(row.get("_id") or row.get("id") or row.get("doc_id"))
        if original in {"", "None"}:
            raise ValueError(f"Corpus row without id in {path}: {row}")
        docs.append(
            Doc(
                doc_id=namespaced_id(ns, original),
                title=str(row.get("title", "") or ""),
                text=str(row.get("text", row.get("contents", "")) or ""),
                origin=origin,
                provenance_root_id=original,
                metadata=dict(row.get("metadata", {}) or {}),
            )
        )
    _validate_docs(docs, path)
    return docs


def docs_by_id(docs: Iterable[Doc]) -> dict[str, Doc]:
    return {doc.doc_id: doc for doc in docs}


def _validate_docs(docs: list[Doc], path: str | Path) -> None:
    ids = [doc.doc_id for doc in docs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate document ids in {path}")
    empty = [doc.doc_id for doc in docs if not doc.index_text.strip()]
    if empty:
        raise ValueError(f"Empty documents in {path}: {empty[:10]}")
