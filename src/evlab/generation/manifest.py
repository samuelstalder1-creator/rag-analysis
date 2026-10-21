"""Append-only validation receipts alongside compact corpus rows."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from evlab.io import stable_hash, append_jsonl


def manifest_path(corpus: Path) -> Path:
    return corpus.with_suffix(".manifest")


def read_manifest_delta(corpus: Path, offset: int = 0) -> tuple[dict, int]:
    path = manifest_path(corpus)
    if not path.exists():
        return {}, 0
    with path.open(encoding="utf-8") as handle:
        handle.seek(offset)
        rows = [json.loads(line) for line in handle.read().splitlines() if line.strip()]
        return {str(row["doc_id"]): row["receipt"] for row in rows}, handle.tell()


def read_manifest(corpus: Path) -> dict:
    return read_manifest_delta(corpus)[0]


def append_receipt(corpus: Path, doc_id: str, receipt: dict) -> int:
    # Caller holds the corpus lock. Receipts are written before corpus rows.
    append_jsonl(manifest_path(corpus), [{"doc_id": doc_id, "receipt": receipt}])
    return manifest_path(corpus).stat().st_size


def write_manifest(corpus: Path, manifest: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=".manifest-", dir=corpus.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for doc_id, receipt in manifest.items():
                handle.write(
                    json.dumps(
                        {"doc_id": doc_id, "receipt": receipt}, ensure_ascii=False
                    )
                    + "\n"
                )
        Path(name).replace(manifest_path(corpus))
    finally:
        Path(name).unlink(missing_ok=True)


def checked_receipt(row: dict, manifest: dict) -> dict:
    receipt = manifest.get(str(row["_id"]))
    if not receipt or receipt.get("row_hash") != stable_hash(row):
        raise ValueError(
            f"Missing or stale validation manifest for document {row['_id']}; use a new output directory or explicitly regenerate"
        )
    return receipt
