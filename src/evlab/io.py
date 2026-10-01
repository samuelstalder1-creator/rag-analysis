from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from evlab.schema import Qrel


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path(base_dir) if base_dir else Path.cwd()) / candidate


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_qrels(path: str | Path) -> list[Qrel]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter="\t") if row]
    if not rows:
        return []
    first = [cell.strip().lower() for cell in rows[0]]
    has_header = any(cell in {"query-id", "query_id", "qid", "corpus-id", "doc_id", "score"} for cell in first)
    data_rows = rows[1:] if has_header else rows
    qrels: list[Qrel] = []
    for row in data_rows:
        if len(row) >= 4:
            qid, did, score = row[0], row[2], row[3]
        elif len(row) >= 3:
            qid, did, score = row[0], row[1], row[2]
        else:
            raise ValueError(f"Unsupported qrels row in {path}: {row}")
        qrels.append(Qrel(str(qid), str(did), int(float(score))))
    return qrels


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
