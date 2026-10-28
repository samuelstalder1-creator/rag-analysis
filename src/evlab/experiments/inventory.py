"""Inventory of stored runs: status, purpose (smoke/full) and completeness."""

from __future__ import annotations

import json
import re
from pathlib import Path

STATUS_FILE = "run_status.json"
_QUERY_ID = re.compile(r'"query_id": "([^"]*)"')


def find_run_dirs(root: Path) -> list[Path]:
    """Folders holding a ``config.json`` (outermost only)."""
    found: list[Path] = []
    for config_path in sorted(root.rglob("config.json")):
        directory = config_path.parent
        if any(parent in found for parent in directory.parents):
            continue
        found.append(directory)
    return found


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _query_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = _QUERY_ID.search(line)
            if match:
                ids.add(match.group(1))
    return ids


def classify_run(run_dir: Path, root: Path | None = None) -> dict:
    config = _read_json(run_dir / "config.json") or {}
    summary = _read_json(run_dir / "summary.json")
    status_file = _read_json(run_dir / STATUS_FILE)
    manifest = _read_json(run_dir / "manifest.json") or {}
    relative = run_dir.relative_to(root) if root else run_dir

    retrievers = {}
    for sub in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        retrieval, answers = sub / "retrieval.jsonl", sub / "answers.jsonl"
        if not retrieval.exists() and not answers.exists():
            continue
        retrievers[sub.name] = {
            "answers": _line_count(answers) if answers.exists() else 0,
            "has_metrics": (sub / "metrics.json").exists(),
            "retrieval_queries": (
                summary["num_queries"] if summary else len(_query_ids(retrieval))
            ) if retrieval.exists() else 0,
        }

    if status_file:
        status = status_file.get("status", "unknown")
    elif summary:
        status = "complete"  # legacy folder: no status file, but finished
    elif retrievers:
        status = "incomplete"
    else:
        status = "prepared"

    smoke = (
        bool(config.get("query_limit"))
        or any("smoke" in part.lower() for part in relative.parts)
        or str(config.get("experiment", "")).endswith("__smoke")
    )
    if summary:
        expected = summary["num_queries"]
    elif config.get("query_limit"):
        expected = int(config["query_limit"])
    else:
        # No summary: the retrieval output is the best evidence of the planned size.
        expected = max((info["retrieval_queries"] for info in retrievers.values()), default=None) or None
    reader = config.get("reader")
    reader_retrievers = None
    if reader:
        reader_retrievers = [name for name, info in retrievers.items() if info["answers"]]
    incomplete_reader = [
        name for name in (reader_retrievers or [])
        if expected is not None and retrievers[name]["answers"] < expected
    ]
    return {
        "path": str(relative),
        "experiment": config.get("experiment"),
        "status": status,
        "purpose": "smoke" if smoke else "full",
        "query_limit": config.get("query_limit"),
        "expected_queries": expected,
        "retrievers": retrievers,
        "reader": bool(reader),
        "incomplete_reader_outputs": incomplete_reader,
        "has_fingerprints": bool(manifest.get("fingerprints")),
    }


def build_inventory(root: str | Path) -> list[dict]:
    root = Path(root)
    return [classify_run(directory, root) for directory in find_run_dirs(root)]


def summarize(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row['purpose']}/{row['status']}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
