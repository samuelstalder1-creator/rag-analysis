"""Input fingerprints and run status for experiment runs.

Run IDs stay derived from the config (so existing result folders keep their
names). The fingerprints recorded here detect that a folder is reused with
different data, and the status file marks whether a folder holds a finished run.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evlab.io import file_sha256, stable_hash, write_json
from evlab.schema import Doc

# Bump when metric definitions change so old and new numbers are told apart.
EVALUATOR_VERSION = 2  # 2: context_recall@k added
STATUS_FILE = "run_status.json"
_DATASET_FILES = ("queries.jsonl", "answers.jsonl", "generated_labels.jsonl", "qrels/test.tsv")


class RunConflictError(RuntimeError):
    """The output folder holds a run made from different input data."""


def corpus_content_hash(docs: list[Doc]) -> str:
    """Hash over id *and* content of every document, independent of order."""
    digest = hashlib.sha256()
    for doc in sorted(docs, key=lambda item: item.doc_id):
        record = [doc.doc_id, doc.title, doc.text, doc.origin, doc.condition, doc.target_answer]
        digest.update(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataset_file_hashes(nqplus_dir: str | Path) -> dict[str, str]:
    root = Path(nqplus_dir)
    return {name: file_sha256(root / name) for name in _DATASET_FILES if (root / name).is_file()}


def code_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=root, capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {"git_sha": None, "dirty": None}
    return {"git_sha": sha, "dirty": dirty}


def compute_fingerprints(config: dict, docs: list[Doc], reader_spec: dict | None) -> dict[str, Any]:
    """``inputs`` decide whether two runs are comparable; ``code`` is informational."""
    reader = None
    if reader_spec:
        reader = {
            **({"context_order": reader_spec["context_order"]} if "context_order" in reader_spec else {}),
            "llm": reader_spec["llm"],
            "prompt_sha256": stable_hash(reader_spec["prompt_template"]),
            "top_k": reader_spec["top_k"],
            "temperature": reader_spec["temperature"],
            "seed": reader_spec["seed"],
        }
    return {
        "version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "config_hash": stable_hash(config),
        "inputs": {
            "dataset_files": dataset_file_hashes(config["data"]["nqplus_dir"]),
            "corpus": {"doc_count": len(docs), "content_sha256": corpus_content_hash(docs)},
            "reader": reader,
        },
        "code": code_version(),
    }


def changed_inputs(old: dict, new: dict) -> list[str]:
    """Names of fingerprint parts that differ (empty means compatible)."""
    differences = []
    if old.get("evaluator_version") != new.get("evaluator_version"):
        differences.append("evaluator_version")
    old_inputs, new_inputs = old.get("inputs", {}), new.get("inputs", {})
    for key in sorted(set(old_inputs) | set(new_inputs)):
        if old_inputs.get(key) != new_inputs.get(key):
            differences.append(f"inputs.{key}")
    return differences


def begin_run(output_root: Path, fingerprints: dict, *, allow_resume: bool = True) -> bool:
    """Refuse mixed-data reuse, drop stale final artifacts, mark the run as running.

    Returns True when answers already stored in the folder may be reused: the
    previous run recorded fingerprints and config identical to this one. Legacy
    folders without a status file are overwritten (not resumed), as before.
    """
    resumable = False
    status_path = output_root / STATUS_FILE
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8")).get("fingerprints")
        if previous:
            differences = changed_inputs(previous, fingerprints)
            if differences:
                raise RunConflictError(
                    f"{output_root} was produced from different inputs ({', '.join(differences)}). "
                    "Choose another output.dir or move the old folder away."
                )
            resumable = allow_resume and previous.get("config_hash") == fingerprints["config_hash"]
    (output_root / "summary.json").unlink(missing_ok=True)
    for stale in output_root.glob("*/metrics.json"):
        stale.unlink()
    write_json(
        status_path,
        {"status": "running", "started_at": _now(), "finished_at": None, "error": None,
         "fingerprints": fingerprints},
    )
    return resumable


def finish_run(output_root: Path, *, error: BaseException | None = None) -> None:
    status_path = output_root / STATUS_FILE
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        status="failed" if error else "complete",
        finished_at=_now(),
        error=f"{type(error).__name__}: {error}" if error else None,
    )
    write_json(status_path, status)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
