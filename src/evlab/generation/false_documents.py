"""Step 2: generate false documents using answers already saved in queries.jsonl."""

from __future__ import annotations

from pathlib import Path

from evlab.data.nqplus import load_generation_candidates
from evlab.generation.pipeline import run_generation_pipeline


def run_false_document_generation(config: dict, nqplus_dir: str | Path, *, limit: int | None = None) -> dict[str, int]:
    if config.get("source", "candidates") != "candidates":
        raise ValueError("False-document generation requires NQ+ candidates with saved false answers")
    return run_generation_pipeline(
        {**config, "conditions": ["false"]}, load_generation_candidates(nqplus_dir), limit=limit
    )
