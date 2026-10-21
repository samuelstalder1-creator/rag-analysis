"""Step 2: generate false documents using answers saved in generated_false_answers.jsonl."""

from __future__ import annotations

from pathlib import Path

from evlab.data.nqplus import load_generation_candidates
from evlab.generation.pipeline import run_generation_pipeline
from evlab.generation.false_answers import attach_saved_answers


def run_false_document_generation(config: dict, nqplus_dir: str | Path, *, human_corpus: str | Path | None = None, limit: int | None = None) -> dict[str, int]:
    if config.get("source", "candidates") != "candidates":
        raise ValueError("False-document generation requires NQ+ candidates with saved false answers")
    return run_generation_pipeline(
        {**config, "conditions": ["false"]}, attach_saved_answers(load_generation_candidates(nqplus_dir, human_corpus), config), limit=limit
    )
