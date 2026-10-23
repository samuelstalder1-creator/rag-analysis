"""Generate false documents from persisted targets, including imported nqopen+ targets."""

from __future__ import annotations

from pathlib import Path

from evlab.data.nqplus import load_generation_candidates
from evlab.generation.pipeline import run_generation_pipeline
from evlab.generation.false_answers import attach_saved_answers, valid_saved_answer
from evlab.generation.semantic import validation_mode


def run_false_document_generation(
    config: dict,
    nqplus_dir: str | Path,
    *,
    human_corpus: str | Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    retry_seed: int = 0,
) -> dict:
    if type(retry_seed) is not int or retry_seed < 0:
        raise ValueError("retry_seed must be a non-negative integer")
    if config.get("source", "candidates") != "candidates":
        raise ValueError(
            "False-document generation requires query candidates with persisted false answers"
        )
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    candidates = attach_saved_answers(
        load_generation_candidates(nqplus_dir, human_corpus), config
    )
    if dry_run:
        selected = candidates if limit is None else candidates[:limit]
        semantic = validation_mode(config) == "semantic"
        ready = [
            c
            for c in selected
            if valid_saved_answer(
                c,
                require_semantic=semantic,
                require_source_constraints=config.get(
                    "require_source_constraints",
                    config.get("false_answer_source") == "nqopen_import",
                ),
            )
        ]
        ready_ids = {c["query_id"] for c in ready}
        return {
            "total_questions": len(candidates),
            "selected_questions": len(selected),
            "ready_questions": len(ready),
            "needs_review_or_stale": len(selected) - len(ready),
            "ready_documents": sum(len(c["h_docs"]) for c in ready),
            "blocked_query_ids": [
                c["query_id"] for c in selected if c["query_id"] not in ready_ids
            ],
            "false_answer_source": config.get("false_answer_source", "generated"),
        }
    return run_generation_pipeline(
        {**config, "conditions": ["false"], "retry_seed": retry_seed}, candidates, limit=limit
    )
