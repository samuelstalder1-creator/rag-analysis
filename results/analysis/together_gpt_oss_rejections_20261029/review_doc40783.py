"""Regenerate the held document separately, then assess annotation alignment."""

import json
from pathlib import Path

from evlab.config import load_config
from evlab.data.nqplus import load_generation_candidates
from evlab.generation.false_answers import attach_saved_answers
from evlab.generation.pipeline import run_generation_pipeline
from evlab.generation.semantic import judge, ModelOutputError
from evlab.io import iter_jsonl, write_json
from evlab.llm.cache import CachedLLMClient
from evlab.llm.client import build_llm_client


def main():
    root = Path(__file__).resolve().parent
    config = load_config("configs/generation/nqopen1520-together-gpt-oss-120b-false.yaml")
    settings = config["generation"]
    candidates = attach_saved_answers(
        load_generation_candidates(config["data"]["nqplus_dir"], config["data"]["human_corpus"]),
        settings,
    )
    candidate = next(c for c in candidates if c["query_id"] == "test1150")
    review_dir = root / "doc40783_regeneration"
    generated = run_generation_pipeline(
        {**settings, "output_dir": str(review_dir), "exclude_doc_ids": [], "retry_seed": 7},
        [candidate],
    )
    report = {"query_id": "test1150", "doc_id": "doc40783", "generation": generated,
              "main_corpus_modified": False, "evaluator_model": settings["llm"]["model"],
              "evaluation_basis": "Supplied question, annotations and source; no external factual verification"}
    rows = list(iter_jsonl(review_dir / "corpus_0001.jsonl"))
    if rows:
        [row] = rows
        source = candidate["h_docs"][0]
        evidence = {
            "question": candidate["query"], "gold": candidate["answer"],
            "target": candidate["false_answer"], "source_title": source["title"],
            "source": source["text"], "generated": row["text"],
        }
        report["evidence"] = evidence
        (review_dir / "generated_text.txt").write_text(row["text"] + "\n")
        client = CachedLLMClient(build_llm_client(settings["llm"]), settings["cache_path"])
        try:
            report["semantic_review"] = judge(
                client, stage="held_annotation_review", evidence=evidence,
                checks={
                    "gold_answers_question": "The supplied gold is an answer to the exact question, supported by the source. A word merely appearing in the source is insufficient; distinguish court powers from the people who sit on the court.",
                    "generated_supports_target": "The generated passage affirmatively supports the full target as the answer to the exact question.",
                    "counterfactual_pair_is_aligned": "Gold and target are incompatible alternative answers to the SAME question and relation. Both must fit the question's answer type. Judge only the supplied evidence; false if alignment cannot be established.",
                },
            )
            report["status"] = "passed_semantic_review"
        except ModelOutputError as exc:
            report["status"] = "rejected_for_annotation_alignment"
            report["semantic_review"] = {"error": str(exc), "attempts": exc.attempts}
    else:
        report["status"] = "generation_rejected"
    write_json(review_dir / "review.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
