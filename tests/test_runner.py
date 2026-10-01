from pathlib import Path

from evlab.experiments.runner import run_experiment


def test_runner_smoke(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    config = tmp_path / "experiment.yaml"
    config.write_text(
        f"""
experiment: smoke
queries: retrieval
data:
  nqplus_dir: {fixtures / "nqplus"}
  human_corpus: {fixtures / "corpus" / "human.jsonl"}
  cocktail_llm_corpus: {fixtures / "corpus" / "llama.jsonl"}
corpus:
  base: cocktail_mixed
retrievers: [bm25, tfidf, hash_dense]
retrieval:
  top_k: 3
evaluation:
  cutoffs: [1, 3]
  relevance_modes: [strict, relaxed]
output:
  dir: {tmp_path / "runs"}
""",
        encoding="utf-8",
    )
    summary = run_experiment(config)
    assert summary["num_docs"] == 6
    assert summary["num_queries"] == 2
    assert set(summary["retrievers"]) == {"bm25", "tfidf", "hash_dense"}
