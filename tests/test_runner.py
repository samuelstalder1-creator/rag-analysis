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


def test_runner_executes_reader_and_restores_false_target(tmp_path, monkeypatch):
    import json
    from evlab.experiments import runner
    from evlab.generation.output import GenerationOutput
    from evlab.io import write_json, read_jsonl

    fixtures = Path(__file__).parent / "fixtures"
    output = GenerationOutput(tmp_path / "generated", {}, "test", {"false"})
    output.append(
        {
            "condition": "false",
            "sample_idx": 0,
            "provenance_root_id": "doc6",
            "title": "Chicago Fire",
            "text": "Season four had 24 episodes.",
            "validation_receipt": {
                "query_id": "test1",
                "gold_answer": "23",
                "target_answer": "24",
            },
        }
    )

    class Reader:
        model = provider = "test"

        def generate(self, prompt, **kwargs):
            assert "Context:" in prompt
            return json.dumps(
                {
                    "answer": (
                        "24" if "how many episodes" in prompt else "William Shakespeare"
                    ),
                    "citations": [1, 99],
                }
            )

    monkeypatch.setattr(runner, "build_llm_client", lambda cfg: Reader())
    cfg = {
        "experiment": "reader-test",
        "queries": "rag",
        "data": {
            "nqplus_dir": str(fixtures / "nqplus"),
            "human_corpus": str(fixtures / "corpus/human.jsonl"),
        },
        "corpus": {"generated_docs": str(tmp_path / "generated"), "k_false": 1},
        "retrievers": ["bm25"],
        "retrieval": {"top_k": 3},
        "evaluation": {"cutoffs": [1, 3]},
        "reader": {
            "llm": {"provider": "abstain"},
            "cache_path": str(tmp_path / "reader.sqlite"),
            "top_k": 3,
        },
        "output": {"dir": str(tmp_path / "runs")},
    }
    config = tmp_path / "config.json"
    write_json(config, cfg)
    result = run_experiment(config)
    answers = read_jsonl(next((tmp_path / "runs").glob("*/bm25/answers.jsonl")))
    assert answers[0]["label"] == "target" and answers[0]["target_answer"] == "24"
    assert answers[1]["label"] == "correct"
    assert answers[0]["citations_valid"] is False
    assert len(answers[0]["cited_doc_ids"]) == 1
    assert result["retrievers"]["bm25"]["answer_target_rate"] == 0.5


def test_rag_config_requires_explicit_reader(tmp_path):
    import pytest
    from evlab.io import write_json

    config = tmp_path / "config.json"
    write_json(config, {"queries": "rag", "output": {"dir": str(tmp_path)}})
    with pytest.raises(ValueError, match="reader.llm"):
        run_experiment(config)
