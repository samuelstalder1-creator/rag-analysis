import json
from pathlib import Path

import pytest

from evlab.experiments import runner
from evlab.experiments.runner import run_experiment
from evlab.io import read_jsonl
from evlab.rag.config import resolve_reader
from evlab.rag.reader import READER_PROMPT
from evlab.retrieval import specs as specs_module
from evlab.retrieval.fusion import fuse_runs
from evlab.retrieval.specs import resolve_retrievers
from evlab.schema import Hit

FIXTURES = Path(__file__).parent / "fixtures"
REPO = Path(__file__).parents[1]


def hits(qid, ranked, retriever="x"):
    return [Hit(qid, doc, rank, score, retriever) for rank, (doc, score) in enumerate(ranked, start=1)]


def base_config(tmp_path, **extra):
    return {
        "experiment": "spec-test",
        "queries": "retrieval",
        "data": {
            "nqplus_dir": str(FIXTURES / "nqplus"),
            "human_corpus": str(FIXTURES / "corpus/human.jsonl"),
            "cocktail_llm_corpus": str(FIXTURES / "corpus/llama.jsonl"),
        },
        "corpus": {"base": "cocktail_mixed"},
        "retrieval": {"top_k": 3},
        "evaluation": {"cutoffs": [1, 3]},
        "output": {"dir": str(tmp_path / "runs")},
        **extra,
    }


# --- resolution -----------------------------------------------------------

def test_builtin_names_take_defaults_from_shared_retrieval_block():
    names, specs = resolve_retrievers({"retrievers": ["bm25", "tfidf"], "retrieval": {"k1": 0.9, "b": 0.4, "top_k": 50}})
    assert names == ["bm25", "tfidf"]
    assert specs["bm25"] == {"type": "bm25", "k1": 0.9, "b": 0.4}
    assert specs["tfidf"] == {"type": "tfidf"}


def test_definitions_override_and_name_variants():
    names, specs = resolve_retrievers(
        {
            "retrievers": ["bm25", "bm25_short_docs"],
            "retrieval": {"k1": 1.5},
            "retriever_definitions": {
                "bm25": {"k1": 1.2},
                "bm25_short_docs": {"type": "bm25", "b": 0.3},
            },
        }
    )
    assert specs["bm25"]["k1"] == 1.2
    assert specs["bm25_short_docs"] == {"type": "bm25", "k1": 1.5, "b": 0.3}


def test_mapping_form_file_definitions_and_hidden_hybrid_members():
    names, specs = resolve_retrievers(
        {
            "retrievers": {
                "bm25": str(REPO / "configs/retrievers/bm25.yaml"),
                "hybrid": {"type": "hybrid", "members": ["bm25", "e5"], "fusion": "weighted", "weights": {"e5": 2.0}},
            },
            "retrieval": {"dense_device": "cuda", "dense_model": "models/e5"},
        }
    )
    assert names == ["bm25", "hybrid"]
    assert specs["bm25"]["type"] == "bm25" and specs["bm25"]["k1"] == 1.5
    assert specs["e5"] == {"type": "e5", "device": "cuda", "model": "models/e5"}  # member only, not run
    assert specs["hybrid"]["rrf_k"] == 60 and specs["hybrid"]["normalization"] == "minmax"


@pytest.mark.parametrize(
    "config, message",
    [
        ({"retrievers": ["bm42"]}, "Unknown retriever 'bm42'"),
        ({"retrievers": ["bm25"], "retriever_definitions": {"bm25": {"k3": 1}}}, "Unknown keys"),
        ({"retrievers": ["h"], "retriever_definitions": {"h": {"type": "hybrid", "members": ["bm25"]}}}, "at least two"),
        ({"retrievers": ["h"], "retriever_definitions": {"h": {"type": "hybrid", "members": ["bm25", "tfidf"], "fusion": "max"}}}, "fusion"),
        ({"retrievers": ["h"], "retriever_definitions": {"h": {"type": "hybrid", "members": ["bm25", "tfidf"], "weights": {"e5": 1}}}}, "non-members"),
        (
            {
                "retrievers": ["a"],
                "retriever_definitions": {
                    "a": {"type": "hybrid", "members": ["b", "bm25"]},
                    "b": {"type": "hybrid", "members": ["a", "tfidf"]},
                },
            },
            "cycle",
        ),
        ({"retrievers": ["bm25", "bm25"]}, "Duplicate"),
        ({"retrievers": []}, "empty"),
    ],
)
def test_invalid_retriever_configs(config, message):
    with pytest.raises(ValueError, match=message):
        resolve_retrievers(config)


# --- fusion ---------------------------------------------------------------

def test_rrf_fusion():
    runs = {
        "bm25": {"q": hits("q", [("a", 9.0), ("b", 5.0)])},
        "tfidf": {"q": hits("q", [("b", 0.9), ("c", 0.8)])},
    }
    fused = fuse_runs(runs, name="h", top_k=3, method="rrf", rrf_k=60)["q"]
    assert [h.doc_id for h in fused] == ["b", "a", "c"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].retriever == "h" and [h.rank for h in fused] == [1, 2, 3]
    weighted = fuse_runs(runs, name="h", top_k=3, method="rrf", weights={"tfidf": 0.0})["q"]
    assert weighted[0].doc_id == "a"


def test_weighted_fusion_with_minmax():
    runs = {
        "bm25": {"q": hits("q", [("a", 10.0), ("b", 6.0), ("c", 2.0)])},
        "tfidf": {"q": hits("q", [("c", 0.9), ("b", 0.5), ("a", 0.1)])},
    }
    fused = fuse_runs(runs, name="h", top_k=3, method="weighted", weights={"bm25": 1.0, "tfidf": 1.0})["q"]
    # minmax: a = 1 + 0, b = .5 + .5, c = 0 + 1 -> tie broken by doc id
    assert [h.doc_id for h in fused] == ["a", "b", "c"]
    fused = fuse_runs(runs, name="h", top_k=1, method="weighted", weights={"tfidf": 2.0})["q"]
    assert [h.doc_id for h in fused] == ["c"]


# --- runner ---------------------------------------------------------------

def test_runner_runs_hybrid_and_indexes_each_base_once(tmp_path, monkeypatch):
    built = []
    original = specs_module.build_base_retriever

    def counting(spec):
        built.append(spec["type"])
        return original(spec)

    monkeypatch.setattr(specs_module, "build_base_retriever", counting)
    cfg = base_config(
        tmp_path,
        retrievers=["bm25", "tfidf", "bm25_tfidf_rrf"],
        retriever_definitions={"bm25_tfidf_rrf": {"type": "hybrid", "members": ["bm25", "tfidf"], "depth": 6}},
    )
    summary = run_experiment(cfg)
    assert set(summary["retrievers"]) == {"bm25", "tfidf", "bm25_tfidf_rrf"}
    assert sorted(built) == ["bm25", "tfidf"]
    hybrid_rows = read_jsonl(next((tmp_path / "runs").glob("*/bm25_tfidf_rrf/retrieval.jsonl")))
    assert max(row["rank"] for row in hybrid_rows) == 3
    assert summary["retriever_specs"]["bm25_tfidf_rrf"]["members"] == ["bm25", "tfidf"]
    resolved = json.loads(next((tmp_path / "runs").glob("*/resolved.json")).read_text())
    assert resolved["reader"] is None


def test_hybrid_only_run_still_works(tmp_path):
    cfg = base_config(
        tmp_path,
        retrievers={"fused": {"type": "hybrid", "members": ["bm25", "tfidf"], "fusion": "weighted"}},
    )
    summary = run_experiment(cfg)
    assert list(summary["retrievers"]) == ["fused"]
    assert summary["retrievers"]["fused"]["strict_recall@3"] > 0


# --- reader ---------------------------------------------------------------

def test_resolve_reader_from_preset_file_and_prompt_file():
    spec = resolve_reader(
        {"queries": "rag", "reader": {"llm": str(REPO / "configs/readers/qwen.yaml"), "prompt": str(REPO / "prompts/reader_v1.txt"), "temperature": 0.2, "seed": 7}},
        ["bm25"],
    )
    assert spec["llm"]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert spec["temperature"] == 0.2 and spec["seed"] == 7
    assert spec["prompt_template"].strip() == READER_PROMPT.strip()
    assert spec["retrievers"] == ["bm25"]


def test_reader_disabled_without_rag_queries():
    assert resolve_reader({"queries": "retrieval", "reader": {"llm": {"provider": "abstain"}}}, ["bm25"]) is None
    assert resolve_reader({"queries": "retrieval", "reader": {"enabled": True, "llm": {"provider": "abstain"}}}, ["bm25"])


@pytest.mark.parametrize(
    "reader, message",
    [
        ({}, "reader.llm"),
        ({"llm": {"provider": "openai_magic", "model": "x"}}, "provider"),
        ({"llm": {"provider": "local_openai", "model": "x"}}, "base_url"),
        ({"llm": {"provider": "gemini"}}, "model"),
        ({"llm": {"provider": "abstain", "temprature": 0}}, "Unknown keys in reader.llm"),
        ({"llm": {"provider": "abstain"}, "temprature": 0}, "Unknown keys in reader"),
        ({"llm": {"provider": "abstain"}, "prompt": "Q: {question}"}, "context"),
        ({"llm": {"provider": "abstain"}, "prompt": "{question} {context} {\"a\": 1}"}, "braces"),
        ({"llm": {"provider": "abstain"}, "retrievers": ["tfidf"]}, "not in retrievers"),
    ],
)
def test_invalid_reader_configs(reader, message):
    with pytest.raises(ValueError, match=message):
        resolve_reader({"queries": "rag", "reader": reader}, ["bm25"])


def test_reader_settings_reach_the_llm_and_only_selected_retrievers(tmp_path, monkeypatch):
    calls = []

    class Reader:
        provider = "test"
        model = "test-model"

        def generate(self, prompt, *, temperature=0.0, seed=None):
            calls.append((prompt, temperature, seed))
            return json.dumps({"answer": "23", "citations": [1]})

    monkeypatch.setattr(runner, "build_llm_client", lambda cfg: Reader())
    cfg = base_config(
        tmp_path,
        queries="rag",
        retrievers=["bm25", "tfidf"],
        reader={
            "llm": {"provider": "abstain"},
            "temperature": 0.3,
            "seed": 11,
            "top_k": 1,
            "prompt": "CUSTOM {question}\n{context}",
            "retrievers": ["tfidf"],
            "cache_path": str(tmp_path / "reader.sqlite"),
        },
    )
    summary = run_experiment(cfg)
    assert len(calls) == 2  # two rag questions, only tfidf feeds the reader
    assert all(prompt.startswith("CUSTOM") and t == 0.3 and s == 11 for prompt, t, s in calls)
    assert all(prompt.count("[1]") == 1 and "[2]" not in prompt for prompt, _, _ in calls)
    assert "answer_correct_rate" in summary["retrievers"]["tfidf"]
    assert "answer_correct_rate" not in summary["retrievers"]["bm25"]
    assert summary["reader"]["temperature"] == 0.3 and summary["reader"]["retrievers"] == ["tfidf"]
    assert not list((tmp_path / "runs").glob("*/bm25/answers.jsonl"))


def test_run_id_follows_prompt_file_content(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("A {question} {context}", encoding="utf-8")
    cfg = base_config(
        tmp_path,
        queries="rag",
        reader={"llm": {"provider": "abstain"}, "prompt": str(prompt), "cache_path": str(tmp_path / "r.sqlite")},
    )
    first = run_experiment(cfg)["run_id"]
    prompt.write_text("B {question} {context}", encoding="utf-8")
    second = run_experiment(cfg)["run_id"]
    assert first != second
