import csv
import json
from pathlib import Path

import pytest

from evlab.corpus.builder import build_corpus, build_corpus_with_report
from evlab.corpus.mix import nested_sample
from evlab.experiments.runner import run_experiment
from evlab.experiments.sweep import expand_sweep, run_sweep, set_path
from evlab.generation.output import GenerationOutput
from evlab.io import write_json

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def generated(tmp_path):
    """Correct copies for all fixture docs (two samples for doc6) and one false doc."""
    correct = GenerationOutput(
        tmp_path / "correct", {"samples_per_h": 2, "source": "human_corpus"}, "test", {"correct"}
    )
    texts = {
        "doc6": "Season four of Chicago Fire had 23 episodes in total.",
        "doc7": "The tragedy Hamlet was penned by William Shakespeare.",
        "doc8": "Mountain railways are the topic of this unrelated text.",
    }
    for doc_id, text in texts.items():
        correct.append({"condition": "correct", "sample_idx": 0, "provenance_root_id": doc_id, "title": doc_id, "text": text})
    correct.append({"condition": "correct", "sample_idx": 1, "provenance_root_id": "doc6", "title": "doc6", "text": "Chicago Fire's fourth season: 23 episodes."})

    false = GenerationOutput(tmp_path / "false", {}, "test", {"false"})
    false.append(
        {
            "condition": "false",
            "sample_idx": 0,
            "provenance_root_id": "doc6",
            "title": "Chicago Fire (season 4)",
            "text": "The fourth season of Chicago Fire contained 24 episodes.",
            "validation_receipt": {"query_id": "test1", "gold_answer": "23", "target_answer": "24"},
        }
    )
    return {"correct": str(tmp_path / "correct"), "false": str(tmp_path / "false")}


def config(generated, corpus, queries="retrieval", **extra):
    return {
        "experiment": "mix-test",
        "queries": queries,
        "data": {
            "nqplus_dir": str(FIXTURES / "nqplus"),
            "human_corpus": str(FIXTURES / "corpus/human.jsonl"),
        },
        "corpus": {
            "seed": 13,
            "sources": {
                "qwen_correct": {"path": generated["correct"]},
                "false_det": {"path": generated["false"]},
                "llama": {"path": str(FIXTURES / "corpus/llama.jsonl"), "origin": "cocktail_llm"},
            },
            **corpus,
        },
        **extra,
    }


def ids(docs):
    return sorted(doc.doc_id for doc in docs)


def test_nested_sample_is_deterministic_and_nested():
    units = [f"doc{i}" for i in range(1000)]
    quarter = nested_sample(units, 0.25, 13, "x")
    half = nested_sample(units, 0.5, 13, "x")
    assert len(quarter) == 250 and len(half) == 500
    assert half[: len(quarter)] == quarter
    assert nested_sample(units, 0.5, 13, "x") == half
    assert nested_sample(units, 0.5, 14, "x") != half
    assert nested_sample(units, 0.0, 13, "x") == []
    assert sorted(nested_sample(units, 1.0, 13, "x")) == sorted(units)


def test_full_cocktail_background_matches_legacy_mixed_corpus(generated):
    legacy = build_corpus(
        {
            "data": {
                "nqplus_dir": str(FIXTURES / "nqplus"),
                "human_corpus": str(FIXTURES / "corpus/human.jsonl"),
                "cocktail_llm_corpus": str(FIXTURES / "corpus/llama.jsonl"),
            },
            "corpus": {"base": "cocktail_mixed"},
        }
    )
    mixed = build_corpus(config(generated, {"background": [{"source": "llama", "ratio": 1.0}]}))
    assert ids(mixed) == ids(legacy)


def test_background_ratio_add_and_replace(generated):
    none, report = build_corpus_with_report(config(generated, {"background": [{"source": "qwen_correct", "ratio": 0.0}]}))
    assert ids(none) == ["human::doc6", "human::doc7", "human::doc8"]
    assert report["generated_docs"] == 0

    full = build_corpus(config(generated, {"background": [{"source": "qwen_correct", "ratio": 1.0}]}))
    assert len(full) == 6 and sum(doc.origin == "human" for doc in full) == 3

    replaced, report = build_corpus_with_report(
        config(generated, {"background": [{"source": "qwen_correct", "ratio": 1.0, "mode": "replace"}]})
    )
    assert len(replaced) == 3 and all(doc.origin == "generated" for doc in replaced)
    assert report["human_removed"] == 3 and report["generated_share"] == 1.0

    two_samples = build_corpus(config(generated, {"background": [{"source": "qwen_correct", "ratio": 1.0, "k": 2}]}))
    assert len(two_samples) == 7  # doc6 has two samples, the others one

    third = build_corpus(config(generated, {"background": [{"source": "qwen_correct", "ratio": 0.34}]}))
    two_thirds = build_corpus(config(generated, {"background": [{"source": "qwen_correct", "ratio": 0.67}]}))
    gen_third = {d.doc_id for d in third if d.origin == "generated"}
    gen_two_thirds = {d.doc_id for d in two_thirds if d.origin == "generated"}
    assert len(gen_third) == 1 and len(gen_two_thirds) == 2 and gen_third < gen_two_thirds


def test_background_relevant_scope_only_uses_qrel_docs(generated):
    docs, report = build_corpus_with_report(
        config(generated, {"background": [{"source": "qwen_correct", "ratio": 1.0, "scope": "relevant"}]})
    )
    roots = sorted(doc.root_id for doc in docs if doc.origin == "generated")
    assert roots == ["doc6", "doc7"]
    assert report["background"][0]["pool_roots"] == 2


def test_targeted_false_add_and_replace(generated):
    docs, report = build_corpus_with_report(
        config(generated, {"targeted": {"false_docs": {"source": "false_det"}}}, queries="generation")
    )
    false_docs = [doc for doc in docs if doc.condition == "false"]
    assert len(false_docs) == 1 and false_docs[0].target_answer == "24"
    assert "human::doc6" in ids(docs)
    assert report["targeted"]["targeted_queries"] == 1
    assert report["targeted"]["query_ids"] == ["test1"]

    docs = build_corpus(
        config(generated, {"targeted": {"false_docs": {"source": "false_det", "mode": "replace"}}}, queries="generation")
    )
    assert "human::doc6" not in ids(docs) and len(docs) == 3

    docs, report = build_corpus_with_report(
        config(generated, {"targeted": {"query_ratio": 0.0, "false_docs": {"source": "false_det"}}}, queries="generation")
    )
    assert report["false_docs"] == 0 and report["targeted"]["attackable_queries"] == 1


def test_targeted_mixture_of_false_and_correct_with_k(generated):
    docs, report = build_corpus_with_report(
        config(
            generated,
            {
                "targeted": {
                    "false_docs": {"source": "false_det", "k": 2},
                    "correct_docs": {"source": "qwen_correct", "k": 2},
                }
            },
            queries="rag",
        )
    )
    # rag has test1 (doc6, has a false doc) and test2 (doc7, no false doc -> not attackable)
    assert report["targeted"]["query_ids"] == ["test1"]
    assert report["false_docs"] == 1 and report["correct_docs"] == 2
    assert report["targeted"]["false_docs"]["k_shortfall"] == 1
    assert report["targeted"]["correct_docs"]["k_shortfall"] == 0
    assert {doc.root_id for doc in docs if doc.origin == "generated"} == {"doc6"}


def test_targeted_correct_only_and_overlap_with_background_is_deduplicated(generated):
    docs, report = build_corpus_with_report(
        config(
            generated,
            {
                "background": [{"source": "qwen_correct", "ratio": 1.0}],
                "targeted": {"correct_docs": {"source": "qwen_correct", "k": 1}},
            },
            queries="rag",
        )
    )
    assert report["targeted"]["attackable_queries"] == 2
    assert report["targeted"]["correct_docs"]["docs_added"] == 0
    assert len(docs) == len({doc.doc_id for doc in docs}) == 6


def test_background_mapping_form_equals_list_form(generated):
    as_list = build_corpus(config(generated, {"background": [{"source": "qwen_correct", "ratio": 0.67}]}))
    as_map = build_corpus(config(generated, {"background": {"qwen_correct": {"ratio": 0.67}}}))
    assert ids(as_list) == ids(as_map)


@pytest.mark.parametrize(
    "path", sorted((Path(__file__).parents[1] / "configs/archive/experiments/mix").glob("*.yaml")), ids=lambda p: p.name
)
def test_mix_configs_parse_without_pyyaml_and_expand(path, monkeypatch):
    from evlab.config import _parse_small_yaml
    from evlab.corpus.mix import validate_mix_config
    from evlab.rag.config import resolve_reader
    from evlab.retrieval.specs import resolve_retrievers

    monkeypatch.chdir(Path(__file__).resolve().parents[1])  # paths are repo-relative
    text = path.read_text(encoding="utf-8")
    parsed = _parse_small_yaml(text)
    yaml = pytest.importorskip("yaml")
    assert parsed == yaml.safe_load(text)
    for _, _, run_config in expand_sweep(parsed):
        validate_mix_config(run_config["corpus"])
        run_names, _ = resolve_retrievers(run_config)
        resolve_reader(run_config, run_names)
        resolve_reader({**run_config, "queries": "rag"} if run_config.get("reader") else {}, run_names)


def test_colliding_source_ids_fail_unless_namespaced(generated):
    cfg = config(generated, {"background": [{"source": "llama", "ratio": 1.0}, {"source": "other", "ratio": 1.0}]})
    cfg["corpus"]["sources"]["other"] = {"path": str(FIXTURES / "corpus/human.jsonl"), "origin": "cocktail_llm"}
    with pytest.raises(ValueError, match="namespace"):
        build_corpus(cfg)
    cfg["corpus"]["sources"]["other"]["namespace"] = "other_llm"
    assert len(build_corpus(cfg)) == 9


@pytest.mark.parametrize(
    "corpus, message",
    [
        ({"background": [{"source": "missing"}]}, "not defined"),
        ({"background": [{"source": "qwen_correct", "ratio": 1.5}]}, "between 0 and 1"),
        ({"background": [{"source": "qwen_correct", "mode": "swap"}]}, "mode"),
        ({"background": [{"source": "qwen_correct", "ratoi": 0.5}]}, "Unknown keys"),
        ({"k_false": 1}, "legacy"),
        ({"targeted": {"query_ratio": 1.0}}, "false_docs"),
        ({"targeted": {False: {"source": "false_det"}}}, "boolean"),
    ],
)
def test_invalid_mix_configs_fail_early(generated, corpus, message):
    with pytest.raises(ValueError, match=message):
        build_corpus(config(generated, corpus, queries="generation"))


def test_runner_writes_selection_and_targeted_metrics(generated, tmp_path):
    cfg = config(
        generated,
        {"targeted": {"false_docs": {"source": "false_det"}}},
        queries="retrieval",
        retrievers=["bm25"],
        retrieval={"top_k": 3},
        evaluation={"cutoffs": [1, 3]},
        output={"dir": str(tmp_path / "runs")},
    )
    summary = run_experiment(cfg)
    metrics = summary["retrievers"]["bm25"]
    assert summary["corpus"]["false_docs"] == 1
    assert "query_ids" not in summary["corpus"]["targeted"]
    assert metrics["targeted_false@3"] > 0
    assert metrics["targeted_false@3"] >= metrics["false@3"]
    selection = json.loads(next((tmp_path / "runs").glob("*/corpus_selection.json")).read_text())
    assert selection["targeted"]["query_ids"] == ["test1"]


def test_runner_rejects_sweep_configs(tmp_path):
    with pytest.raises(ValueError, match="evlab sweep"):
        run_experiment({"sweep": {"corpus.seed": [1]}})


def test_set_path_and_expand_sweep():
    cfg = {"experiment": "s", "corpus": {"background": [{"ratio": 0.1}]}}
    set_path(cfg, "corpus.background.0.ratio", 0.5)
    set_path(cfg, "corpus.targeted.false.k", 3)
    assert cfg["corpus"]["background"][0]["ratio"] == 0.5
    assert cfg["corpus"]["targeted"]["false"]["k"] == 3

    cfg["variants"] = [{"name": "add", "set": {"corpus.background.0.mode": "add"}}, {"name": "replace", "set": {"corpus.background.0.mode": "replace"}}]
    cfg["sweep"] = {"corpus.background.0.ratio": [0.0, 1.0]}
    runs = expand_sweep(cfg)
    assert len(runs) == 4
    labels = [label for label, _, _ in runs]
    assert labels[0] == "add__background.0.ratio=0.0"
    run_cfg = runs[3][2]
    assert run_cfg["corpus"]["background"][0] == {"ratio": 1.0, "mode": "replace"}
    assert "sweep" not in run_cfg and "variants" not in run_cfg
    assert len({c["experiment"] for _, _, c in runs}) == 4


def test_sweep_dry_run_and_full_run(generated, tmp_path):
    cfg = config(
        generated,
        {"background": [{"source": "qwen_correct", "ratio": 0.0}]},
        retrievers=["bm25"],
        retrieval={"top_k": 3},
        evaluation={"cutoffs": [1, 3]},
        output={"dir": str(tmp_path / "runs")},
        sweep_output={"dir": str(tmp_path / "sweeps")},
        sweep={"corpus.background.0.ratio": [0.0, 1.0]},
    )
    path = tmp_path / "sweep.json"
    write_json(path, cfg)

    preview = run_sweep(path, dry_run=True)
    assert [row["corpus.generated_docs"] for row in preview["rows"]] == [0, 3]

    result = run_sweep(path)
    with open(Path(result["sweep_dir"]) / "summary.csv", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [row["corpus.background.0.ratio"] for row in rows] == ["0.0", "1.0"]
    assert float(rows[0]["llm@3"]) == 0.0 and float(rows[1]["llm@3"]) > 0.0
