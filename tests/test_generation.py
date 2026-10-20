import json
from pathlib import Path

import pytest

from evlab.cli import _human_corpus_generation_candidates
from evlab.generation.false_answer import fallback_false_answers, generate_false_answer, is_valid_false_answer, parse_candidates
from evlab.generation.pipeline import _conditions, run_generation_pipeline
from evlab.generation.minimal_edit import deterministic_minimal_edit
from evlab.generation.validate import validate_false_copy
from evlab.io import read_jsonl
from evlab.generation.output import GenerationOutput, migrate_compact_files
from evlab.generation.registry import read_registry
from evlab.corpus.builder import build_corpus, load_generated_docs, select_generated_docs


def test_parse_false_answer_candidates():
    assert parse_candidates('{"candidates": ["22", "24", "25"]}') == ["22", "24", "25"]


def test_false_answer_rejects_gold_alias():
    assert not is_valid_false_answer("23", "23", ["23"])
    assert is_valid_false_answer("24", "23", ["23"])


def test_false_answer_fallback_for_bad_model_response():
    class BadClient:
        provider = "test"
        model = "bad"

        def generate(self, prompt, *, temperature=0.0, seed=None):
            return "not json"

    assert generate_false_answer(BadClient(), question="How many?", answer="23", aliases=["23"]) == "24"


def test_false_answer_fallbacks_preserve_coarse_type():
    assert fallback_false_answers("1999", ["1999"]) == ["2000", "1998", "2001"]
    assert fallback_false_answers("23", ["23"]) == ["24", "22", "25"]


def test_generation_conditions_accept_yaml_false_boolean():
    assert _conditions({"conditions": [False]}) == {"false"}
    assert _conditions({"conditions": ["Correct", "False"]}) == {"correct", "false"}


def test_human_corpus_generation_candidates_use_all_docs(tmp_path):
    corpus = tmp_path / "human.jsonl"
    corpus.write_text(
        '{"_id":"doc1","title":"One","text":"First text."}\n'
        '{"_id":"doc2","title":"Two","text":"Second text."}\n',
        encoding="utf-8",
    )
    candidates = _human_corpus_generation_candidates(str(corpus))
    assert [candidate["query_id"] for candidate in candidates] == ["doc1", "doc2"]
    assert candidates[0]["h_docs"][0]["text"] == "First text."


def test_minimal_edit_replaces_aliases():
    text = "The season contained 23 episodes."
    assert deterministic_minimal_edit(text, ["23"], "24") == "The season contained 24 episodes."


def test_false_validation_rejects_old_answer():
    result = validate_false_copy("The season contained 23 episodes.", target_answer="24", answer_aliases=["23"])
    assert not result.ok
    assert "missing_target_answer" in result.reasons
    assert "still_contains_gold_answer" in result.reasons


@pytest.mark.parametrize(
    ("title", "expected_title"),
    [
        ("Patrick Brown (politician)", "Jane Smith (politician)"),
        ("Patrick Walter Brown (politician)", "Jane Smith (politician)"),
        ("Ontario party leadership", "Ontario party leadership"),
        ("", ""),
    ],
)
def test_false_generation_preserves_title_without_gold_answer(tmp_path, monkeypatch, title, expected_title):
    from evlab.generation import pipeline

    class Client:
        provider = "test"
        model = "title-test"

        def generate(self, prompt, *, temperature=0.0, seed=None):
            if "Generate three plausible but incorrect answers" in prompt:
                raise AssertionError("Document generation must use the saved false answer")
            return "Jane Smith is the party leader."

    monkeypatch.setattr(pipeline, "build_llm_client", lambda config: Client())
    result = run_generation_pipeline(
        {
            "output_dir": str(tmp_path),
            "cache_path": str(tmp_path / "llm.sqlite"),
            "conditions": ["false"],
            "llm": {"provider": "abstain"},
        },
        [{
            "query_id": "test3", "query": "Who is the leader?", "answer": "Patrick Brown",
            "answer_aliases": ["Patrick Brown", "Patrick Walter Brown"], "false_answer": "Jane Smith",
            "h_docs": [{"doc_id": "doc17", "title": title,
                        "text": "Patrick Brown is the party leader.", "aliases_in_doc": ["Patrick Brown"]}],
        }],
    )
    assert result == {"generated": 1, "rejects": 0}
    combined = read_jsonl(next(tmp_path / name for name, meta in read_registry(tmp_path)["files"].items() if meta["condition"] == "false"))
    assert set(combined[0]) == {"_id", "title", "text"}
    assert combined[0]["_id"] == "doc17"
    assert combined[0]["title"] == expected_title
    assert "Patrick" not in combined[0]["text"]


def test_false_generation_rejects_title_with_remaining_gold_alias(tmp_path, monkeypatch):
    from evlab.generation import pipeline

    monkeypatch.setattr(pipeline, "minimal_edit_with_fallback", lambda *args, **kwargs: "Jane Smith and Patrick Brown")
    monkeypatch.setattr(pipeline, "paraphrase", lambda *args, **kwargs: "Jane Smith is the leader.")
    result = run_generation_pipeline(
        {"output_dir": str(tmp_path), "cache_path": str(tmp_path / "llm.sqlite"),
         "conditions": ["false"], "llm": {"provider": "abstain"}},
        [{"query_id": "test3", "query": "Who leads?", "answer": "Patrick Brown", "false_answer": "Jane Smith",
          "h_docs": [{"doc_id": "doc17", "title": "Patrick Brown (politician)",
                      "text": "Patrick Brown leads."}]}],
    )
    assert result == {"generated": 0, "rejects": 1}
    reject = read_jsonl(next(tmp_path.glob("*--rejects.jsonl")))[0]
    assert reject["reasons"] == ["title_still_contains_gold_answer"]
    assert reject["generated_title"] == "Jane Smith and Patrick Brown"


def test_generation_reject_logs_text_for_analysis(tmp_path, monkeypatch):
    class BadParaphraseClient:
        provider = "local_openai"
        model = "bad-paraphrase"

        def generate(self, prompt, *, temperature=0.0, seed=None):
            if "Generate three plausible but incorrect answers" in prompt:
                return '{"candidates": ["24"]}'
            return "This rewrite removed the answer."

    config = {
        "output_dir": str(tmp_path),
        "cache_path": str(tmp_path / "llm.sqlite"),
        "conditions": ["correct"],
        "llm": {"provider": "abstain"},
    }
    from evlab.generation import pipeline

    monkeypatch.setattr(pipeline, "build_llm_client", lambda llm_config: BadParaphraseClient())
    run_generation_pipeline(
        config,
        [
            {
                "query_id": "test1",
                "query": "how many episodes",
                "answer": "23",
                "answer_aliases": ["23"],
                "h_docs": [{"doc_id": "doc6", "title": "Title", "text": "There were 23 episodes."}],
            }
        ],
    )
    rejects = read_jsonl(next(tmp_path.glob("*--rejects.jsonl")))
    assert rejects[0]["stage"] == "correct_validation"
    assert rejects[0]["generated_text"] == "This rewrite removed the answer."
    assert rejects[0]["source_text"] == "There were 23 episodes."
    assert not list(tmp_path.glob("*progress.json"))
    assert all(meta["document_count"] == 0 for meta in read_registry(tmp_path)["files"].values())


def test_compact_corpora_keep_samples_conditions_and_models_separate(tmp_path):
    config = {"samples_per_h": 2, "temperature": 0.3}
    output = GenerationOutput(tmp_path, config, "Qwen/Qwen3.5-2B", {"correct", "false"})
    for condition in ("correct", "false"):
        for sample in range(2):
            output.append({"condition": condition, "sample_idx": sample,
                           "provenance_root_id": "doc6", "title": "Season four",
                           "text": "There were 23 episodes." if condition == "correct" else "There were 24 episodes."})
    other = GenerationOutput(tmp_path, config, "Other/model", {"correct"})
    assert not other.contains("correct", 0, "doc6")
    resumed = GenerationOutput(tmp_path, config, "Qwen/Qwen3.5-2B", {"correct", "false"})
    assert resumed.contains("false", 1, "doc6")
    docs = load_generated_docs(tmp_path, query_ids_by_doc={"doc6": "test1"})
    assert len(docs) == len({d.doc_id for d in docs}) == 4
    assert all(d.root_id == "doc6" and d.query_id == "test1" for d in docs)
    assert {d.sample_idx for d in docs} == {0, 1}
    assert {d.condition for d in docs} == {"correct", "false"}
    selected = select_generated_docs(docs, {"k_correct": 1, "k_false": 1, "generation_depth": 1})
    assert len(selected) == 2
    assert all(d.sample_idx == 0 for d in selected)
    fixtures = Path(__file__).parent / "fixtures"
    corpus = build_corpus({
        "data": {"human_corpus": str(fixtures / "corpus/human.jsonl"),
                 "nqplus_dir": str(fixtures / "nqplus")},
        "corpus": {"generated_docs": tmp_path,
                   "k_correct": 2, "k_false": 2, "generation_depth": 1},
    })
    generated = [doc for doc in corpus if doc.origin == "generated"]
    assert len(generated) == 4
    assert all(doc.query_id == "test1" and doc.root_id == "doc6" for doc in generated)


def test_resume_imports_legacy_output_and_skips_model_calls(tmp_path, monkeypatch):
    from evlab.generation import pipeline

    legacy = {"doc_id": "gen::old", "title": "Title", "text": "There were 23 episodes.",
              "query_id": "test1", "provenance_root_id": "doc6", "sample_idx": 0,
              "condition": "correct", "generator_model": "test/model"}
    (tmp_path / "generated_docs.jsonl").write_text(json.dumps(legacy) + "\n")

    class NoCallsClient:
        provider = "test"
        model = "test/model"

        def generate(self, *args, **kwargs):
            raise AssertionError("Completed documents must not be generated again")

    monkeypatch.setattr(pipeline, "build_llm_client", lambda config: NoCallsClient())
    result = run_generation_pipeline(
        {"output_dir": str(tmp_path), "cache_path": str(tmp_path / "llm.sqlite"),
         "conditions": ["correct"], "llm": {"provider": "abstain"}},
        [{"query_id": "test1", "query": "How many?", "answer": "23",
          "h_docs": [{"doc_id": "doc6", "title": "Title", "text": "There were 23 episodes."}]}],
    )
    assert result == {"generated": 0, "rejects": 0}
    assert read_jsonl(next(tmp_path / name for name, meta in read_registry(tmp_path)["files"].items() if meta["condition"] == "correct")) == [
        {"_id": "doc6", "title": "Title", "text": "There were 23 episodes."}
    ]
    assert json.loads((tmp_path / "generated_docs.jsonl").read_text()) == legacy


def test_registry_migration_preserves_content_ids_and_resume(tmp_path):
    from evlab.config import _parse_small_yaml

    old = tmp_path / "Qwen-Qwen3.5-2B--false--tmp0.3--s00--d1--candidates--v1.jsonl"
    content = '{"_id":"doc6","title":"Season four","text":"There were 24 episodes."}\n'
    old.write_text(content)
    before = load_generated_docs(old, query_ids_by_doc={"doc6": "test1"})[0]
    renamed = migrate_compact_files(tmp_path, model="Qwen/Qwen3.5-2B")
    assert renamed == {old.name: "corpus_0001.jsonl"}
    assert not old.exists()
    assert (tmp_path / "corpus_0001.jsonl").read_text() == content
    after = load_generated_docs(tmp_path, query_ids_by_doc={"doc6": "test1"})[0]
    assert after.doc_id == before.doc_id
    assert after.generator_model == "Qwen/Qwen3.5-2B"
    assert after.condition == "false" and after.root_id == "doc6"
    registry = read_registry(tmp_path)
    assert _parse_small_yaml((tmp_path / "metadata.yaml").read_text()) == registry
    assert registry["files"]["corpus_0001.jsonl"]["document_count"] == 1
    output = GenerationOutput(tmp_path, {"temperature": 0.3}, "Qwen/Qwen3.5-2B", {"false"})
    assert output.paths[("false", 0)].name == "corpus_0001.jsonl"
    assert output.contains("false", 0, "doc6")
    other = GenerationOutput(tmp_path, {"temperature": 0.4}, "Qwen/Qwen3.5-2B", {"false"})
    assert other.paths[("false", 0)].name == "corpus_0002.jsonl"
    assert not other.contains("false", 0, "doc6")
