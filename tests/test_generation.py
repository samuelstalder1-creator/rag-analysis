import json

from evlab.cli import _human_corpus_generation_candidates
from evlab.generation.false_answer import fallback_false_answers, generate_false_answer, is_valid_false_answer, parse_candidates
from evlab.generation.pipeline import _conditions, run_generation_pipeline
from evlab.generation.minimal_edit import deterministic_minimal_edit
from evlab.generation.validate import validate_false_copy
from evlab.io import read_jsonl


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
    rejects = read_jsonl(tmp_path / "rejects.jsonl")
    assert rejects[0]["stage"] == "correct_validation"
    assert rejects[0]["generated_text"] == "This rewrite removed the answer."
    assert rejects[0]["source_text"] == "There were 23 episodes."
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "complete"
    assert progress["candidates_seen"] == 1
