"""Runs without an LLM judge: validation.mode = deterministic."""

import pytest

from evlab.data.nqplus import load_generation_candidates
from evlab.generation import false_answers, pipeline
from evlab.generation.false_answer import is_valid_false_answer
from evlab.generation.false_answers import attach_saved_answers, run_false_answer_generation
from evlab.generation.false_documents import run_false_document_generation
from evlab.generation.registry import read_registry
from evlab.io import read_jsonl

from test_false_answer_stages import setup_data


@pytest.mark.parametrize(
    "candidate, gold, aliases",
    [
        ("Christopher Lloyd", "Christopher Allen Lloyd", ["Christopher Allen Lloyd"]),
        ("Timothy Schmit", "Timothy B. Schmit", ["Timothy B. Schmit"]),
        ("Patrick Brown Jr.", "Patrick Brown", ["Patrick Brown"]),
        ("Hold On Tight", "Hold On", ["Hold On"]),
    ],
)
def test_near_aliases_are_not_false_answers(candidate, gold, aliases):
    assert not is_valid_false_answer(candidate, gold, aliases, question="who sang it")


@pytest.mark.parametrize(
    "candidate, gold",
    [("Bryan Adams", "Timothy B. Schmit"), ("John A. Brown", "Patrick Brown"), ("tail", "midpiece")],
)
def test_distinct_answers_remain_valid(candidate, gold):
    assert is_valid_false_answer(candidate, gold, [gold], question="who sang it")


def _deterministic(config):
    return {**config, "validation": {"mode": "deterministic"}}


def _no_judge(config):
    raise AssertionError("deterministic mode must not build an LLM judge")


class GeneratorClient:
    provider = "test"
    model = "test/stages"

    def __init__(self, paraphrase="A total of 24 episodes were broadcast."):
        self.paraphrase = paraphrase
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if "Generate one plausible but incorrect answer" in prompt:
            return '{"candidate": "24"}'
        if prompt.startswith("Select only answer-bearing"):
            return '{"spans": [0]}'
        assert not prompt.startswith("SEMANTIC_VALIDATION")
        return self.paraphrase


def _run_answers(tmp_path, monkeypatch, client):
    root, config = setup_data(tmp_path)
    config = _deterministic(config)
    monkeypatch.setattr(false_answers, "build_llm_client", lambda c: client)
    monkeypatch.setattr(pipeline, "build_llm_client", lambda c: client)
    monkeypatch.setattr(false_answers, "build_verifier", _no_judge)
    monkeypatch.setattr(pipeline, "build_verifier", _no_judge)
    result = run_false_answer_generation(config, root, human_corpus=tmp_path / "human.jsonl")
    return root, config, result


def test_deterministic_answers_and_documents_without_judge(tmp_path, monkeypatch):
    client = GeneratorClient()
    root, config, result = _run_answers(tmp_path, monkeypatch, client)
    assert result == {"generated": 1, "reused": 0, "rejects": 0}
    [row] = read_jsonl(tmp_path / "output" / "generated_false_answers.jsonl")
    assert row["false_answer"] == "24"
    assert row["verification"]["mode"] == "deterministic"
    assert row["verification"]["semantic_verified"] is False

    docs = run_false_document_generation(config, root, human_corpus=tmp_path / "human.jsonl")
    assert docs == {"generated": 2, "rejects": 0}
    registry = read_registry(tmp_path / "output")
    false_files = [e for e in registry["files"].values() if e["condition"] == "false"]
    assert false_files and all(e["validation_mode"] == "deterministic" for e in false_files)
    assert not any(p.startswith("SEMANTIC_VALIDATION") for p in client.prompts)


def test_semantic_run_rejects_deterministic_answers(tmp_path, monkeypatch):
    root, config, _ = _run_answers(tmp_path, monkeypatch, GeneratorClient())
    semantic = {k: v for k, v in config.items() if k != "validation"}
    candidates = attach_saved_answers(
        load_generation_candidates(root, tmp_path / "human.jsonl"), semantic
    )
    assert "false_answer" not in candidates[0]
    with pytest.raises(ValueError, match="unverified saved false answers"):
        pipeline.run_generation_pipeline(semantic, candidates)


def test_gold_residue_after_paraphrase_is_rejected(tmp_path, monkeypatch):
    client = GeneratorClient("There were 24 episodes, although 23 were planned.")
    root, config, _ = _run_answers(tmp_path, monkeypatch, client)
    docs = run_false_document_generation(config, root, human_corpus=tmp_path / "human.jsonl")
    assert docs == {"generated": 0, "rejects": 2}
    rejects = [r for p in (tmp_path / "output").glob("*--rejects.jsonl") for r in read_jsonl(p)]
    assert {r["reason"] for r in rejects} == {"gold_answer_residue"}


def test_unknown_validation_mode_fails(tmp_path, monkeypatch):
    root, config = setup_data(tmp_path)
    with pytest.raises(ValueError, match="validation.mode"):
        run_false_answer_generation({**config, "validation": {"mode": "none"}}, root,
                                    human_corpus=tmp_path / "human.jsonl")


def test_document_stage_can_skip_questions_without_answer(tmp_path, monkeypatch):
    root, config = setup_data(tmp_path, two_questions=True)
    config = {**_deterministic(config), "skip_missing_false_answers": True}
    client = GeneratorClient()  # always proposes "24", which is gold for q2
    monkeypatch.setattr(false_answers, "build_llm_client", lambda c: client)
    monkeypatch.setattr(pipeline, "build_llm_client", lambda c: client)
    monkeypatch.setattr(false_answers, "build_verifier", _no_judge)
    monkeypatch.setattr(pipeline, "build_verifier", _no_judge)
    human = tmp_path / "human.jsonl"
    answers = run_false_answer_generation(config, root, human_corpus=human)
    assert answers == {"generated": 1, "reused": 0, "rejects": 1}
    docs = run_false_document_generation(config, root, human_corpus=human)
    assert docs == {"generated": 2, "rejects": 0, "skipped_questions": 1}

    strict = {k: v for k, v in config.items() if k != "skip_missing_false_answers"}
    with pytest.raises(ValueError, match="q2"):
        run_false_document_generation(strict, root, human_corpus=human)


def test_candidate_list_skips_gold_and_rejected_proposals(tmp_path, monkeypatch):
    class ListClient(GeneratorClient):
        def generate(self, prompt, **kwargs):
            if "WRONG multiple-choice distractors" in prompt:
                assert "Write 5 different" in prompt
                return '{"candidates": ["23", "twenty-three", "22", "21"]}'
            return super().generate(prompt, **kwargs)

    root, config = setup_data(tmp_path)
    config = {**_deterministic(config), "false_answer_candidates_per_call": 5}
    client = ListClient()
    monkeypatch.setattr(false_answers, "build_llm_client", lambda c: client)
    monkeypatch.setattr(false_answers, "build_verifier", _no_judge)
    run_false_answer_generation(config, root, human_corpus=tmp_path / "human.jsonl")
    [row] = read_jsonl(tmp_path / "output" / "generated_false_answers.jsonl")
    assert row["false_answer"] == "22"
    assert row["verification"]["candidates_per_call"] == 5


def test_candidate_list_with_only_gold_is_rejected(tmp_path, monkeypatch):
    class GoldClient(GeneratorClient):
        def generate(self, prompt, **kwargs):
            return '{"candidates": ["23", "23"]}'

    root, config = setup_data(tmp_path)
    config = {**_deterministic(config), "false_answer_candidates_per_call": 3}
    monkeypatch.setattr(false_answers, "build_llm_client", lambda c: GoldClient())
    monkeypatch.setattr(false_answers, "build_verifier", _no_judge)
    result = run_false_answer_generation(config, root, human_corpus=tmp_path / "human.jsonl")
    assert result == {"generated": 0, "reused": 0, "rejects": 1}


def test_all_proposals_are_saved_for_later_selection(tmp_path, monkeypatch):
    class ListClient(GeneratorClient):
        def generate(self, prompt, **kwargs):
            if "WRONG multiple-choice distractors" in prompt:
                return '{"candidates": ["23", "22", "21", "22", "lots"]}'
            return super().generate(prompt, **kwargs)

    root, config = setup_data(tmp_path)
    config = {**_deterministic(config), "false_answer_candidates_per_call": 5}
    monkeypatch.setattr(false_answers, "build_llm_client", lambda c: ListClient())
    monkeypatch.setattr(false_answers, "build_verifier", _no_judge)
    run_false_answer_generation(config, root, human_corpus=tmp_path / "human.jsonl")
    [row] = read_jsonl(tmp_path / "output" / "generated_false_answers.jsonl")
    assert [(c["text"], c["status"]) for c in row["candidates"]] == [
        ("23", "gold_equivalent"), ("22", "valid"), ("21", "valid"),
        ("lots", "invalid_type_or_placeholder"),
    ]
    assert row["valid_candidates"] == ["22", "21"]
    assert row["selection"] == {"method": "first_valid_in_model_order",
                                "selected": "22", "selector_model": None}
    assert row["false_answer"] == "22"


def test_question_copy_is_not_a_false_answer():
    q = "nitty gritty dirt band fishin in the dark album"
    assert not is_valid_false_answer(q, "Hold On", ["Hold On"], question=q)
    assert is_valid_false_answer("Plain Dirt Fashion", "Hold On", ["Hold On"], question=q)


def test_rule_based_span_selection_edits_all_mentions(tmp_path, monkeypatch):
    class NoSpanClient(GeneratorClient):
        def generate(self, prompt, **kwargs):
            assert not prompt.startswith("Select only answer-bearing")
            return super().generate(prompt, **kwargs)

    client = NoSpanClient()
    root, config, _ = _run_answers(tmp_path, monkeypatch, client)
    config = {**config, "span_selection": "all"}
    docs = run_false_document_generation(config, root, human_corpus=tmp_path / "human.jsonl")
    assert docs == {"generated": 2, "rejects": 0}


def test_failed_llm_span_selection_logs_raw_response(tmp_path, monkeypatch):
    client = GeneratorClient()
    root, config, _ = _run_answers(tmp_path, monkeypatch, client)
    client.generate = lambda prompt, **kw: "not json" if prompt.startswith("Select only") else "x"
    docs = run_false_document_generation(config, root, human_corpus=tmp_path / "human.jsonl")
    assert docs["generated"] == 0
    rejects = [r for p in (tmp_path / "output").glob("*--rejects.jsonl") for r in read_jsonl(p)]
    assert rejects and all(r["attempts"][0]["response"] == "not json" for r in rejects)


def test_targets_in_source_and_articles_are_handled(tmp_path, monkeypatch):
    class ListClient(GeneratorClient):
        def generate(self, prompt, **kwargs):
            if "WRONG multiple-choice distractors" in prompt:
                # "episodes" occurs in the source; "the 22" loses its article.
                return '{"candidates": ["23", "episodes", "the 21"]}'
            return super().generate(prompt, **kwargs)

    root, config = setup_data(tmp_path)
    config = {**_deterministic(config), "false_answer_candidates_per_call": 5}
    monkeypatch.setattr(false_answers, "build_llm_client", lambda c: ListClient())
    monkeypatch.setattr(false_answers, "build_verifier", _no_judge)
    run_false_answer_generation(config, root, human_corpus=tmp_path / "human.jsonl")
    [row] = read_jsonl(tmp_path / "output" / "generated_false_answers.jsonl")
    assert [(c["text"], c["status"]) for c in row["candidates"]] == [
        ("23", "gold_equivalent"), ("episodes", "invalid_type_or_placeholder"), ("21", "valid"),
    ]
    assert row["false_answer"] == "21"


def test_entity_already_in_source_is_rejected():
    from evlab.generation.false_answer import proposal_status
    src = ["One B-29 dropped Little Boy on Hiroshima; Fat Man was dropped on Nagasaki."]
    args = ("Little Boy", ["Little Boy"], "name of the bomb dropped on hiroshima", set(), src)
    assert proposal_status("Fat Man", *args) == "occurs_in_source"
    assert proposal_status("Trinity", *args) == "valid"


def test_article_matching_keeps_proper_names():
    from evlab.generation.false_answer import match_article
    assert match_article("the tail", "midpiece") == "tail"
    assert match_article("The Beatles", "Elvis Presley") == "The Beatles"
    assert match_article("the tail", "the midpiece") == "the tail"
