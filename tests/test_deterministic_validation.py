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
