import pytest

from evlab.cli import main
from evlab.data.nqplus import load_generation_candidates
from evlab.generation import false_answers, pipeline
from evlab.generation.false_answers import run_false_answer_generation
from evlab.generation.registry import read_registry
from evlab.generation.semantic import validator_signature
from evlab.io import read_jsonl, write_json, write_jsonl


def setup_data(tmp_path, *, two_questions=False):
    root = tmp_path / "nqplus"
    queries = [
        {
            "_id": "q1",
            "text": "How many episodes?",
            "metadata": {
                "answer": "23",
                "answer_aliases": ["23"],
                "untouched": "keep me",
            },
        }
    ]
    candidate = {
        "query_id": "q1",
        "query": "How many episodes?",
        "answer": "23",
        "answer_aliases": ["23"],
        "h_docs": [
            {"doc_id": did, "title": "23 episodes", "text": "There were 23 episodes."}
            for did in ("d1", "d2")
        ],
    }
    candidates = [candidate]
    if two_questions:
        queries.append(
            {
                "_id": "q2",
                "text": "How many seasons?",
                "metadata": {"answer": "24", "answer_aliases": ["24"]},
            }
        )
        candidates.append(
            {
                "query_id": "q2",
                "query": "How many seasons?",
                "answer": "24",
                "answer_aliases": ["24"],
                "h_docs": [
                    {
                        "doc_id": "d3",
                        "title": "Seasons",
                        "text": "There were 24 seasons.",
                    }
                ],
            }
        )
    write_jsonl(root / "queries.jsonl", queries)
    for row, candidate in zip(queries, candidates):
        row["metadata"].update(
            relevant_docs=[d["doc_id"] for d in candidate["h_docs"]],
            docs_with_answer=[d["doc_id"] for d in candidate["h_docs"]],
            usable_for={"retrieval": True, "reader": True, "generation": True},
        )
    write_jsonl(root / "queries.jsonl", queries)
    write_jsonl(
        tmp_path / "human.jsonl",
        [
            {"_id": doc["doc_id"], "title": doc["title"], "text": doc["text"]}
            for candidate in candidates
            for doc in candidate["h_docs"]
        ],
    )
    config = {
        "output_dir": str(tmp_path / "output"),
        "cache_path": str(tmp_path / "llm.sqlite"),
        "conditions": ["false"],
        "llm": {"provider": "abstain", "model": "test/stages"},
    }
    return root, config


def test_cli_two_stages_share_saved_answer_and_resume(tmp_path, monkeypatch, verifier):
    root, config = setup_data(tmp_path)
    config_path = tmp_path / "config.json"
    write_json(
        config_path,
        {
            "data": {
                "nqplus_dir": str(root),
                "human_corpus": str(tmp_path / "human.jsonl"),
            },
            "generation": config,
        },
    )
    calls = []

    class AnswerClient:
        provider = "test"
        model = "test/stages"

        def generate(self, prompt, **kwargs):
            assert "Generate one plausible but incorrect answer" in prompt
            calls.append("answer")
            return '{"candidate": "24"}'

    class DocumentClient:
        provider = "test"
        model = "test/stages"

        def generate(self, prompt, **kwargs):
            assert "Generate one plausible but incorrect answer" not in prompt
            calls.append("document")
            return "A total of 24 episodes were broadcast."

    monkeypatch.setattr(
        false_answers, "build_llm_client", lambda config: AnswerClient()
    )
    monkeypatch.setattr(pipeline, "build_llm_client", lambda config: DocumentClient())
    original = (root / "queries.jsonl").read_bytes()
    main(["generate-false-answers", "--config", str(config_path)])
    assert (root / "queries.jsonl").read_bytes() == original
    stored = read_jsonl(root / "queries.jsonl")[0]
    assert "false_answer" not in stored["metadata"]
    assert read_jsonl(false_answers.answers_path(config))[0]["false_answer"] == "24"
    assert stored["metadata"]["answer"] == "23"
    assert stored["metadata"]["untouched"] == "keep me"
    assert not list((tmp_path / "output").glob("corpus_*.jsonl"))
    main(["generate-false-documents", "--config", str(config_path)])
    registry = read_registry(tmp_path / "output")
    assert len(registry["answer_generations"]) == 1
    assert len(registry["files"]) == 1
    filename = next(iter(registry["files"]))
    docs = read_jsonl(tmp_path / "output" / filename)
    assert len(docs) == 2
    assert all(set(doc) == {"_id", "title", "text"} for doc in docs)
    assert all(
        doc["title"] == "24 episodes"
        and "24" in doc["text"]
        and "23" not in doc["text"]
        for doc in docs
    )
    assert calls == ["answer", "document"]
    main(["generate-false-answers", "--config", str(config_path)])
    main(["generate-false-documents", "--config", str(config_path)])
    assert calls == ["answer", "document"]
    assert read_jsonl(tmp_path / "output" / filename) == docs
    assert not list((tmp_path / "output").glob("*progress.json"))
    assert (root / "queries.jsonl").read_bytes() == original


@pytest.mark.parametrize("answer", [None, "", "23"])
def test_document_stage_fails_before_model_calls_without_valid_saved_answer(
    tmp_path, monkeypatch, answer
):
    root, config = setup_data(tmp_path)
    candidates = load_generation_candidates(root, include_text=False)
    candidates[0]["false_answer"] = answer

    def no_client(config):
        raise AssertionError(
            "The document stage must validate input before constructing a client"
        )

    monkeypatch.setattr(pipeline, "build_llm_client", no_client)
    with pytest.raises(ValueError, match="generate-false-answers"):
        pipeline.run_generation_pipeline(config, candidates)
    assert not (tmp_path / "output").exists()


def test_answer_checkpoints_survive_failure_and_retry_only_missing(
    tmp_path, monkeypatch, verifier
):
    root, config = setup_data(tmp_path, two_questions=True)
    calls = []
    failing = True

    class Client:
        provider = "test"
        model = "test/stages"

        def generate(self, prompt, **kwargs):
            calls.append(prompt)
            if "How many seasons?" in prompt:
                if failing:
                    raise RuntimeError("Server temporarily unavailable")
                return '{"candidate": "25"}'
            return '{"candidate": "24"}'

    monkeypatch.setattr(false_answers, "build_llm_client", lambda config: Client())
    result = run_false_answer_generation(
        config, root, human_corpus=tmp_path / "human.jsonl"
    )
    assert result == {"generated": 1, "reused": 0, "rejects": 1}
    rows = read_jsonl(false_answers.answers_path(config))
    assert len(rows) == 1
    assert rows[0]["false_answer"] == "24"
    failing = False
    result = run_false_answer_generation(
        config, root, human_corpus=tmp_path / "human.jsonl"
    )
    assert result == {"generated": 1, "reused": 1, "rejects": 0}
    assert len(calls) == 3
    assert [
        row["false_answer"] for row in read_jsonl(false_answers.answers_path(config))
    ] == ["24", "25"]


def test_changed_gold_answer_invalidates_saved_target(tmp_path, verifier):
    root, config = setup_data(tmp_path)
    false_answers._save_answers(
        false_answers.answers_path(config),
        {"q1": "24"},
        expected={"q1": ("How many episodes?", "23", ["23"])},
        source_path=root / "queries.jsonl",
        verifications={
            "q1": {"version": "v2", "validator_hash": validator_signature()}
        },
    )
    rows = read_jsonl(root / "queries.jsonl")
    rows[0]["metadata"].update(answer="24", answer_aliases=["24"])
    write_jsonl(root / "queries.jsonl", rows)
    candidates = false_answers.attach_saved_answers(
        load_generation_candidates(root, include_text=False), config
    )
    assert "false_answer" not in candidates[0]
