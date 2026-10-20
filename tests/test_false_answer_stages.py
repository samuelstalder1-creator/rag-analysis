import pytest

from evlab.cli import main
from evlab.generation import false_answers, pipeline
from evlab.generation.false_answers import import_legacy_false_answers, run_false_answer_generation
from evlab.generation.registry import read_registry
from evlab.io import read_jsonl, write_json, write_jsonl


def setup_data(tmp_path, *, two_questions=False):
    root = tmp_path / "nqplus"
    queries = [{"_id": "q1", "text": "How many episodes?", "metadata": {
        "answer": "23", "answer_aliases": ["23"], "untouched": "keep me"}}]
    candidate = {"query_id": "q1", "query": "How many episodes?", "answer": "23", "answer_aliases": ["23"],
                 "h_docs": [{"doc_id": did, "title": "23 episodes", "text": "There were 23 episodes."}
                            for did in ("d1", "d2")]}
    candidates = [candidate]
    if two_questions:
        queries.append({"_id": "q2", "text": "How many seasons?", "metadata": {"answer": "24", "answer_aliases": ["24"]}})
        candidates.append({"query_id": "q2", "query": "How many seasons?", "answer": "24", "answer_aliases": ["24"],
                           "h_docs": [{"doc_id": "d3", "title": "Seasons", "text": "There were 24 seasons."}]})
    write_jsonl(root / "queries.jsonl", queries)
    write_jsonl(root / "generation_candidates.jsonl", candidates)
    config = {"output_dir": str(tmp_path / "output"), "cache_path": str(tmp_path / "llm.sqlite"),
              "conditions": ["false"], "llm": {"provider": "abstain", "model": "test/stages"}}
    return root, config


def test_cli_two_stages_share_saved_answer_and_resume(tmp_path, monkeypatch):
    root, config = setup_data(tmp_path)
    config_path = tmp_path / "config.json"
    write_json(config_path, {"data": {"nqplus_dir": str(root)}, "generation": config})
    calls = []

    class AnswerClient:
        provider = "test"
        model = "test/stages"

        def generate(self, prompt, **kwargs):
            assert "Generate three plausible but incorrect answers" in prompt
            calls.append("answer")
            return '{"candidates": ["24"]}'

    class DocumentClient:
        provider = "test"
        model = "test/stages"

        def generate(self, prompt, **kwargs):
            assert "Generate three plausible but incorrect answers" not in prompt
            calls.append("document")
            return prompt.split("Original Text: ", 1)[1].split("\n\nRewrite the text.", 1)[0]

    monkeypatch.setattr(false_answers, "build_llm_client", lambda config: AnswerClient())
    monkeypatch.setattr(pipeline, "build_llm_client", lambda config: DocumentClient())
    main(["generate-false-answers", "--config", str(config_path)])
    stored = read_jsonl(root / "queries.jsonl")[0]
    assert stored["metadata"]["false_answer"] == "24"
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
    assert all(doc["title"] == "24 episodes" and "24" in doc["text"] and "23" not in doc["text"] for doc in docs)
    assert calls == ["answer", "document"]
    main(["generate-false-answers", "--config", str(config_path)])
    main(["generate-false-documents", "--config", str(config_path)])
    assert calls == ["answer", "document"]
    assert read_jsonl(tmp_path / "output" / filename) == docs
    assert not list((tmp_path / "output").glob("*progress.json"))


@pytest.mark.parametrize("answer", [None, "", "23"])
def test_document_stage_fails_before_model_calls_without_valid_saved_answer(tmp_path, monkeypatch, answer):
    root, config = setup_data(tmp_path)
    candidates = read_jsonl(root / "generation_candidates.jsonl")
    candidates[0]["false_answer"] = answer
    def no_client(config):
        raise AssertionError("The document stage must validate input before constructing a client")
    monkeypatch.setattr(pipeline, "build_llm_client", no_client)
    with pytest.raises(ValueError, match="generate-false-answers"):
        pipeline.run_generation_pipeline(config, candidates)
    assert not (tmp_path / "output").exists()


def test_answer_checkpoints_survive_failure_and_retry_only_missing(tmp_path, monkeypatch):
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
                return '{"candidates": ["25"]}'
            return '{"candidates": ["24"]}'

    monkeypatch.setattr(false_answers, "build_llm_client", lambda config: Client())
    result = run_false_answer_generation(config, root)
    assert result == {"generated": 1, "reused": 0, "imported": 0, "rejects": 1}
    rows = read_jsonl(root / "queries.jsonl")
    assert rows[0]["metadata"]["false_answer"] == "24"
    assert "false_answer" not in rows[1]["metadata"]
    failing = False
    result = run_false_answer_generation(config, root)
    assert result == {"generated": 1, "reused": 1, "imported": 0, "rejects": 0}
    assert len(calls) == 3
    assert [row["metadata"]["false_answer"] for row in read_jsonl(root / "queries.jsonl")] == ["24", "25"]


def test_import_recovers_legacy_targets_and_rejects_conflicting_saved_answers(tmp_path):
    root, config = setup_data(tmp_path)
    output = tmp_path / "output"
    legacy = {"query_id": "q1", "target_answer": "24", "generator_model": "test/stages"}
    write_jsonl(output / "legacy-backup" / "false_docs.jsonl", [legacy, legacy])
    assert import_legacy_false_answers(root, output) == 1
    assert import_legacy_false_answers(root, output) == 0
    assert read_jsonl(root / "queries.jsonl")[0]["metadata"]["false_answer"] == "24"
    registry = read_registry(output)
    job = next(iter(registry["answer_generations"].values()))
    assert job["origin"] == "legacy_documents" and job["answer_count"] == 1
    assert job["query_ids"] == ["q1"]
    rows = read_jsonl(root / "queries.jsonl")
    rows[0]["metadata"]["false_answer"] = "25"
    write_jsonl(root / "queries.jsonl", rows)
    with pytest.raises(ValueError, match="Conflicting existing and legacy"):
        import_legacy_false_answers(root, output)
    assert read_jsonl(root / "queries.jsonl")[0]["metadata"]["false_answer"] == "25"
