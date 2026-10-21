import hashlib
from pathlib import Path

import pytest

from evlab.generation import false_answers
from evlab.generation.maintenance import cleanup_false_generation, generation_status
from evlab.generation.output import GenerationOutput
from evlab.generation.semantic import validator_signature
from evlab.generation.registry import read_registry, update_registry
from evlab.io import read_jsonl, write_jsonl


def prepare(tmp_path):
    original = tmp_path / "cocktail/nq"
    docs = [
        {"_id": "d1", "title": "Season", "text": "There were 23 episodes."},
        {"_id": "d2", "title": "Series", "text": "There were 10 episodes."},
    ]
    for name in ("human", "llama-2-7b-chat-tmp0.2"):
        write_jsonl(original / f"corpus/{name}.jsonl", docs)
    write_jsonl(
        original / "queries.jsonl",
        [{"_id": "q1", "text": "How many?"}, {"_id": "q2", "text": "How many more?"}],
    )
    (original / "qrels").mkdir()
    (original / "qrels/test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n"
    )
    nqplus = tmp_path / "nqplus/clean"
    write_jsonl(
        nqplus / "queries.jsonl",
        [
            {
                "_id": "q1",
                "text": "How many?",
                "metadata": {
                    "answer": "23",
                    "answer_aliases": ["23"],
                    "false_answer": "24",
                },
            },
            {
                "_id": "q2",
                "text": "How many more?",
                "metadata": {"answer": "10", "answer_aliases": ["10"]},
            },
        ],
    )
    rows = read_jsonl(nqplus / "queries.jsonl")
    for row, did in zip(rows, ("d1", "d2")):
        row["metadata"].update(
            relevant_docs=[did],
            docs_with_answer=[did],
            usable_for={"retrieval": True, "reader": True, "generation": True},
        )
    write_jsonl(nqplus / "queries.jsonl", rows)
    output = tmp_path / "generated"
    false_answers._save_answers(
        output / "generated_false_answers.jsonl",
        {"q1": "24"},
        expected={"q1": ("How many?", "23", ["23"])},
        source_path=nqplus / "queries.jsonl",
        verifications={
            "q1": {"version": "v2", "validator_hash": validator_signature()}
        },
    )
    write_jsonl(output / "corpus_0001.jsonl", docs)
    write_jsonl(
        output / "corpus_0002.jsonl",
        [{"_id": "d1", "title": "Season", "text": "There were 24 episodes."}],
    )

    def register(registry):
        registry["files"] = {
            "corpus_0001.jsonl": {
                "condition": "correct",
                "source": "human_corpus",
                "model": "test",
                "document_count": 2,
            },
            "corpus_0002.jsonl": {
                "condition": "false",
                "source": "candidates",
                "model": "test",
                "document_count": 1,
            },
        }
        registry["answer_generations"] = {"answers": {"query_ids": ["q1"]}}

    update_registry(output, register)
    write_jsonl(
        output / "rejects.jsonl",
        [
            {
                "condition": "false",
                "stage": "false_validation",
                "target_answer": "24",
                "generated_text": "False text",
            },
            {"stage": "false_answer", "reason": "Failed target"},
            {
                "condition": "correct",
                "stage": "correct_validation",
                "generated_text": "Correct reject",
            },
        ],
    )
    write_jsonl(
        output / "old/false_docs.jsonl", [{"condition": "false", "target_answer": "24"}]
    )
    write_jsonl(output / "test--false-answer-rejects.jsonl", [{"target_answer": "24"}])
    cache = tmp_path / "cache.sqlite"
    cache.write_bytes(b"old generated responses")
    Path(str(cache) + "-wal").write_bytes(b"old responses")
    config = {
        "data": {
            "human_corpus": str(original / "corpus/human.jsonl"),
            "nqplus_dir": str(nqplus),
        },
        "generation": {"output_dir": str(output), "cache_path": str(cache)},
    }
    return original, nqplus, output, config


def test_cleanup_removes_false_artifacts_and_preserves_originals_and_correct(tmp_path):
    original, nqplus, output, config = prepare(tmp_path)
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in original.rglob("*")
        if p.is_file()
    }
    correct = (output / "corpus_0001.jsonl").read_bytes()
    queries_before = (nqplus / "queries.jsonl").read_bytes()
    status = generation_status(config)
    assert status["original_cocktail"]["queries"] == 2
    assert status["original_cocktail"]["qrels"] == 2
    assert status["original_cocktail"]["provided_llm_documents"] == 2
    assert status["prepared_nqplus"]["saved_false_answers"] == 1
    assert status["coverage"]["false_candidate_generated"] == 1
    result = cleanup_false_generation(config)
    assert result["false_answer_fields_removed"] == 0
    assert result["false_document_rows_removed"] == 2
    assert (output / "corpus_0001.jsonl").read_bytes() == correct
    assert not (output / "corpus_0002.jsonl").exists()
    assert not list(output.rglob("false_docs.jsonl"))
    assert not (tmp_path / "cache.sqlite").exists()
    assert not Path(str(tmp_path / "cache.sqlite") + "-wal").exists()
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in before.items()
    )
    assert read_jsonl(output / "rejects.jsonl") == [
        {
            "condition": "correct",
            "stage": "correct_validation",
            "generated_text": "Correct reject",
        }
    ]
    assert (nqplus / "queries.jsonl").read_bytes() == queries_before
    assert not (output / "generated_false_answers.jsonl").exists()
    registry = read_registry(output)
    assert "answer_generations" not in registry
    assert set(registry["files"]) == {"corpus_0001.jsonl"}
    status = generation_status(config)
    assert status["coverage"] == {
        "correct_human_generated": 2,
        "correct_human_missing": 0,
        "candidate_ids_with_correct_copy": 2,
        "false_candidate_generated": 0,
        "false_candidate_missing": 2,
    }
    assert status["prepared_nqplus"]["saved_false_answers"] == 0
    assert status["prepared_nqplus"]["missing_false_answers"] == 2
    second = cleanup_false_generation(config)
    assert second["deleted_files"] == []
    assert second["false_answer_fields_removed"] == 0


def test_reset_does_not_resurrect_legacy_false_answers_or_documents(
    tmp_path, monkeypatch, verifier
):
    original, nqplus, output, config = prepare(tmp_path)
    cleanup_false_generation(config)
    write_jsonl(
        output / "old/false_docs.jsonl",
        [{"query_id": "q1", "condition": "false", "target_answer": "24"}],
    )

    class Client:
        provider = "test"
        model = "test"

        def generate(self, prompt, **kwargs):
            return '{"candidates": ["25"]}'

    monkeypatch.setattr(false_answers, "build_llm_client", lambda config: Client())
    generation = {
        **config["generation"],
        "llm": {"provider": "abstain", "model": "test"},
    }
    false_answers.run_false_answer_generation(
        generation, nqplus, human_corpus=original / "corpus/human.jsonl", limit=1
    )
    assert read_jsonl(false_answers.answers_path(generation))[0]["false_answer"] == "25"
    assert read_jsonl(nqplus / "queries.jsonl")[0]["metadata"]["false_answer"] == "24"
    write_jsonl(
        output / "false_docs.jsonl",
        [
            {
                "query_id": "q1",
                "condition": "false",
                "generator_model": "test",
                "sample_idx": 0,
                "provenance_root_id": "d1",
                "title": "Old",
                "text": "Old false text",
            }
        ],
    )
    writer = GenerationOutput(output, generation, "test", {"false"})
    assert not writer.contains("false", 0, "d1")


def test_cleanup_rejects_original_dataset_as_prepared_directory(tmp_path):
    original, nqplus, output, config = prepare(tmp_path)
    config["data"]["nqplus_dir"] = str(original)
    with pytest.raises(ValueError, match="separate from the original"):
        cleanup_false_generation(config)
    assert (output / "corpus_0002.jsonl").exists()
