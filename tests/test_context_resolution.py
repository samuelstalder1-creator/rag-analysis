import copy
import importlib.util
from pathlib import Path

import pytest

from evlab.data.nqplus import load_generation_candidates
from evlab.generation.false_answers import attach_saved_answers, valid_saved_answer
from evlab.io import file_sha256, read_jsonl, write_json, write_jsonl


def resolver():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/resolve_nqopen_context_cases.py"
    )
    spec = importlib.util.spec_from_file_location("context_resolution_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    source = tmp_path / "base"
    originals = [
        {
            "_id": "q1",
            "text": "Was the film published before the book?",
            "metadata": {
                "answer": "film",
                "answer_aliases": ["film"],
                "relevant_docs": ["d1"],
                "docs_with_answer": ["d1"],
                "usable_for": {"generation": True, "reader": True, "retrieval": True},
            },
        },
        {
            "_id": "q2",
            "text": "When did the airline start flying?",
            "metadata": {
                "answer": "November 1999",
                "answer_aliases": ["November 1999"],
                "relevant_docs": ["d2"],
                "docs_with_answer": ["d2"],
                "usable_for": {"generation": True, "reader": True, "retrieval": True},
            },
        },
        {
            "_id": "q3",
            "text": "How many episodes?",
            "metadata": {
                "answer": "23",
                "answer_aliases": ["23"],
                "relevant_docs": ["d3"],
                "docs_with_answer": ["d3"],
                "usable_for": {"generation": True, "reader": True, "retrieval": True},
            },
        },
    ]
    docs = [
        {
            "_id": "d1",
            "title": "A story",
            "text": "The novelization was published before the film.",
        },
        {
            "_id": "d2",
            "title": "An airline",
            "text": "The airline was established in November 1999.",
        },
        {"_id": "d3", "title": "A series", "text": "No episode count is provided."},
    ]
    audit = [
        {
            "query_id": "q1",
            "gold": "film",
            "false_answer": "radio drama",
            "target_origin": "cleaned_review",
            "context_review_required": True,
            "blocking_reasons": [],
        },
        {
            "query_id": "q2",
            "gold": "November 1999",
            "false_answer": "November 1998",
            "target_origin": "cleaned_review",
            "context_review_required": True,
            "blocking_reasons": [],
        },
        {
            "query_id": "q3",
            "gold": "23",
            "false_answer": "24",
            "target_origin": "original_nqopen_import",
            "context_review_required": False,
            "blocking_reasons": ["full_gold_missing_in_source"],
        },
    ]
    resolutions = {
        "version": 1,
        "method": "assistant_source_review",
        "resolutions": [
            {
                "query_id": "q1",
                "question": "Which published format preceded the film?",
                "gold": "novelization",
                "answer_aliases": ["novelization", "book"],
                "reason": "Correct the contradicted film gold.",
            },
            {
                "query_id": "q2",
                "question": "When was the airline established?",
                "reason": "The passage dates establishment, not flights.",
            },
        ],
    }
    write_jsonl(source / "queries.jsonl", originals)
    write_jsonl(source / "all_cases_checked.jsonl", audit)
    write_jsonl(source / "false_answers.jsonl", [])
    write_jsonl(source / "false_answer_review_cleaned.jsonl", [])
    write_json(source / "dataset.json", {"version": 1})
    write_jsonl(tmp_path / "human.jsonl", docs)
    write_json(tmp_path / "resolutions.json", resolutions)
    config = {
        "data": {
            "nqplus_dir": str(source / "resolved"),
            "human_corpus": str(tmp_path / "human.jsonl"),
        },
        "dataset_resolution": {
            "source_dir": str(source),
            "resolutions_file": str(tmp_path / "resolutions.json"),
            "expected_questions": 3,
            "expected_context_cases": 2,
        },
    }
    return config, resolutions


def test_resolution_versions_questions_golds_and_valid_target_receipts(tmp_path):
    config, _ = fixture(tmp_path)
    module = resolver()
    source = Path(config["dataset_resolution"]["source_dir"])
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    stats = module.build(config)
    assert stats["resolved_context_cases"] == 2
    assert stats["question_changes"] == 2 and stats["gold_corrections"] == 1
    assert stats["ready_false_answers"] == 2 and stats["source_blocked_questions"] == 1
    output = Path(config["data"]["nqplus_dir"])
    rows = read_jsonl(output / "queries.jsonl")
    assert rows[0]["metadata"]["answer"] == "novelization"
    assert "film" not in rows[0]["metadata"]["answer_aliases"]
    assert rows[0]["metadata"]["context_resolution"]["original_gold"] == "film"
    assert (
        rows[0]["metadata"]["context_resolution"]["original_question"]
        == "Was the film published before the book?"
    )
    assert {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()} == before
    candidates = load_generation_candidates(output, config["data"]["human_corpus"])
    gen = {
        "false_answers_file": str(output / "false_answers.jsonl"),
        "false_answer_source": "nqopen_import",
        "validation": {"mode": "deterministic"},
    }
    attached = attach_saved_answers(candidates, gen)
    assert [
        c["query_id"]
        for c in attached
        if valid_saved_answer(
            c, require_semantic=False, require_source_constraints=True
        )
    ] == ["q1", "q2"]
    resolved = read_jsonl(output / "context_cases_resolved.jsonl")
    assert all(r["source_constraints"]["eligible"] for r in resolved)
    assert all(
        r["source_review_completed"] and not r["semantic_verified"] for r in resolved
    )


def test_target_present_in_source_cannot_be_marked_resolved(tmp_path):
    config, spec = fixture(tmp_path)
    spec["resolutions"][0]["false_answer"] = "film"
    write_json(config["dataset_resolution"]["resolutions_file"], spec)
    with pytest.raises(ValueError, match="original relevant document"):
        resolver().build(config)
    assert not Path(config["data"]["nqplus_dir"]).exists()


def test_original_relevant_source_cannot_be_hidden_to_avoid_target_collision(tmp_path):
    config, spec = fixture(tmp_path)
    source = Path(config["dataset_resolution"]["source_dir"])
    queries = read_jsonl(source / "queries.jsonl")
    queries[0]["metadata"]["relevant_docs"].append("d4")
    write_jsonl(source / "queries.jsonl", queries)
    corpus = read_jsonl(config["data"]["human_corpus"])
    corpus.append(
        {
            "_id": "d4",
            "title": "Other adaptation",
            "text": "A radio drama adaptation followed.",
        }
    )
    write_jsonl(config["data"]["human_corpus"], corpus)
    spec["resolutions"][0]["answer_doc_ids"] = ["d1"]
    write_json(config["dataset_resolution"]["resolutions_file"], spec)
    with pytest.raises(ValueError, match="original relevant document"):
        resolver().build(config)
    assert not Path(config["data"]["nqplus_dir"]).exists()


def test_new_answer_document_must_be_an_original_relevant_document(tmp_path):
    config, spec = fixture(tmp_path)
    spec["resolutions"][0]["answer_doc_ids"] = ["d2"]
    write_json(config["dataset_resolution"]["resolutions_file"], spec)
    with pytest.raises(ValueError, match="original relevant document IDs"):
        resolver().build(config)
    assert not Path(config["data"]["nqplus_dir"]).exists()
