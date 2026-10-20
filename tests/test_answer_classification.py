import json

import pytest

from evlab.data.answer_classification import classify_answer_set, split_answer_flags
from evlab.io import read_jsonl, write_json, write_jsonl


@pytest.mark.parametrize(("question", "aliases", "kind"), [
    ("Who were the three recipients?", ["Alice", "Bob", "Carol"], "list"),
    ("What are the colors of the flag?", ["red", "blue"], "list"),
    ("Who wrote the song?", ["Alice and Bob", "Alice", "Bob"], "list"),
    ("Who wrote the song?", ["Alice", "Bob"], "unresolved"),
    ("When did it open?", ["1970", "1971"], "unresolved"),
    ("What is it called?", ["at-sign", "At sign"], "variants"),
    ("Where was the movie filmed?", ["London", "England"], "unresolved"),
    ("Who are you in this game?", ["Agent", "third person"], "unresolved"),
])
def test_classification_does_not_invent_conflicts(question, aliases, kind):
    assert classify_answer_set(question, aliases)["kind"] == kind


def test_review_must_match_answers():
    with pytest.raises(ValueError, match="no longer match"):
        classify_answer_set("Who?", ["Alice"], {"kind": "variants", "aliases": ["Bob"]})


def test_split_updates_flags_candidates_report_and_review_queue(tmp_path):
    rows = []
    cases = [
        ("q0", "How many?", ["23"], "single"),
        ("q1", "Who leads?", ["Alice Adams", "A. Adams"], "list_or_conflict"),
        ("q2", "What are the colors?", ["red", "blue"], "list_or_conflict"),
        ("q3", "How many episodes?", ["24", "21"], "list_or_conflict"),
        ("q4", "When did it open?", ["1970", "1971"], "list_or_conflict"),
    ]
    for qid, question, aliases, kind in cases:
        rows.append({"_id": qid, "text": question, "metadata": {
            "answer": aliases[0], "answer_aliases": aliases, "answer_kind": kind,
            "relevant_docs": ["d" + qid[1:]], "docs_with_answer": ["d" + qid[1:]],
            "flags": {"answer_list_or_conflict": kind == "list_or_conflict", "no_qrels": False},
            "usable_for": {"retrieval": True, "reader": True, "generation": kind == "single"},
        }})
    write_jsonl(tmp_path / "queries.jsonl", rows)
    write_jsonl(tmp_path / "rag_queries.jsonl", [{"query_id": qid} for qid, *_ in cases])
    existing = {"query_id": "q0", "query": "How many?", "answer": "23", "answer_aliases": ["23"],
                "answer_kind": "single", "h_docs": [{"doc_id": "d0", "title": "", "text": "23",
                                                      "llm_copy_keeps_answer": True}], "other_relevant_docs": []}
    write_jsonl(tmp_path / "generation_candidates.jsonl", [existing])
    write_jsonl(tmp_path / "human.jsonl", [
        {"_id": "d0", "title": "", "text": "23"},
        {"_id": "d1", "title": "Leader", "text": "Alice Adams is also known as A. Adams."},
        {"_id": "d2", "title": "Colors", "text": "red and blue"},
        {"_id": "d3", "title": "Season", "text": "The season has 24 episodes."},
        {"_id": "d4", "title": "Opening", "text": "1970 and 1971 refer to different openings."},
    ])
    write_json(tmp_path / "report.json", {"flag_answer_list_or_conflict": 4})
    reviews = {
        "q1": {"kind": "variants", "aliases": ["Alice Adams", "A. Adams"], "reason": "Explicit alias.",
               "evidence": [{"doc_id": "d1", "quote": "also known as A. Adams"}]},
        "q3": {"kind": "conflict", "aliases": ["24", "21"], "reason": "Competing episode counts.",
               "evidence": [{"doc_id": "d3", "quote": "24 episodes"}]},
    }
    write_json(tmp_path / "reviews.json", reviews)
    kwargs = {"human_corpus": tmp_path / "human.jsonl", "reviews_file": tmp_path / "reviews.json"}
    original_queries = (tmp_path / "queries.jsonl").read_bytes()
    invalid_reviews = json.loads(json.dumps(reviews))
    invalid_reviews["q1"]["evidence"][0]["quote"] = "Quote absent from the source"
    write_json(tmp_path / "reviews.json", invalid_reviews)
    with pytest.raises(ValueError, match="evidence does not match"):
        split_answer_flags(tmp_path, **kwargs)
    assert (tmp_path / "queries.jsonl").read_bytes() == original_queries
    write_json(tmp_path / "reviews.json", reviews)
    result = split_answer_flags(tmp_path, **kwargs)
    assert result["classification"] == {"variants": 1, "list": 1, "conflict": 1, "unresolved": 1}
    assert result["new_generation_query_ids"] == ["q1"]
    migrated = {row["_id"]: row["metadata"] for row in read_jsonl(tmp_path / "queries.jsonl")}
    assert all("answer_list_or_conflict" not in row["flags"] for row in migrated.values())
    assert migrated["q2"]["flags"]["answer_is_list"]
    assert migrated["q3"]["flags"]["answer_conflict"]
    assert migrated["q4"]["flags"]["answer_review_required"]
    assert not migrated["q4"]["flags"]["answer_conflict"]
    assert read_jsonl(tmp_path / "answer_review.jsonl")[0]["query_id"] == "q4"
    candidates = read_jsonl(tmp_path / "generation_candidates.jsonl")
    assert candidates[0] == existing
    assert [row["query_id"] for row in candidates] == ["q0", "q1"]
    second = split_answer_flags(tmp_path, **kwargs)
    assert second["classification"] == result["classification"]
    assert second["new_generation_candidates"] == 0
    assert read_jsonl(tmp_path / "generation_candidates.jsonl") == candidates
    report = json.loads((tmp_path / "report.json").read_text())
    assert "flag_answer_list_or_conflict" not in report
    assert report["flag_answer_is_list"] == 1
    assert report["flag_answer_conflict"] == 1
