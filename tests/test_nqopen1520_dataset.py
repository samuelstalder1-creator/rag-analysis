import json

from evlab.data.nqplus import is_clean_dataset, load_generation_candidates
from evlab.generation.false_answers import attach_saved_answers, valid_saved_answer


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _dataset(tmp_path):
    root = tmp_path / "nqopen1520"
    _write(root / "queries.jsonl", [
        {"_id": "test1", "text": "how many episodes are in chicago fire season 4", "metadata": {}},
        {"_id": "test2", "text": "who sings love will keep us alive", "metadata": {}},
    ])
    (root / "qrels").mkdir()
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\ntest1\tdoc6\t1\ntest2\tdoc10\t1\ntest2\tdoc11\t1\n", encoding="utf-8")
    _write(root / "answers.jsonl", [
        {"_id": "test1", "answer": "23", "answer_aliases": ["23"], "false_answer": "25"},
        {"_id": "test2", "answer": "Timothy B. Schmit", "answer_aliases": ["Timothy B. Schmit"], "false_answer": "Don Henley"},
    ])
    corpus = tmp_path / "human.jsonl"
    _write(corpus, [
        {"_id": "doc6", "title": "Chicago Fire (season 4)", "text": "The season contained 23 episodes.", "metadata": {}},
        {"_id": "doc10", "title": "Love Will Keep Us Alive", "text": "Lead vocals by Timothy B. Schmit.", "metadata": {}},
        {"_id": "doc11", "title": "Other", "text": "Timothy B. Schmit sang it.", "metadata": {}},
        {"_id": "doc99", "title": "Unused", "text": "Not referenced.", "metadata": {}},
    ])
    return root, corpus


def test_clean_dataset_candidates(tmp_path):
    root, corpus = _dataset(tmp_path)
    assert is_clean_dataset(root)
    candidates = load_generation_candidates(root, corpus)
    assert [c["query_id"] for c in candidates] == ["test1", "test2"]
    assert [d["doc_id"] for d in candidates[1]["h_docs"]] == ["doc10", "doc11"]
    assert candidates[0]["h_docs"][0]["text"] == "The season contained 23 episodes."
    assert candidates[0]["h_docs"][0]["aliases_in_doc"] == ["23"]


def test_answers_jsonl_targets_are_attached_and_valid(tmp_path):
    root, corpus = _dataset(tmp_path)
    config = {"false_answer_source": "nqopen_import", "false_answers_file": str(root / "answers.jsonl"),
              "require_source_constraints": True, "validation": {"mode": "deterministic"}}
    candidates = attach_saved_answers(load_generation_candidates(root, corpus), config)
    assert [c["false_answer"] for c in candidates] == ["25", "Don Henley"]
    assert all(valid_saved_answer(c, require_semantic=False, require_source_constraints=True) for c in candidates)

