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



def test_clean_dataset_queries_and_fixed_corpus_metrics(tmp_path):
    from evlab.corpus.builder import build_corpus_with_report
    from evlab.data.nqplus import load_qrels, load_retrieval_queries
    from evlab.evaluation.retrieval import evaluate_retrieval
    from evlab.evaluation.source import evaluate_origin_counts
    from evlab.schema import Hit

    root, human = _dataset(tmp_path)
    queries = load_retrieval_queries(root)
    assert [q.query_id for q in queries] == ["test1", "test2"]
    assert queries[1].relevant_doc_ids == ("doc10", "doc11") and queries[0].answer == "23"

    _write(tmp_path / "correct.jsonl", [{"_id": "doc6", "title": "Chicago Fire (season 4)", "text": "The season had 23 episodes."}])
    _write(tmp_path / "false.jsonl", [{"_id": "doc6", "title": "Chicago Fire (season 4)", "text": "The season had 25 episodes."}])
    config = {"data": {"nqplus_dir": str(root), "human_corpus": str(human)},
              "corpus": {"fixed": {"correct": {"path": str(tmp_path / "correct.jsonl")},
                                   "poisoned": {"path": str(tmp_path / "false.jsonl")}}}}
    docs, report = build_corpus_with_report(config)
    by_id = {d.doc_id: d for d in docs}
    assert report["sources"]["correct"]["documents"] == 1 and len(docs) == 4 + 2
    rewrite = next(d for d in docs if d.condition == "correct")
    poison = next(d for d in docs if d.condition == "false")
    assert rewrite.root_id == poison.root_id == "doc6" and rewrite.query_id == "test1"

    # Only the poisoned copy is retrieved: relaxed recall still counts the relevant document.
    run = {"test1": [Hit("test1", poison.doc_id, 1, 1.0, "bm25")], "test2": []}
    qrels = load_qrels(root)
    assert evaluate_retrieval(run, qrels, docs_by_id=by_id, relevance_mode="relaxed", cutoffs=(1,))["recall@1"] == 0.5
    assert evaluate_retrieval(run, qrels, docs_by_id=by_id, relevance_mode="strict", cutoffs=(1,))["recall@1"] == 0.0
    from evlab.evaluation.source import evaluate_version_recall

    # test1 has one relevant document (doc6); test2 two, none found. Poisoned copy found for test1.
    version = evaluate_version_recall(run, by_id, qrels, cutoffs=(1,))
    assert version == {"r_human@1": 0.0, "r_rewrite@1": 0.0, "r_poison@1": 0.5}
    # A copy of another question's document is not counted.
    foreign = {"test2": [Hit("test2", poison.doc_id, 1, 1.0, "bm25")], "test1": []}
    assert evaluate_version_recall(foreign, by_id, qrels, cutoffs=(1,))["r_poison@1"] == 0.0
    counts = evaluate_origin_counts(run, by_id, cutoffs=(1,))
    assert counts == {"rewrite@1": 0.0, "poison@1": 0.5, "human_docs@1": 0.0}
    config["corpus"]["fixed"]["poisoned"]["enabled"] = False
    assert len(build_corpus_with_report(config)[0]) == 5
