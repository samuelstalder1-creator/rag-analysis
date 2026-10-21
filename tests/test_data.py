from pathlib import Path
import shutil

import pytest

from evlab.data.nqplus import load_generation_candidates, load_retrieval_queries, load_rag_queries, load_qrels, validate_nqplus
from evlab.io import write_jsonl


FIXTURES = Path(__file__).parent / "fixtures" / "nqplus"


def test_load_nqplus_queries():
    queries = load_retrieval_queries(FIXTURES)
    assert len(queries) == 2
    assert queries[0].answer == "23"
    assert queries[0].relevant_doc_ids == ("doc6",)


def test_validate_nqplus_fixture():
    stats = validate_nqplus(FIXTURES)
    assert stats["retrieval_queries"] == 2
    assert stats["generation_candidates"] == 1
    assert len(load_generation_candidates(FIXTURES, FIXTURES.parent / "corpus/human.jsonl")) == 1


def test_queries_alone_supply_reader_generation_and_relevance(tmp_path):
    shutil.copyfile(FIXTURES / "queries.jsonl", tmp_path / "queries.jsonl")
    assert len(load_rag_queries(tmp_path)) == 2
    assert [(q.query_id, q.doc_id) for q in load_qrels(tmp_path)] == [("test1", "doc6"), ("test2", "doc7")]
    candidates = load_generation_candidates(tmp_path, include_text=False)
    assert candidates[0]["h_docs"] == [{"doc_id": "doc6"}]
    # Obsolete exports cannot override the current query annotations.
    (tmp_path / "generation_candidates.jsonl").write_text("invalid stale export")
    (tmp_path / "rag_queries.jsonl").write_text("invalid stale export")
    assert load_generation_candidates(tmp_path, include_text=False) == candidates
    assert len(load_rag_queries(tmp_path)) == 2


def test_candidates_read_current_source_and_reject_missing_documents(tmp_path):
    shutil.copyfile(FIXTURES / "queries.jsonl", tmp_path / "queries.jsonl")
    source = tmp_path / "human.jsonl"
    write_jsonl(source, [{"_id": "doc6", "title": "Season four", "text": "Updated source: 23 episodes."}])
    doc = load_generation_candidates(tmp_path, source)[0]["h_docs"][0]
    assert doc["text"] == "Updated source: 23 episodes."
    assert doc["aliases_in_doc"] == ["23"]
    write_jsonl(source, [])
    with pytest.raises(ValueError, match="Missing source documents"):
        load_generation_candidates(tmp_path, source)
