from pathlib import Path

from evlab.data.nqplus import load_generation_candidates, load_retrieval_queries, validate_nqplus


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
    assert len(load_generation_candidates(FIXTURES)) == 1
