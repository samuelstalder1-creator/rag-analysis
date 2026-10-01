from evlab.evaluation.retrieval import evaluate_retrieval
from evlab.retrieval.bm25 import BM25Retriever
from evlab.schema import Doc, Qrel, Query


def test_bm25_finds_relevant_doc():
    docs = [
        Doc("human::doc6", "Chicago Fire", "The fourth season contained 23 episodes.", "human", provenance_root_id="doc6"),
        Doc("human::doc8", "Noise", "Mountain railways are unrelated.", "human", provenance_root_id="doc8"),
    ]
    query = Query("test1", "how many episodes are in chicago fire season 4", answer="23", relevant_doc_ids=("doc6",))
    retriever = BM25Retriever()
    retriever.index(docs)
    run = retriever.search([query], top_k=2)
    assert run["test1"][0].doc_id == "human::doc6"


def test_strict_does_not_count_generated_copy_as_human_relevant():
    docs = {
        "generated::doc6": Doc("generated::doc6", "Chicago Fire", "Season four had 23 episodes.", "generated", provenance_root_id="doc6"),
    }
    run = {"test1": []}
    run["test1"] = [
        type("HitObj", (), {"query_id": "test1", "doc_id": "generated::doc6", "rank": 1, "score": 1.0, "retriever": "x"})()
    ]
    qrels = [Qrel("test1", "doc6", 1)]
    strict = evaluate_retrieval(run, qrels, docs_by_id=docs, relevance_mode="strict", cutoffs=(1,))
    relaxed = evaluate_retrieval(run, qrels, docs_by_id=docs, relevance_mode="relaxed", cutoffs=(1,))
    assert strict["recall@1"] == 0.0
    assert relaxed["recall@1"] == 1.0
    assert relaxed["ndcg@1"] <= 1.0


def test_relaxed_deduplicates_human_and_generated_copy():
    docs = {
        "cocktail_llm::doc6": Doc(
            "cocktail_llm::doc6",
            "Chicago Fire",
            "Season four had 23 episodes.",
            "cocktail_llm",
            provenance_root_id="doc6",
        ),
        "human::doc6": Doc(
            "human::doc6",
            "Chicago Fire",
            "The fourth season contained 23 episodes.",
            "human",
            provenance_root_id="doc6",
        ),
    }
    run = {
        "test1": [
            type("HitObj", (), {"query_id": "test1", "doc_id": "cocktail_llm::doc6", "rank": 1, "score": 2.0, "retriever": "x"})(),
            type("HitObj", (), {"query_id": "test1", "doc_id": "human::doc6", "rank": 2, "score": 1.0, "retriever": "x"})(),
        ]
    }
    metrics = evaluate_retrieval(run, [Qrel("test1", "doc6", 1)], docs_by_id=docs, relevance_mode="relaxed", cutoffs=(2,))
    assert metrics["ndcg@2"] == 1.0
