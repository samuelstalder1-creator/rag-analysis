from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from heapq import nsmallest
from typing import Sequence

from evlab.schema import Doc, Hit, Query

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is",
    "of", "on", "or", "the", "to", "was", "were", "what", "when", "where", "which", "who",
    "whom", "whose", "why",
}


class BM25Retriever:
    name = "bm25"

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, remove_query_stopwords: bool = True) -> None:
        self.k1 = k1
        self.b = b
        self.remove_query_stopwords = remove_query_stopwords
        self.docs: list[Doc] = []
        self.doc_ids: list[str] = []
        self.posting_lists: dict[str, list[tuple[int, float]]] = {}
        self.document_frequencies: Counter[str] = Counter()
        self.num_documents = 0

    def index(self, docs: Sequence[Doc]) -> None:
        self.docs = list(docs)
        self.doc_ids = [doc.doc_id for doc in self.docs]
        tokens_by_doc = [_tokenize(doc.index_text) for doc in self.docs]
        lengths = [len(tokens) for tokens in tokens_by_doc]
        avg_len = sum(lengths) / len(lengths) if lengths else 0.0
        term_frequencies = [Counter(tokens) for tokens in tokens_by_doc]
        self.document_frequencies = Counter(term for tf in term_frequencies for term in tf)
        posting_lists: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        for idx, tf in enumerate(term_frequencies):
            doc_len = lengths[idx]
            for term, frequency in tf.items():
                denominator = frequency + self.k1 * (1 - self.b + self.b * doc_len / (avg_len or 1.0))
                posting_lists[term].append((idx, (frequency * (self.k1 + 1)) / denominator))
        self.posting_lists = dict(posting_lists)
        self.num_documents = len(self.docs)

    def search(self, queries: Sequence[Query], top_k: int) -> dict[str, list[Hit]]:
        return {query.query_id: self.search_one(query, top_k) for query in queries}

    def search_one(self, query: Query, top_k: int) -> list[Hit]:
        query_terms = _tokenize(query.text)
        if self.remove_query_stopwords:
            filtered = [term for term in query_terms if term not in QUERY_STOPWORDS]
            query_terms = filtered or query_terms
        scores: Counter[int] = Counter()
        for term, query_count in Counter(query_terms).items():
            df = self.document_frequencies.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (self.num_documents - df + 0.5) / (df + 0.5))
            for idx, doc_weight in self.posting_lists.get(term, []):
                scores[idx] += query_count * idf * doc_weight
        top_scored = nsmallest(top_k, [(-score, self.doc_ids[idx]) for idx, score in scores.items()])
        if len(top_scored) < top_k:
            seen = {doc_id for _, doc_id in top_scored}
            for doc_id in self.doc_ids:
                if doc_id not in seen:
                    top_scored.append((-0.0, doc_id))
                if len(top_scored) == top_k:
                    break
        return [
            Hit(query_id=query.query_id, doc_id=doc_id, rank=rank, score=-neg_score, retriever=self.name)
            for rank, (neg_score, doc_id) in enumerate(top_scored, start=1)
        ]


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
