from __future__ import annotations

import math
import re
from collections import Counter
from collections import defaultdict
from heapq import nlargest
from typing import Sequence

from evlab.schema import Doc, Hit, Query

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "their", "to", "was", "were", "what",
    "when", "where", "which", "who", "whom", "whose", "why", "with",
}


class TfidfRetriever:
    name = "tfidf"

    def __init__(self) -> None:
        self.docs: list[Doc] = []
        self.doc_vectors: list[dict[str, float]] = []
        self.doc_norms: list[float] = []
        self.idf: dict[str, float] = {}
        self.posting_lists: dict[str, list[tuple[int, float]]] = {}

    def index(self, docs: Sequence[Doc]) -> None:
        self.docs = list(docs)
        tokenized = [_tokenize(doc.index_text) for doc in self.docs]
        df = Counter(term for tokens in tokenized for term in set(tokens))
        n = len(self.docs)
        self.idf = {term: math.log((1 + n) / (1 + freq)) + 1 for term, freq in df.items()}
        self.doc_vectors = [_tfidf(tokens, self.idf) for tokens in tokenized]
        self.doc_norms = [_norm(vec) for vec in self.doc_vectors]
        postings: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
        for idx, vec in enumerate(self.doc_vectors):
            norm = self.doc_norms[idx] or 1.0
            for term, value in vec.items():
                postings[term].append((idx, value / norm))
        self.posting_lists = dict(postings)

    def search(self, queries: Sequence[Query], top_k: int) -> dict[str, list[Hit]]:
        return {query.query_id: self.search_one(query, top_k) for query in queries}

    def search_one(self, query: Query, top_k: int) -> list[Hit]:
        qvec = _tfidf(_tokenize(query.text), self.idf)
        qnorm = _norm(qvec)
        scores: Counter[int] = Counter()
        for term, qvalue in qvec.items():
            if qvalue == 0:
                continue
            for idx, doc_weight in self.posting_lists.get(term, []):
                scores[idx] += (qvalue / (qnorm or 1.0)) * doc_weight
        scored = [(score, self.docs[idx].doc_id) for idx, score in scores.items()]
        if len(scored) < top_k:
            seen = {doc_id for _, doc_id in scored}
            for doc in self.docs:
                if doc.doc_id not in seen:
                    scored.append((0.0, doc.doc_id))
                if len(scored) == top_k:
                    break
        top = nlargest(top_k, scored)
        return [
            Hit(query_id=query.query_id, doc_id=doc_id, rank=rank, score=score, retriever=self.name)
            for rank, (score, doc_id) in enumerate(top, start=1)
        ]


def _tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


def _tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    return {term: count * idf.get(term, 0.0) for term, count in counts.items()}


def _norm(vec: dict[str, float]) -> float:
    return math.sqrt(sum(value * value for value in vec.values()))
