from __future__ import annotations

import hashlib
import math
import re
from heapq import nlargest
from pathlib import Path
from typing import Sequence

from evlab.schema import Doc, Hit, Query
from evlab.io import stable_hash


class HashDenseRetriever:
    """Dependency-free dense-like baseline for tests and smoke runs.

    Production dense retrieval should use `SentenceTransformerRetriever`, but
    this keeps the pipeline runnable before heavy model dependencies are synced.
    """

    name = "hash_dense"

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.docs: list[Doc] = []
        self.vectors: list[list[float]] = []

    def index(self, docs: Sequence[Doc]) -> None:
        self.docs = list(docs)
        self.vectors = [_hash_vector(doc.embedding_text, self.dimensions) for doc in self.docs]

    def search(self, queries: Sequence[Query], top_k: int) -> dict[str, list[Hit]]:
        return {query.query_id: self.search_one(query, top_k) for query in queries}

    def search_one(self, query: Query, top_k: int) -> list[Hit]:
        qvec = _hash_vector(query.text, self.dimensions)
        top = nlargest(top_k, [(_cosine(qvec, vec), doc.doc_id) for doc, vec in zip(self.docs, self.vectors)])
        return [
            Hit(query_id=query.query_id, doc_id=doc_id, rank=rank, score=score, retriever=self.name)
            for rank, (score, doc_id) in enumerate(top, start=1)
        ]


class SentenceTransformerRetriever:
    name = "sentence_transformer"

    def __init__(
        self,
        model: str,
        *,
        batch_size: int = 128,
        search_batch_size: int = 32,
        normalize: bool = True,
        local_files_only: bool = False,
        cache_dir: str | Path = "cache/embeddings",
        device: str | None = None,
        query_prefix: str = "",
        doc_prefix: str = "",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install dense retrieval support with: uv sync --extra dense") from exc
        self.model_name = model
        self.batch_size = batch_size
        self.search_batch_size = search_batch_size
        self.normalize = normalize
        kwargs = {"local_files_only": local_files_only}
        if device:
            kwargs["device"] = device
        self.encoder = SentenceTransformer(model, **kwargs)
        self.cache_dir = Path(cache_dir) / _safe_name(model)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.docs: list[Doc] = []
        self.vectors = None
        self.name = f"dense:{model}"
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def index(self, docs: Sequence[Doc]) -> None:
        import numpy as np  # type: ignore

        self.docs = list(docs)
        cache_key = stable_hash(
            {
                "model": self.model_name,
                "normalize": self.normalize,
                "doc_prefix": self.doc_prefix,
                "docs": [(doc.doc_id, hashlib.sha256(doc.embedding_text.encode("utf-8")).hexdigest()) for doc in self.docs],
            }
        )
        vector_path = self.cache_dir / f"{cache_key}.npy"
        ids_path = self.cache_dir / f"{cache_key}.ids"
        if vector_path.exists() and ids_path.exists():
            cached_ids = ids_path.read_text(encoding="utf-8").splitlines()
            if cached_ids == [doc.doc_id for doc in self.docs]:
                self.vectors = np.load(vector_path, mmap_mode="r")
                return
        prefix_vectors, prefix_len = self._load_cached_prefix([doc.doc_id for doc in self.docs])
        texts = [self.doc_prefix + doc.embedding_text for doc in self.docs[prefix_len:]]
        vectors = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=True,
            convert_to_numpy=True,
        ) if texts else np.empty((0, prefix_vectors.shape[1] if prefix_vectors is not None else 0), dtype=np.float32)
        vectors = np.asarray(vectors, dtype=np.float32)
        if prefix_vectors is not None:
            vectors = np.vstack([np.asarray(prefix_vectors, dtype=np.float32), vectors])
        np.save(vector_path, vectors)
        ids_path.write_text("\n".join(doc.doc_id for doc in self.docs) + "\n", encoding="utf-8")
        self.vectors = np.load(vector_path, mmap_mode="r")

    def _load_cached_prefix(self, doc_ids: list[str]):
        import numpy as np  # type: ignore

        best_ids_path: Path | None = None
        best_len = 0
        for ids_path in self.cache_dir.glob("*.ids"):
            cached_ids = ids_path.read_text(encoding="utf-8").splitlines()
            if not cached_ids or len(cached_ids) > len(doc_ids):
                continue
            if doc_ids[: len(cached_ids)] == cached_ids and len(cached_ids) > best_len:
                vector_path = ids_path.with_suffix(".npy")
                if vector_path.exists():
                    best_ids_path = ids_path
                    best_len = len(cached_ids)
        if best_ids_path is None or best_len == 0:
            return None, 0
        vector_path = best_ids_path.with_suffix(".npy")
        return np.load(vector_path, mmap_mode="r"), best_len

    def search(self, queries: Sequence[Query], top_k: int) -> dict[str, list[Hit]]:
        if self.vectors is None:
            raise RuntimeError("Retriever must be indexed before search")
        import numpy as np  # type: ignore

        query_list = list(queries)
        query_vectors = self.encoder.encode(
            [self.query_prefix + query.text for query in query_list],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        query_vectors = np.asarray(query_vectors, dtype=np.float32)
        run: dict[str, list[Hit]] = {}
        vectors = np.asarray(self.vectors)
        for start in range(0, len(query_list), self.search_batch_size):
            batch_queries = query_list[start : start + self.search_batch_size]
            qbatch = query_vectors[start : start + self.search_batch_size]
            score_matrix = qbatch @ vectors.T
            effective_top_k = min(top_k, score_matrix.shape[1])
            top_indices = np.argpartition(-score_matrix, kth=effective_top_k - 1, axis=1)[:, :effective_top_k]
            for row, query in enumerate(batch_queries):
                candidate_indices = top_indices[row]
                candidate_scores = score_matrix[row, candidate_indices]
                order = np.argsort(-candidate_scores)
                run[query.query_id] = [
                    Hit(
                        query_id=query.query_id,
                        doc_id=self.docs[int(candidate_indices[idx])].doc_id,
                        rank=rank,
                        score=float(candidate_scores[idx]),
                        retriever=self.name,
                    )
                    for rank, idx in enumerate(order, start=1)
                ]
        return run


def _hash_vector(text: str, dimensions: int) -> list[float]:
    vec = [0.0] * dimensions
    for token in text.lower().split():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(value * value for value in vec)) or 1.0
    return [value / norm for value in vec]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
