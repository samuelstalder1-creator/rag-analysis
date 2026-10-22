"""Named retriever definitions from an experiment config.

Which retrievers run, and how each one is configured::

    retrievers: [bm25, tfidf, bm25_tfidf_rrf]   # each gets its own metrics/answers
    retriever_definitions:                       # optional
      bm25:
        k1: 1.2
        b: 0.75
      bm25_tfidf_rrf:
        type: hybrid
        members: [bm25, tfidf]
        fusion: rrf                              # rrf | weighted
        rrf_k: 60
        weights: {bm25: 1.0, tfidf: 1.0}
        depth: 100                               # candidates taken from each member

``retrievers`` may also be a mapping ``name -> definition`` (everything listed
runs). A definition may be a path to a YAML file (``configs/retrievers/*.yaml``).
Names of built-in types (bm25, tfidf, e5, minilm, hash_dense) work without a
definition. Their defaults come from the shared ``retrieval`` block, as before.

Each base retriever is indexed once per run and searched at the largest depth
anything needs; hybrids reuse those member runs.
"""

from __future__ import annotations

from typing import Sequence

from evlab.config import load_config
from evlab.schema import Doc, Hit, Query

DENSE_TYPES = {"minilm", "e5", "dense"}
BASE_TYPES = {"bm25", "tfidf", "hash_dense"} | DENSE_TYPES
DENSE_DEFAULT_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "e5": "intfloat/e5-base-v2",
}
DENSE_KEYS = {
    "model", "batch_size", "search_batch_size", "local_files_only", "cache_dir",
    "device", "score_device", "max_seq_length", "query_prefix", "doc_prefix",
}
TYPE_KEYS = {
    "bm25": {"k1", "b", "remove_query_stopwords"},
    "tfidf": set(),
    "hash_dense": {"dimensions"},
    "minilm": DENSE_KEYS,
    "e5": DENSE_KEYS,
    "dense": DENSE_KEYS,
    "hybrid": {"members", "fusion", "weights", "rrf_k", "normalization", "depth"},
}


def resolve_retrievers(config: dict) -> tuple[list[str], dict[str, dict]]:
    """Return the retrievers to run and fully resolved specs for all of them
    (including hybrid members that are not run on their own)."""
    raw = config.get("retrievers", ["bm25"])
    definitions = dict(config.get("retriever_definitions") or {})
    if isinstance(raw, dict):
        run_names = [str(name) for name in raw]
        definitions.update(raw)
    elif isinstance(raw, (list, tuple)):
        run_names = [str(name) for name in raw]
    elif isinstance(raw, str):
        run_names = [raw]
    else:
        raise ValueError("retrievers must be a list of names or a mapping of definitions")
    if not run_names:
        raise ValueError("retrievers is empty")
    if len(set(run_names)) != len(run_names):
        raise ValueError(f"Duplicate retriever names: {run_names}")
    global_retrieval = config.get("retrieval", {}) or {}

    specs: dict[str, dict] = {}

    def resolve(name: str, stack: tuple[str, ...]) -> None:
        if name in specs:
            return
        if name in stack:
            raise ValueError(f"Hybrid retriever cycle: {' -> '.join(stack + (name,))}")
        definition = definitions.get(name) or {}
        if isinstance(definition, str):
            definition = _load_definition_file(definition)
        definition = dict(definition)
        kind = str(definition.pop("type", name))
        if kind not in TYPE_KEYS:
            raise ValueError(
                f"Unknown retriever '{name}'. Use a built-in name ({sorted(set(TYPE_KEYS) - {'hybrid'})}) "
                "or define it in retriever_definitions with a `type`"
            )
        unknown = set(definition) - TYPE_KEYS[kind]
        if unknown:
            raise ValueError(
                f"Unknown keys for retriever '{name}' (type {kind}): {sorted(unknown)}; "
                f"allowed: {sorted(TYPE_KEYS[kind])}"
            )
        spec = {"type": kind, **_global_defaults(kind, global_retrieval), **definition}
        if kind == "hybrid":
            members = spec.get("members")
            if not isinstance(members, list) or len(members) < 2:
                raise ValueError(f"Hybrid retriever '{name}' needs at least two members")
            spec["members"] = [str(member) for member in members]
            spec.setdefault("fusion", "rrf")
            spec.setdefault("rrf_k", 60)
            spec.setdefault("normalization", "minmax")
            from evlab.retrieval.fusion import FUSION_METHODS, NORMALIZATIONS

            if spec["fusion"] not in FUSION_METHODS:
                raise ValueError(f"Hybrid '{name}': fusion must be one of {sorted(FUSION_METHODS)}")
            if spec["normalization"] not in NORMALIZATIONS:
                raise ValueError(f"Hybrid '{name}': normalization must be one of {sorted(NORMALIZATIONS)}")
            if spec.get("depth") is not None and int(spec["depth"]) < 1:
                raise ValueError(f"Hybrid '{name}': depth must be >= 1")
            weights = spec.get("weights") or {}
            extra = set(weights) - set(spec["members"])
            if extra:
                raise ValueError(f"Hybrid '{name}' has weights for non-members: {sorted(extra)}")
            for member in spec["members"]:
                resolve(member, stack + (name,))
        specs[name] = spec

    for name in run_names:
        resolve(name, ())
    return run_names, specs


class RetrievalSession:
    """Runs the configured retrievers over one corpus, reusing member runs."""

    def __init__(self, specs: dict[str, dict], docs: Sequence[Doc], queries: Sequence[Query]) -> None:
        self.specs = specs
        self.docs = docs
        self.queries = queries
        self._runs: dict[str, dict[str, list[Hit]]] = {}
        self._depth: dict[str, int] = {}

    def plan(self, run_names: list[str], top_k: int) -> None:
        """Work out how deep each retriever must search."""
        def need(name: str, depth: int) -> None:
            if depth <= self._depth.get(name, 0):
                return
            self._depth[name] = depth
            spec = self.specs[name]
            if spec["type"] == "hybrid":
                for member in spec["members"]:
                    need(member, int(spec.get("depth") or depth))

        for name in run_names:
            need(name, top_k)

    def run(self, name: str, top_k: int) -> dict[str, list[Hit]]:
        if name not in self._runs:
            depth = self._depth.get(name, top_k)
            spec = self.specs[name]
            if spec["type"] == "hybrid":
                member_depth = int(spec.get("depth") or depth)
                members = {m: self.run(m, member_depth) for m in spec["members"]}
                from evlab.retrieval.fusion import fuse_runs

                self._runs[name] = fuse_runs(
                    members,
                    name=name,
                    top_k=depth,
                    method=str(spec["fusion"]),
                    weights=spec.get("weights"),
                    rrf_k=float(spec["rrf_k"]),
                    normalization=str(spec["normalization"]),
                )
            else:
                retriever = build_base_retriever(spec)
                retriever.index(self.docs)
                self._runs[name] = retriever.search(self.queries, top_k=depth)
                del retriever
        return _truncate(self._runs[name], top_k)


def build_base_retriever(spec: dict):
    kind = spec["type"]
    if kind == "bm25":
        from evlab.retrieval.bm25 import BM25Retriever

        return BM25Retriever(
            k1=float(spec.get("k1", 1.5)),
            b=float(spec.get("b", 0.75)),
            remove_query_stopwords=bool(spec.get("remove_query_stopwords", True)),
        )
    if kind == "tfidf":
        from evlab.retrieval.tfidf import TfidfRetriever

        return TfidfRetriever()
    if kind == "hash_dense":
        from evlab.retrieval.dense import HashDenseRetriever

        return HashDenseRetriever(dimensions=int(spec.get("dimensions", 256)))
    if kind in DENSE_TYPES:
        from evlab.retrieval.dense import SentenceTransformerRetriever

        model = spec.get("model") or DENSE_DEFAULT_MODELS.get(kind)
        if not model:
            raise ValueError("Dense retriever needs a `model`")
        e5 = kind == "e5"
        return SentenceTransformerRetriever(
            str(model),
            batch_size=int(spec.get("batch_size", 128)),
            search_batch_size=int(spec.get("search_batch_size", 16)),
            local_files_only=bool(spec.get("local_files_only", True)),
            cache_dir=str(spec.get("cache_dir", "cache/embeddings")),
            device=spec.get("device"),
            score_device=spec.get("score_device"),
            query_prefix=str(spec.get("query_prefix", "query: " if e5 else "")),
            doc_prefix=str(spec.get("doc_prefix", "passage: " if e5 else "")),
            max_seq_length=int(spec["max_seq_length"]) if spec.get("max_seq_length") is not None else None,
        )
    raise ValueError(f"Not a base retriever type: {kind}")


def _global_defaults(kind: str, retrieval: dict) -> dict:
    """Map the shared `retrieval` block (old style) to spec keys for this type."""
    if kind == "bm25":
        return {key: retrieval[key] for key in ("k1", "b", "remove_query_stopwords") if key in retrieval}
    if kind in DENSE_TYPES:
        return {
            key[len("dense_"):]: value
            for key, value in retrieval.items()
            if key.startswith("dense_") and key[len("dense_"):] in DENSE_KEYS
        }
    return {}


def _load_definition_file(path: str) -> dict:
    loaded = dict(load_config(path))
    if "retriever" in loaded:
        loaded["type"] = loaded.pop("retriever")
    return loaded


def _truncate(run: dict[str, list[Hit]], k: int) -> dict[str, list[Hit]]:
    return {qid: hits[:k] for qid, hits in run.items()}
