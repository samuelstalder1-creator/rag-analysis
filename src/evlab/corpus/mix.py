"""Ratio-based corpus composition for retrieval and RAG runs.

A mix config names generated *sources* and then says how much of each enters
the run corpus. Two independent axes:

``background``  corpus-wide contamination, sampled per human document (root).
                "50 % of the corpus also exists as a Qwen paraphrase."
``targeted``    evidence aimed at the evaluated questions, sampled per question.
                "30 % of the attackable questions receive k false copies of
                their answer documents, optionally plus k correct copies."

Each entry has a ``mode``: ``add`` keeps the human document next to the copy,
``replace`` removes the human document whose copy entered the corpus.

Sampling is deterministic and *nested*: with the same seed, the documents (or
questions) selected at ratio 0.25 are a subset of those selected at 0.5. A
ratio sweep therefore adds contamination on top of the previous step instead
of drawing a fresh random sample.

Example::

    corpus:
      seed: 13
      sources:
        qwen_correct: {path: data/generated/qwen35-2b-all}
        false_det: {path: data/generated/false-deterministic}
        cocktail_llama:
          path: data/cocktail/nq/corpus/llama-2-7b-chat-tmp0.2.jsonl
          origin: cocktail_llm
      background:
        - {source: qwen_correct, ratio: 0.5, mode: add}
      targeted:
        query_ratio: 1.0
        false_docs: {source: false_det, k: 1, mode: add}
        correct_docs: {source: qwen_correct, k: 1, mode: add}

``background`` may also be a mapping keyed by source name
(``background: {qwen_correct: {ratio: 0.5}}``), which gives sweep paths like
``corpus.background.qwen_correct.ratio``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path

from evlab.data.cocktail import load_corpus
from evlab.data.nqplus import (
    load_generation_queries,
    load_rag_queries,
    load_retrieval_queries,
)
from evlab.io import stable_hash
from evlab.schema import Doc, Query

LOGGER = logging.getLogger(__name__)

MODES = {"add", "replace"}
SCOPES = {"corpus", "relevant"}
ORIGINS = {"generated", "cocktail_llm"}
LEGACY_KEYS = {"base", "generated_docs", "k_correct", "k_false"}
CORPUS_KEYS = {"seed", "sources", "background", "targeted"}
SOURCE_KEYS = {"path", "origin", "namespace", "generation_depth"}
BACKGROUND_KEYS = {"source", "ratio", "mode", "scope", "k", "condition", "salt"}
TARGETED_KEYS = {"query_ratio", "queries", "salt", "false_docs", "correct_docs"}
# Config keys per targeted kind. (A bare `false:` key is read as a boolean by YAML.)
TARGET_KINDS = (("false", "false_docs"), ("correct", "correct_docs"))
TARGET_PART_KEYS = {"source", "k", "mode"}

# Loading the 104k-document corpora dominates corpus construction. Sweeps build
# many corpora in one process, so loaded sources are kept for reuse.
_SOURCE_CACHE: dict[tuple, list[Doc]] = {}


def is_mix_config(corpus_config: dict | None) -> bool:
    return bool(corpus_config) and "sources" in corpus_config


def nested_sample(units: list[str], ratio: float, seed: object, salt: str) -> list[str]:
    """Deterministic subset of ``round(ratio * len(units))`` units.

    Units are ranked by a seeded hash, so a larger ratio always contains the
    selection of a smaller one. The result keeps rank order, which also makes the
    generated part of the corpus grow as a prefix across a sweep (dense
    embedding caches reuse that prefix in ``add`` mode).
    """
    ranked = sorted(set(units), key=lambda unit: stable_hash([seed, salt, unit]))
    count = math.floor(ratio * len(ranked) + 0.5)
    return ranked[:count]


def build_mixed_corpus(config: dict) -> tuple[list[Doc], dict]:
    from evlab.corpus.builder import query_ids_by_doc, verify_generated_docs

    data = config["data"]
    corpus = config["corpus"]
    validate_mix_config(corpus)
    seed = corpus.get("seed", config.get("seed", 13))
    query_set = str(config.get("queries", "retrieval"))

    human = _cached(("human", str(data["human_corpus"])), lambda: load_corpus(
        data["human_corpus"], origin="human", namespace="human"
    ))
    human_roots = [doc.root_id for doc in human]
    qmap_key = str(data["nqplus_dir"])
    sources = _SourceSet(corpus["sources"], data, qmap_key, query_ids_by_doc)

    selected: dict[str, Doc] = {}
    removed_roots: set[str] = set()
    report: dict = {"mode": "mix", "seed": seed}

    background_reports = []
    eval_queries: list[Query] | None = None
    for label, entry in background_entries(corpus):
        name = entry["source"]
        ratio = _ratio(entry.get("ratio", 1.0), f"{label}.ratio")
        mode = entry.get("mode", "add")
        k = int(entry.get("k", 1))
        scope = entry.get("scope", "corpus")
        by_root = sources.by_root(name, condition=_condition(entry.get("condition")))
        if scope == "corpus":
            in_scope = set(human_roots)
        else:
            if eval_queries is None:
                eval_queries = _load_queries(data["nqplus_dir"], query_set)
            in_scope = {did for query in eval_queries for did in query.relevant_doc_ids}
        pool = [root for root in human_roots if root in in_scope and root in by_root]
        chosen = nested_sample(pool, ratio, seed, f"background:{entry.get('salt', name)}")
        added, shortfall = _take(chosen, by_root, k, selected)
        if mode == "replace":
            removed_roots.update(chosen)
        background_reports.append(
            {
                "source": name,
                "ratio": ratio,
                "mode": mode,
                "scope": scope,
                "k": k,
                "pool_roots": len(pool),
                "selected_roots": len(chosen),
                "docs_added": added,
                "k_shortfall": shortfall,
            }
        )
    report["background"] = background_reports

    targeted = corpus.get("targeted")
    if targeted:
        report["targeted"] = _apply_targeted(
            targeted, data, query_set, seed, sources, selected, removed_roots
        )

    generated = list(selected.values())
    verify_generated_docs(
        [doc for doc in generated if doc.origin == "generated"], human, data
    )
    kept_human = [doc for doc in human if doc.root_id not in removed_roots]
    docs = kept_human + generated
    report.update(
        num_docs=len(docs),
        human_docs=len(kept_human),
        human_removed=len(human) - len(kept_human),
        generated_docs=len(generated),
        generated_share=(len(generated) / len(docs)) if docs else 0.0,
        false_docs=sum(1 for doc in generated if doc.condition == "false"),
        correct_docs=sum(1 for doc in generated if doc.condition == "correct"),
        cocktail_llm_docs=sum(1 for doc in generated if doc.origin == "cocktail_llm"),
    )
    return docs, report


def _apply_targeted(targeted, data, default_query_set, seed, sources, selected, removed_roots):
    query_set = str(targeted.get("queries") or default_query_set)
    ratio = _ratio(targeted.get("query_ratio", 1.0), "targeted.query_ratio")
    queries = _load_queries(data["nqplus_dir"], query_set)
    parts = {kind: targeted[key] for kind, key in TARGET_KINDS if targeted.get(key)}
    if not parts:
        raise ValueError("corpus.targeted needs a 'false_docs' and/or 'correct_docs' entry")

    indexes = {}
    if "false" in parts:
        false_index: dict[tuple[str, str], list[Doc]] = {}
        for doc in sources.docs(parts["false"]["source"]):
            if doc.condition == "false" and doc.query_id:
                false_index.setdefault((doc.query_id, doc.root_id), []).append(doc)
        indexes["false"] = false_index
    if "correct" in parts:
        indexes["correct"] = sources.by_root(parts["correct"]["source"], exclude_false=True)

    def answer_roots(query: Query) -> list[str]:
        roots = query.metadata.get("docs_with_answer") or query.relevant_doc_ids
        return list(dict.fromkeys(str(root) for root in roots))

    def docs_for(kind: str, query: Query) -> dict[str, list[Doc]]:
        if kind == "false":
            return {r: _ordered(indexes["false"].get((query.query_id, r), [])) for r in answer_roots(query)}
        return {r: indexes["correct"].get(r, []) for r in answer_roots(query)}

    # A question is attackable when the primary kind has at least one document
    # for it: false evidence if configured, otherwise correct copies.
    primary = "false" if "false" in parts else "correct"
    attackable = [q.query_id for q in queries if any(docs_for(primary, q).values())]
    chosen = set(nested_sample(attackable, ratio, seed, f"targeted:{targeted.get('salt', 'queries')}"))

    part_reports = {}
    for kind, spec in parts.items():
        part_reports[kind] = {
            "source": spec["source"],
            "k": int(spec.get("k", 1)),
            "mode": spec.get("mode", "add"),
            "docs_added": 0,
            "roots_with_docs": 0,
            "k_shortfall": 0,
        }
    for query in queries:
        if query.query_id not in chosen:
            continue
        for kind, spec in parts.items():
            k = int(spec.get("k", 1))
            info = part_reports[kind]
            for root, docs in docs_for(kind, query).items():
                picked = docs[:k]
                info["k_shortfall"] += max(k - len(picked), 0)
                if not picked:
                    continue
                info["roots_with_docs"] += 1
                for doc in picked:
                    info["docs_added"] += int(_add(selected, doc))
                if spec.get("mode", "add") == "replace":
                    removed_roots.add(root)

    return {
        "queries": query_set,
        "query_ratio": ratio,
        "evaluated_queries": len(queries),
        "attackable_queries": len(attackable),
        "targeted_queries": len(chosen),
        "query_ids": sorted(chosen),
        **{f"{kind}_docs": info for kind, info in part_reports.items()},
    }


class _SourceSet:
    def __init__(self, specs: dict, data: dict, qmap_key: str, qmap_builder) -> None:
        self.specs = specs
        self.data = data
        self.qmap_key = qmap_key
        self.qmap_builder = qmap_builder
        self._qmap: dict[str, str] | None = None

    def docs(self, name: str) -> list[Doc]:
        spec = self.specs[name]
        origin = spec.get("origin", "generated")
        path = str(spec["path"])
        if origin == "cocktail_llm":
            namespace = spec.get("namespace", "cocktail_llm")
            return _cached(
                ("cocktail", path, namespace),
                lambda: load_corpus(path, origin="cocktail_llm", namespace=namespace),
            )
        from evlab.corpus.builder import load_generated_docs

        depth = int(spec.get("generation_depth", 1))
        # Generated files without an explicit id_namespace are all called
        # gen::corpus_0001::<id>, so two output directories would collide.
        # Prefixing the source name keeps every source's IDs distinct.
        docs = _cached(
            ("generated", name, str(Path(path).resolve()), self.qmap_key),
            lambda: [
                replace(doc, doc_id=f"gen::{name}::{doc.doc_id.split('::', 1)[-1]}")
                for doc in load_generated_docs(path, query_ids_by_doc=self.qmap())
            ],
        )
        return [doc for doc in docs if doc.generation_depth == depth]

    def qmap(self) -> dict[str, str]:
        if self._qmap is None:
            self._qmap = self.qmap_builder(self.data)
        return self._qmap

    def by_root(self, name: str, *, condition: str | None = None, exclude_false: bool = False) -> dict[str, list[Doc]]:
        grouped: dict[str, list[Doc]] = {}
        for doc in self.docs(name):
            if condition and doc.condition != condition:
                continue
            if exclude_false and doc.condition == "false":
                continue
            grouped.setdefault(doc.root_id, []).append(doc)
        return {root: _ordered(docs) for root, docs in grouped.items()}


def validate_mix_config(corpus: dict) -> None:
    legacy = LEGACY_KEYS & set(corpus)
    if legacy:
        raise ValueError(
            f"corpus.sources cannot be combined with legacy keys {sorted(legacy)}; "
            "express them as background/targeted entries"
        )
    _no_unknown(corpus, CORPUS_KEYS, "corpus")
    sources = corpus.get("sources") or {}
    if not isinstance(sources, dict) or not sources:
        raise ValueError("corpus.sources must map source names to {path: ...}")
    for name, spec in sources.items():
        if not isinstance(spec, dict) or "path" not in spec:
            raise ValueError(f"corpus.sources.{name} needs a path")
        _no_unknown(spec, SOURCE_KEYS, f"corpus.sources.{name}")
        if spec.get("origin", "generated") not in ORIGINS:
            raise ValueError(f"corpus.sources.{name}.origin must be one of {sorted(ORIGINS)}")

    def check_source(where: str, name: object) -> None:
        if name not in sources:
            raise ValueError(f"{where}.source '{name}' is not defined in corpus.sources")

    for where, entry in background_entries(corpus):
        _no_unknown(entry, BACKGROUND_KEYS, where)
        check_source(where, entry.get("source"))
        _ratio(entry.get("ratio", 1.0), f"{where}.ratio")
        _mode(entry.get("mode", "add"), where)
        _k(entry.get("k", 1), where)
        if entry.get("scope", "corpus") not in SCOPES:
            raise ValueError(f"{where}.scope must be one of {sorted(SCOPES)}")
        if _condition(entry.get("condition")) not in {None, "correct", "false"}:
            raise ValueError(f"{where}.condition must be correct, false or omitted")

    targeted = corpus.get("targeted")
    if targeted:
        if any(isinstance(key, bool) for key in targeted):
            raise ValueError(
                "corpus.targeted has a boolean key: YAML reads `false:` as False. "
                "Use false_docs / correct_docs"
            )
        _no_unknown(targeted, TARGETED_KEYS, "corpus.targeted")
        _ratio(targeted.get("query_ratio", 1.0), "corpus.targeted.query_ratio")
        if targeted.get("queries") not in {None, "retrieval", "rag", "generation"}:
            raise ValueError("corpus.targeted.queries must be retrieval, rag or generation")
        for _, key in TARGET_KINDS:
            part = targeted.get(key)
            if not part:
                continue
            where = f"corpus.targeted.{key}"
            _no_unknown(part, TARGET_PART_KEYS, where)
            check_source(where, part.get("source"))
            _mode(part.get("mode", "add"), where)
            _k(part.get("k", 1), where)


def background_entries(corpus: dict) -> list[tuple[str, dict]]:
    """Background entries as ``(label, entry)``.

    Accepts a list of entries or a mapping keyed by source name (the key is the
    default ``source``)::

        background:
          qwen_correct: {ratio: 0.5}        # == [{source: qwen_correct, ratio: 0.5}]
    """
    background = corpus.get("background") or []
    if isinstance(background, dict):
        entries = []
        for key, entry in background.items():
            entry = dict(entry or {})
            entry.setdefault("source", key)
            entries.append((f"corpus.background.{key}", entry))
        return entries
    if isinstance(background, list):
        return [(f"corpus.background[{i}]", entry) for i, entry in enumerate(background)]
    raise ValueError("corpus.background must be a list or a mapping keyed by source name")


def clear_source_cache() -> None:
    _SOURCE_CACHE.clear()


def _cached(key: tuple, load):
    if key not in _SOURCE_CACHE:
        _SOURCE_CACHE[key] = load()
    return _SOURCE_CACHE[key]


def _add(selected: dict[str, Doc], doc: Doc) -> bool:
    """Add a document; the same document twice is a no-op, a different one an error."""
    existing = selected.get(doc.doc_id)
    if existing is None:
        selected[doc.doc_id] = doc
        return True
    if existing is not doc and (existing.text, existing.title) != (doc.text, doc.title):
        raise ValueError(
            f"Two sources produce the document ID {doc.doc_id}; give one of them a "
            "distinct `namespace` in corpus.sources"
        )
    return False


def _take(roots: list[str], by_root: dict[str, list[Doc]], k: int, selected: dict[str, Doc]) -> tuple[int, int]:
    added = shortfall = 0
    for root in roots:
        picked = by_root.get(root, [])[:k]
        shortfall += max(k - len(picked), 0)
        for doc in picked:
            added += int(_add(selected, doc))
    return added, shortfall


def _ordered(docs: list[Doc]) -> list[Doc]:
    return sorted(docs, key=lambda doc: (doc.sample_idx or 0, doc.doc_id))


def _load_queries(nqplus_dir, query_set: str) -> list[Query]:
    loaders = {
        "retrieval": load_retrieval_queries,
        "rag": load_rag_queries,
        "generation": load_generation_queries,
    }
    if query_set not in loaders:
        raise ValueError(f"Unsupported query set: {query_set}")
    return loaders[query_set](nqplus_dir)


def _condition(value: object) -> str | None:
    # YAML 1.1 reads an unquoted `false` as a boolean.
    return "false" if value is False else value  # type: ignore[return-value]


def _ratio(value: object, where: str) -> float:
    ratio = float(value)  # type: ignore[arg-type]
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"{where} must be between 0 and 1, got {value}")
    return ratio


def _mode(value: object, where: str) -> None:
    if value not in MODES:
        raise ValueError(f"{where}.mode must be one of {sorted(MODES)}, got {value}")


def _k(value: object, where: str) -> None:
    if int(value) < 0:  # type: ignore[arg-type]
        raise ValueError(f"{where}.k must be >= 0")


def _no_unknown(mapping: dict, allowed: set[str], where: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in {where}: {sorted(unknown)} (allowed: {sorted(allowed)})")
