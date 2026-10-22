"""Fuse the ranked lists of several retrievers into one ranking."""

from __future__ import annotations

from evlab.schema import Hit

FUSION_METHODS = {"rrf", "weighted"}
NORMALIZATIONS = {"minmax", "zscore", "none"}


def fuse_runs(
    runs: dict[str, dict[str, list[Hit]]],
    *,
    name: str,
    top_k: int,
    method: str = "rrf",
    weights: dict[str, float] | None = None,
    rrf_k: float = 60.0,
    normalization: str = "minmax",
) -> dict[str, list[Hit]]:
    """Combine member runs (``{member: {query_id: hits}}``) per query.

    ``rrf``       Reciprocal Rank Fusion: sum of ``w / (rrf_k + rank)``.
                  Uses only ranks, so score scales do not matter.
    ``weighted``  Weighted sum of per-query normalised scores (``minmax``,
                  ``zscore`` or ``none``). A document missing from a member's
                  list contributes 0 for that member.

    Ties are broken by document ID so fused rankings are deterministic.
    """
    if method not in FUSION_METHODS:
        raise ValueError(f"Unknown fusion method {method!r}; use one of {sorted(FUSION_METHODS)}")
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"Unknown normalization {normalization!r}; use one of {sorted(NORMALIZATIONS)}")
    weights = {member: float((weights or {}).get(member, 1.0)) for member in runs}
    query_ids = sorted({qid for run in runs.values() for qid in run})
    fused: dict[str, list[Hit]] = {}
    for qid in query_ids:
        scores: dict[str, float] = {}
        for member, run in runs.items():
            hits = run.get(qid, [])
            weight = weights[member]
            if method == "rrf":
                for hit in hits:
                    scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + weight / (rrf_k + hit.rank)
            else:
                for doc_id, value in _normalized(hits, normalization).items():
                    scores[doc_id] = scores.get(doc_id, 0.0) + weight * value
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        fused[qid] = [
            Hit(query_id=qid, doc_id=doc_id, rank=rank, score=score, retriever=name)
            for rank, (doc_id, score) in enumerate(ranked, start=1)
        ]
    return fused


def _normalized(hits: list[Hit], normalization: str) -> dict[str, float]:
    if not hits:
        return {}
    values = [hit.score for hit in hits]
    if normalization == "none":
        return {hit.doc_id: hit.score for hit in hits}
    if normalization == "minmax":
        low, high = min(values), max(values)
        span = high - low
        return {hit.doc_id: (hit.score - low) / span if span else 1.0 for hit in hits}
    mean = sum(values) / len(values)
    std = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    return {hit.doc_id: (hit.score - mean) / std if std else 0.0 for hit in hits}
