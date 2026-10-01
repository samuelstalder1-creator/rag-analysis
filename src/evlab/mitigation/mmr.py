from __future__ import annotations

from evlab.schema import Hit


def pass_through_mmr_placeholder(hits: list[Hit], *, top_k: int) -> list[Hit]:
    """Placeholder for MMR; keeps the contract while embeddings are wired in."""
    return hits[:top_k]
