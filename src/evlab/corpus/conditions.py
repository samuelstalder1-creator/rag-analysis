from __future__ import annotations

from evlab.io import stable_hash
from evlab.schema import Doc


def corpus_manifest(docs: list[Doc], config: dict) -> dict[str, object]:
    return {
        "num_docs": len(docs),
        "num_human": sum(1 for doc in docs if doc.origin == "human"),
        "num_cocktail_llm": sum(1 for doc in docs if doc.origin == "cocktail_llm"),
        "num_generated": sum(1 for doc in docs if doc.origin == "generated"),
        "doc_set_hash": stable_hash(sorted(doc.doc_id for doc in docs)),
        "config_hash": stable_hash(config),
    }
