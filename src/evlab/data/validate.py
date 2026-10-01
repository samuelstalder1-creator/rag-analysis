from __future__ import annotations

from pathlib import Path

from evlab.data.cocktail import load_corpus
from evlab.data.nqplus import validate_nqplus


def validate_dataset_config(config: dict) -> dict[str, int]:
    data = config.get("data", config)
    nqplus_dir = Path(data["nqplus_dir"])
    stats = validate_nqplus(nqplus_dir)
    if "human_corpus" in data:
        stats["human_docs"] = len(load_corpus(data["human_corpus"], origin="human", namespace="human"))
    if "cocktail_llm_corpus" in data:
        stats["cocktail_llm_docs"] = len(
            load_corpus(data["cocktail_llm_corpus"], origin="cocktail_llm", namespace="cocktail_llm")
        )
    return stats
