"""Replay exact cached requests through the current pipelines, without network.

Run from the repository root. Existing rows are skipped by their fingerprints;
successful cached responses get ordinary, freshly computed validation receipts.
Cache misses return empty text to the validators and are never cached themselves.
"""

import json
from pathlib import Path

from evlab.config import load_config
from evlab.generation.correct_repair import run_correct_repair
from evlab.generation.false_documents import run_false_document_generation
from evlab.llm.cache import CachedLLMClient
from evlab.llm.local import LocalOpenAIClient


class CacheMiss(Exception):
    pass


def deny_network(*args, **kwargs):
    raise CacheMiss("Network disabled during cache recovery")


original_generate = CachedLLMClient.generate
hits = misses = 0


def cached_only(self, *args, **kwargs):
    global hits, misses
    try:
        response = original_generate(self, *args, **kwargs)
    except CacheMiss:
        misses += 1
        return ""
    hits += 1
    return response


def main():
    LocalOpenAIClient.generate = deny_network
    CachedLLMClient.generate = cached_only
    results = []
    for seed in range(4):
        for kind in ("correct", "false"):
            config = load_config(
                f"configs/generation/nqopen1520-together-gpt-oss-120b-{kind}.yaml"
            )
            if kind == "correct":
                result = run_correct_repair(config, retry_seed=seed)
            else:
                result = run_false_document_generation(
                    config["generation"], config["data"]["nqplus_dir"],
                    human_corpus=config["data"]["human_corpus"], retry_seed=seed,
                )
            results.append({"kind": kind, "retry_seed": seed, "result": result})
            print(json.dumps(results[-1]), flush=True)
    report = {"network_requests": 0, "cache_hits": hits, "cache_misses": misses,
              "runs": results}
    Path(__file__).with_name("cache_recovery.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
