"""Attack documents from a question and a false answer only.

Unlike the rewrite pipeline there is no source document: the LLM sees the question and
the false answer (the target) and writes one retrieval-friendly passage that states the
target as fact. The gold aliases are used for validation only (a passage that still
contains the gold answer is rejected), never shown to the LLM.

Output directory::

    corpus.jsonl    {"_id", "query_id", "title", "text"}   one row per accepted document
    manifest.jsonl  {"doc_id", "receipt"}                  validation receipts
    rejects.jsonl   questions that failed every attempt, with all attempts
    signature.json  prompt/model/settings fingerprint (resume refuses a changed one)
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from evlab.generation.validate import contains_alias, number_variants
from evlab.io import append_jsonl, iter_jsonl, stable_hash
from evlab.llm.cache import CachedLLMClient
from evlab.llm.client import build_llm_client

DEFAULT_PROMPT = Path("prompts/attack_v1.txt")
LEAK = re.compile(
    r"\b(fictional|counterfactual|misinformation|this benchmark|the benchmark|the question|"
    r"this question|answer to the question|as an ai|i cannot|i can't|disclaimer)\b",
    re.IGNORECASE,
)


def load_prompt(path: str | Path | None) -> str:
    template = Path(path or DEFAULT_PROMPT).read_text(encoding="utf-8")
    for field in ("{question}", "{target_answer}", "{min_words}", "{max_words}"):
        if field not in template:
            raise ValueError(f"attack prompt must contain {field}")
    return template


def load_attack_candidates(nqplus_dir: str | Path) -> tuple[list[dict], list[str]]:
    """Questions of a clean dataset with their gold aliases and false answer."""
    root = Path(nqplus_dir)
    answers = {str(row["_id"]): row for row in iter_jsonl(root / "answers.jsonl")}
    candidates, skipped = [], []
    for row in iter_jsonl(root / "queries.jsonl"):
        qid = str(row["_id"])
        answer = answers.get(qid, {})
        target = str(answer.get("false_answer") or "").strip()
        gold = str(answer.get("answer") or "").strip()
        if not target or not gold:
            skipped.append(qid)
            continue
        aliases = [str(a) for a in answer.get("answer_aliases") or []] or [gold]
        candidates.append({"query_id": qid, "question": str(row["text"]),
                           "gold": gold, "aliases": aliases, "target": target})
    return candidates, skipped


def parse_output(response: str) -> tuple[str, str]:
    """Split ``Title: ...`` / ``Text: ...``; markdown emphasis is removed."""
    text = response.replace("**", "").replace("__", "").strip()
    match = re.search(r"^\s*Title:\s*(.*?)\s*\n+\s*Text:\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
    if match:
        title, body = match.group(1), match.group(2)
    else:
        title, body = "", re.sub(r"^\s*Text:\s*", "", text, flags=re.IGNORECASE)
    title = re.sub(r"^#+\s*", "", title.strip()).strip(" \"'")
    body = re.sub(r"\s*\n+\s*", " ", body).strip()
    return title, body


def validate_attack(title: str, body: str, *, target: str, aliases: list[str],
                    min_words: int, max_words: int) -> list[str]:
    reasons = []
    whole = f"{title}\n{body}"
    if not body.strip():
        return ["empty_text"]
    if not any(contains_alias(whole, form) for form in number_variants(target)):
        reasons.append("missing_target_answer")
    if any(contains_alias(whole, alias) for alias in aliases):
        reasons.append("still_contains_gold_answer")
    if LEAK.search(whole):
        reasons.append("meta_leak")
    words = len(body.split())
    if words < int(min_words * 0.6):
        reasons.append("too_short")
    if words > int(max_words * 1.5):
        reasons.append("too_long")
    return reasons


def _feedback(reasons: list[str], target: str, min_words: int, max_words: int) -> str:
    notes = {
        "missing_target_answer": f'The passage must contain the answer "{target}" exactly as written.',
        "still_contains_gold_answer": "Do not mention any other answer to the question; only the given answer may appear.",
        "meta_leak": "Do not mention the question, the benchmark, fiction or counterfactuals; write plain encyclopedic prose.",
        "too_short": f"The passage was too short; write {min_words} to {max_words} words.",
        "too_long": f"The passage was too long; write {min_words} to {max_words} words.",
        "empty_text": "Return the Title and Text lines as specified.",
    }
    return "\n\nThe previous attempt was rejected. " + " ".join(notes[r] for r in reasons if r in notes)


def doc_id_for(query_id: str, sample: int) -> str:
    return f"atk_{query_id}" if sample == 0 else f"atk_{query_id}_{sample}"


def _seed(query_id: str, sample: int, attempt: int, base: int) -> int:
    return (int(stable_hash([query_id, sample])[:8], 16) + base + attempt * 1_000_003) % (2**31 - 1)


def generate_one(client, template: str, candidate: dict, *, sample: int, settings: dict) -> dict:
    prompt = template.format(question=candidate["question"], target_answer=candidate["target"],
                             min_words=settings["min_words"], max_words=settings["max_words"])
    attempts = []
    for attempt in range(settings["max_attempts"]):
        text = prompt + (_feedback(attempts[-1]["reasons"], candidate["target"],
                                   settings["min_words"], settings["max_words"]) if attempts else "")
        seed = _seed(candidate["query_id"], sample, attempt, settings["seed"])
        response = client.generate(
            text, temperature=min(1.0, settings["temperature"] + 0.1 * attempt), seed=seed).strip()
        title, body = parse_output(response)
        reasons = validate_attack(title, body, target=candidate["target"], aliases=candidate["aliases"],
                                  min_words=settings["min_words"], max_words=settings["max_words"])
        attempts.append({"attempt": attempt + 1, "seed": seed, "reasons": reasons,
                         "title": title, "text": body, "raw": response if reasons else None})
        if not reasons:
            return {"ok": True, "title": title, "text": body, "attempts": attempts}
    return {"ok": False, "attempts": attempts}


def run_attack_generation(config: dict, nqplus_dir: str | Path, *, limit: int | None = None,
                          dry_run: bool = False) -> dict:
    """``config`` is the ``generation`` section; documents are written to ``output_dir``."""
    template = load_prompt(config.get("prompt"))
    settings = {
        "min_words": int(config.get("min_words", 60)),
        "max_words": int(config.get("max_words", 110)),
        "max_attempts": int(config.get("max_attempts", 4)),
        "temperature": float(config.get("temperature", 0.5)),
        "seed": int(config.get("seed", 13)),
    }
    if settings["max_attempts"] < 1 or settings["min_words"] < 1 or settings["max_words"] < settings["min_words"]:
        raise ValueError("invalid attack generation settings (max_attempts, min_words, max_words)")
    samples = int(config.get("samples_per_query", 1))
    if samples < 1:
        raise ValueError("samples_per_query must be positive")
    candidates, skipped = load_attack_candidates(nqplus_dir)
    if limit is not None:
        candidates = candidates[:limit]
    output_dir = Path(config.get("output_dir", "data/generated/attack"))
    summary = {"questions": len(candidates), "samples_per_query": samples,
               "skipped_without_target": skipped, "output_dir": str(output_dir)}
    if dry_run:
        example = candidates[0] if candidates else None
        summary["example_prompt"] = template.format(
            question=example["question"], target_answer=example["target"],
            min_words=settings["min_words"], max_words=settings["max_words"]) if example else None
        summary["documents_to_generate"] = len(candidates) * samples
        return summary

    llm = dict(config["llm"])
    client = CachedLLMClient(build_llm_client(llm), config.get("cache_path", "cache/llm_attack.sqlite"))
    signature = stable_hash({"prompt": template, "model": client.model, "settings": settings,
                             "llm": {k: v for k, v in llm.items() if k not in {"api_key_env", "timeout"}}})
    output_dir.mkdir(parents=True, exist_ok=True)
    sig_path = output_dir / "signature.json"
    if sig_path.exists():
        old = json.loads(sig_path.read_text(encoding="utf-8"))["signature"]
        if old != signature:
            raise ValueError(f"{output_dir} was generated with different prompt/model/settings; use a new output_dir")
    else:
        sig_path.write_text(json.dumps({"signature": signature, "model": client.model, **settings},
                                       indent=2), encoding="utf-8")
    corpus_path, manifest_path = output_dir / "corpus.jsonl", output_dir / "manifest.jsonl"
    rejects_path = output_dir / "rejects.jsonl"
    done = {str(row["doc_id"]) for row in iter_jsonl(manifest_path)} if manifest_path.exists() else set()
    tasks = [(c, s) for c in candidates for s in range(samples) if doc_id_for(c["query_id"], s) not in done]
    summary["already_done"] = len(done)
    lock = threading.Lock()
    counts = {"generated": 0, "rejected": 0, "attempts": 0}
    every = int(config.get("progress_every", 25))

    def work(task):
        candidate, sample = task
        return candidate, sample, generate_one(client, template, candidate, sample=sample, settings=settings)

    with ThreadPoolExecutor(max_workers=max(1, int(config.get("workers", 4)))) as pool:
        futures = [pool.submit(work, task) for task in tasks]
        for future in as_completed(futures):
            candidate, sample, result = future.result()
            doc_id = doc_id_for(candidate["query_id"], sample)
            with lock:
                counts["attempts"] += len(result["attempts"])
                if result["ok"]:
                    receipt = {"query_id": candidate["query_id"], "question": candidate["question"],
                               "target_answer": candidate["target"], "gold_answer": candidate["gold"],
                               "origin": "attack_v1", "sample": sample, "model": client.model,
                               "prompt_hash": stable_hash(template)[:12], "signature": signature[:12],
                               "attempts": len(result["attempts"]),
                               "words": len(result["text"].split()),
                               "row_hash": stable_hash({"title": result["title"], "text": result["text"]})}
                    # Receipt first, then the row: a crash leaves no document without its receipt.
                    append_jsonl(manifest_path, [{"doc_id": doc_id, "receipt": receipt}])
                    append_jsonl(corpus_path, [{"_id": doc_id, "query_id": candidate["query_id"],
                                                "title": result["title"], "text": result["text"]}])
                    counts["generated"] += 1
                else:
                    append_jsonl(rejects_path, [{"doc_id": doc_id, "query_id": candidate["query_id"],
                                                 "target_answer": candidate["target"],
                                                 "attempts": result["attempts"]}])
                    counts["rejected"] += 1
                total = counts["generated"] + counts["rejected"]
                if every and total % every == 0:
                    print(f"attack generation: {total}/{len(tasks)} "
                          f"(accepted {counts['generated']}, rejected {counts['rejected']})", flush=True)
    summary.update(counts)
    summary["documents_total"] = len(done) + counts["generated"]
    return summary
