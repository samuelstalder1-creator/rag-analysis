"""Revalidate a generated attack corpus and write a reproducible coverage report."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json

from evlab.config import load_config
from evlab.evaluation.answers import classify_answer
from evlab.generation.attack import (
    VALIDATION_VERSION, doc_id_for, load_attack_candidates, resolve_attack_settings, validate_attack,
)
from evlab.io import file_sha256, iter_jsonl, stable_hash


def audit(config: dict) -> dict:
    generation = config["generation"]
    dataset = Path(config["data"]["nqplus_dir"])
    output = Path(generation["output_dir"])
    settings = resolve_attack_settings(generation)
    candidates, skipped = load_attack_candidates(dataset)
    samples = int(generation.get("samples_per_query", 1))
    expected = {doc_id_for(c["query_id"], s): c for c in candidates for s in range(samples)}
    corpus = list(iter_jsonl(output / "corpus.jsonl"))
    manifest = list(iter_jsonl(output / "manifest.jsonl"))
    docs = {r["_id"]: r for r in corpus}
    receipts = {r["doc_id"]: r["receipt"] for r in manifest}
    issues = []
    for name, values in [("corpus", [r["_id"] for r in corpus]),
                         ("manifest", [r["doc_id"] for r in manifest])]:
        for doc_id, count in Counter(values).items():
            if count > 1:
                issues.append({"doc_id": doc_id, "reason": f"duplicate_{name}", "count": count})
    if set(docs) != set(receipts):
        issues.append({"reason": "corpus_manifest_id_mismatch",
                       "without_receipt": sorted(set(docs) - set(receipts)),
                       "without_document": sorted(set(receipts) - set(docs))})
    checked = []
    for doc_id, row in docs.items():
        if doc_id not in expected:
            issues.append({"doc_id": doc_id, "reason": "unknown_document_id"})
            continue
        candidate = expected[doc_id]
        reasons = validate_attack(row["title"], row["text"], target=candidate["target"],
                                  aliases=candidate["aliases"],
                                  min_words=settings["min_words"], max_words=settings["max_words"],
                                  strict_length=settings.get("strict_length", False))
        receipt = receipts.get(doc_id, {})
        row_hash = stable_hash({"title": row["title"], "text": row["text"]})
        if receipt.get("row_hash") != row_hash:
            reasons.append("row_hash_mismatch")
        if receipt.get("target_answer") != candidate["target"]:
            reasons.append("stale_target_answer")
        if receipt.get("gold_answer") != candidate["gold"]:
            reasons.append("stale_gold_answer")
        if row.get("query_id") != candidate["query_id"] or receipt.get("query_id") != candidate["query_id"]:
            reasons.append("query_id_mismatch")
        if classify_answer(candidate["target"], gold_aliases=candidate["aliases"],
                           target_answer=candidate["target"]) != "target":
            reasons.append("target_not_distinguishable_in_evaluation")
        issues.extend({"doc_id": doc_id, "reason": reason} for reason in reasons)
        checked.append({"doc_id": doc_id, "row_hash": row_hash, "passed": not reasons})
    missing = sorted(set(expected) - set(docs))
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "validation_version": 3 if settings.get("strict_length") else VALIDATION_VERSION,
        "length_settings": settings,
        "mean_body_words": sum(len(r["text"].split()) for r in corpus) / len(corpus) if corpus else None,
        "questions": len(candidates), "samples_per_query": samples,
        "expected_documents": len(expected), "documents": len(corpus),
        "unique_documents": len(docs), "manifest_rows": len(manifest),
        "coverage": len(set(docs) & set(expected)) / len(expected) if expected else 0,
        "missing_document_ids": missing, "skipped_without_target": skipped,
        "issues": issues, "complete": not missing and not issues and not skipped,
        "checks": checked,
        "sha256": {str(p): file_sha256(p) for p in [
            dataset / "answers.jsonl", dataset / "queries.jsonl",
            output / "corpus.jsonl", output / "manifest.jsonl",
            Path("src/evlab/generation/attack.py"), Path("src/evlab/evaluation/answers.py"),
        ]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    report = audit(config)
    destination = args.report or Path(config["generation"]["output_dir"]) / "validation_report.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"checks", "sha256"}}, indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
