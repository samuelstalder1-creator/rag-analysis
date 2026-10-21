"""Separate answer variants, lists, conflicts, and cases needing review."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from evlab.generation.validate import contains_alias
from evlab.io import iter_jsonl, write_jsonl, stable_hash


CARDINAL = r"(?:\d+|two|three|four|five|six|seven|eight|nine|ten)"
LIST_QUESTION = re.compile(
    rf"\b(?:name|list|describe|identify) (?:the )?{CARDINAL}\b"
    rf"|^(?:the )?{CARDINAL} (?:main )?(?:types|examples|tools|phases)\b"
    rf"|^what (?:are|were) (?:the )?(?:{CARDINAL}\b|colou?rs\b|functions\b|roles\b|parts\b|names\b|types\b|active materials\b|main sources\b)"
    rf"|^who (?:are|were) (?:the )?(?:{CARDINAL}\b|members\b|judges\b|leaders\b|characters\b|twins\b)"
    r"|^who starred\b|\b(?:star cast|original cast|video cast)\b"
    r"|^what (?:mlb |nba |nfl )?teams\b|^what states\b"
    r"|\b(?:begin|start) and (?:end|stop)\b"
    r"|\bking and queen\b"
    r"|\band (?:what|where|when|by whom|why)\b"
    r"|^all the gases\b|^different ways to spell\b"
    r"|^what awards\b|^which two\b|\bbetween what two\b",
    re.IGNORECASE,
)


def normalize_alias(alias: str) -> str:
    text = unicodedata.normalize("NFKC", alias).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def classify_answer_set(question: str, aliases: list[str], review: dict | None = None) -> dict:
    if review is not None:
        if review["kind"] not in {"variants", "list", "conflict", "list_with_conflict", "unresolved"}:
            raise ValueError(f"Unsupported reviewed answer kind: {review['kind']}")
        if set(review["aliases"]) != set(aliases):
            raise ValueError("Reviewed answer strings no longer match the dataset")
        return {"kind": review["kind"], "method": "evidence_review",
                "reason": review["reason"], "evidence": review["evidence"]}
    normalized = {normalize_alias(alias) for alias in aliases if alias.strip()}
    if len(normalized) == 1:
        return {"kind": "variants", "method": "normalization",
                "reason": "Answer strings differ only in case, punctuation, or whitespace."}
    if LIST_QUESTION.search(question):
        return {"kind": "list", "method": "question_structure",
                "reason": "Question explicitly requests multiple members or multiple answer slots."}
    # A combined answer plus its separately annotated members is a list, not aliases.
    for combined in aliases:
        if " and " not in combined.casefold():
            continue
        members = [alias for alias in aliases if alias != combined and contains_alias(combined, alias)]
        for first in members:
            for second in members:
                if first != second and not contains_alias(first, second) and not contains_alias(second, first):
                    if re.search(re.escape(first) + r"\s+and\s+" + re.escape(second), combined, re.IGNORECASE):
                        return {"kind": "list", "method": "combined_answer",
                                "reason": "Annotations contain both a coordinated answer and its distinct members."}
    return {"kind": "unresolved", "method": "needs_review",
            "reason": "Distinct strings alone do not establish a list, synonyms, or a same-scope conflict."}


def split_answer_flags(
    nqplus_dir: str | Path,
    *,
    human_corpus: str | Path,
    reviews_file: str | Path,
) -> dict:
    root = Path(nqplus_dir)
    queries = list(iter_jsonl(root / "queries.jsonl"))
    source_hashes = {str(row["_id"]): stable_hash(row) for row in queries}
    # Source records are the input on every run; stale labels can be regenerated.
    labels_path = root / "generated_labels.jsonl"
    previous_rows = list(iter_jsonl(labels_path)) if labels_path.exists() else queries
    previous_generation = {row["_id"] for row in previous_rows if row["metadata"]["usable_for"]["generation"]}
    reviews = json.loads(Path(reviews_file).read_text(encoding="utf-8"))
    needed = {str(did) for row in queries for did in row["metadata"].get("relevant_docs", [])}
    human = {str(row["_id"]): row for row in iter_jsonl(human_corpus) if str(row["_id"]) in needed}
    classified = []
    for row in queries:
        metadata = row["metadata"]
        flags = metadata["flags"]
        previous = metadata.get("answer_classification", {})
        was_ambiguous = flags.pop("answer_list_or_conflict", False) or previous.get("previous_kind") == "list_or_conflict"
        kind = metadata["answer_kind"]
        if was_ambiguous:
            review = reviews.get(row["_id"])
            if review is not None:
                for evidence in review["evidence"]:
                    doc = human[evidence["doc_id"]]
                    if evidence["doc_id"] not in metadata["relevant_docs"] or evidence["quote"] not in doc["title"] + "\n" + doc["text"]:
                        raise ValueError(f"Review evidence does not match source for {row['_id']}")
            result = classify_answer_set(row["text"], metadata["answer_aliases"], review)
            kind = metadata["answer_kind"] = result["kind"]
            metadata["answer_classification"] = {"previous_kind": "list_or_conflict", **result}
            classified.append({"query_id": row["_id"], "query": row["text"],
                               "answer_aliases": metadata["answer_aliases"],
                               "relevant_docs": metadata["relevant_docs"], **result})
        flags["answer_is_list"] = kind in {"list", "list_with_conflict"}
        flags["answer_conflict"] = kind in {"conflict", "list_with_conflict"}
        flags["answer_review_required"] = kind == "unresolved"
        metadata["usable_for"]["generation"] = bool(
            metadata["usable_for"]["reader"] and kind in {"single", "variants"}
        )

    eligible = [row for row in queries if row["metadata"]["usable_for"]["generation"]]
    added = [row["_id"] for row in eligible if row["_id"] not in previous_generation]
    summary = {"previously_combined_cases": len(classified),
               "classification": dict(Counter(row["kind"] for row in classified)),
               "new_generation_candidates": len(added), "new_generation_query_ids": added,
               "generation_candidates": len(eligible),
               "excluded_generation": len(queries) - len(eligible),
               "excluded_generation_by_kind": dict(Counter(row["metadata"]["answer_kind"] for row in queries
                                                          if not row["metadata"]["usable_for"]["generation"])),
               "review_required": sum(row["metadata"]["flags"]["answer_review_required"] for row in queries),
               "labels_file": str(root / "generated_labels.jsonl"),
               "note": "Source queries are unchanged; labels are joined by query ID when loading."}
    # Save only derived fields, with a hash linking them to unchanged source rows.
    write_jsonl(root / "generated_labels.jsonl", [
        {"_id": row["_id"], "source_hash": source_hashes[str(row["_id"])],
         "metadata": {key: row["metadata"][key] for key in
                      ("answer_kind", "flags", "usable_for", "answer_classification") if key in row["metadata"]}}
        for row in queries
    ])
    return summary
