"""Separate answer variants, lists, conflicts, and cases needing review."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from evlab.generation.validate import contains_alias
from evlab.io import iter_jsonl, write_json, write_jsonl


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
    llm_corpus: str | Path | None = None,
) -> dict:
    root = Path(nqplus_dir)
    queries = list(iter_jsonl(root / "queries.jsonl"))
    candidates = list(iter_jsonl(root / "generation_candidates.jsonl"))
    reviews = json.loads(Path(reviews_file).read_text(encoding="utf-8"))
    needed = {str(did) for row in queries for did in row["metadata"].get("relevant_docs", [])}
    human = {str(row["_id"]): row for row in iter_jsonl(human_corpus) if str(row["_id"]) in needed}
    llm = ({str(row["_id"]): row for row in iter_jsonl(llm_corpus) if str(row["_id"]) in needed}
           if llm_corpus is not None else {})
    classified = []
    review_queue = []
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
            if kind == "unresolved":
                review_queue.append(classified[-1])
        flags["answer_is_list"] = kind in {"list", "list_with_conflict"}
        flags["answer_conflict"] = kind in {"conflict", "list_with_conflict"}
        flags["answer_review_required"] = kind == "unresolved"
        metadata["usable_for"]["generation"] = bool(
            metadata["usable_for"]["reader"] and kind in {"single", "variants"}
        )

    query_by_id = {row["_id"]: row for row in queries}
    # Preserve existing order so generation seeds and resume behavior remain stable.
    candidates = [row for row in candidates if query_by_id[row["query_id"]]["metadata"]["usable_for"]["generation"]]
    existing = {row["query_id"] for row in candidates}
    added = []
    for row in queries:
        metadata = row["metadata"]
        if not metadata["usable_for"]["generation"] or row["_id"] in existing:
            continue
        aliases = metadata["answer_aliases"]
        h_docs = []
        for did in metadata["docs_with_answer"]:
            source = human[did]
            source_content = source.get("title", "") + "\n" + source["text"]
            llm_doc = llm.get(did)
            h_docs.append({"doc_id": did, "title": source.get("title", ""), "text": source["text"],
                           "aliases_in_doc": [alias for alias in aliases if contains_alias(source_content, alias)],
                           "llm_copy_keeps_answer": (
                               any(contains_alias(llm_doc.get("title", "") + "\n" + llm_doc["text"], alias) for alias in aliases)
                               if llm_doc else None
                           )})
        candidates.append({"query_id": row["_id"], "query": row["text"], "answer": metadata["answer"],
                           "answer_aliases": aliases, "answer_kind": metadata["answer_kind"], "h_docs": h_docs,
                           "other_relevant_docs": [did for did in metadata["relevant_docs"] if did not in metadata["docs_with_answer"]]})
        added.append(row["_id"])

    rag = list(iter_jsonl(root / "rag_queries.jsonl"))
    for row in rag:
        metadata = query_by_id[row["query_id"]]["metadata"]
        row["answer_kind"] = metadata["answer_kind"]
        row["answer_flags"] = {key: metadata["flags"][key] for key in
                               ("answer_is_list", "answer_conflict", "answer_review_required")}
    summary = {"previously_combined_cases": len(classified),
               "classification": dict(Counter(row["kind"] for row in classified)),
               "new_generation_candidates": len(added), "new_generation_query_ids": added,
               "generation_candidates": len(candidates),
               "excluded_generation": len(queries) - len(candidates),
               "excluded_generation_by_kind": dict(Counter(row["metadata"]["answer_kind"] for row in queries
                                                          if not row["metadata"]["usable_for"]["generation"])),
               "note": "Flags indicate detected conditions; unresolved cases require review. List detection does not certify list membership."}
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    report.setdefault("pre_cleaning_no_qrels", report.get("flag_no_qrels", 0))
    report["flag_counts_scope"] = "clean_queries"
    report.pop("flag_answer_list_or_conflict", None)
    for key in queries[0]["metadata"]["flags"]:
        report[f"flag_{key}"] = sum(bool(row["metadata"]["flags"][key]) for row in queries)
    h_docs = [doc for row in candidates for doc in row["h_docs"]]
    report.update({"3_generation": len(candidates), "gen_h_docs_total": len(h_docs),
                   "gen_unique_h_docs": len({doc["doc_id"] for doc in h_docs}),
                   "gen_multiple_h_docs": sum(len(row["h_docs"]) > 1 for row in candidates),
                   "gen_kind_single": sum(row["answer_kind"] == "single" for row in candidates),
                   "gen_kind_variants": sum(row["answer_kind"] == "variants" for row in candidates),
                   "gen_llm_copy_keeps_answer_all_h": sum(all(doc.get("llm_copy_keeps_answer") is True for doc in row["h_docs"]) for row in candidates),
                   "answer_classification": summary})
    # Build and check everything before changing the dataset.
    write_jsonl(root / "queries.jsonl", queries)
    write_jsonl(root / "rag_queries.jsonl", rag)
    write_jsonl(root / "generation_candidates.jsonl", candidates)
    write_jsonl(root / "answer_classification.jsonl", classified)
    write_jsonl(root / "answer_review.jsonl", review_queue)
    write_json(root / "report.json", report)
    return summary
