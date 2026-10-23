"""Build a separate source-grounded version; preserve the original benchmark inputs."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path

from evlab.config import load_config
from evlab.generation.false_answer import answer_type, is_valid_false_answer
from evlab.generation.false_answers import (
    answer_source_hash,
    deterministic_answer_receipt,
)
from evlab.generation.source_constraints import (
    check_source_constraints,
    full_answer_forms,
)
from evlab.generation.validate import contains_alias
from evlab.io import (
    file_sha256,
    iter_jsonl,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl,
)


def build(config: dict) -> dict:
    settings = config["dataset_resolution"]
    source = Path(settings["source_dir"])
    output = Path(config["data"]["nqplus_dir"])
    if (
        output.resolve() == source.resolve()
        or output.resolve() in source.resolve().parents
    ):
        raise ValueError("Resolved output must not overwrite the original dataset")
    audit = read_jsonl(source / "all_cases_checked.jsonl")
    queries = read_jsonl(source / "queries.jsonl")
    spec = json.loads(Path(settings["resolutions_file"]).read_text())
    resolutions = {r["query_id"]: r for r in spec["resolutions"]}
    expected = {
        r["query_id"]
        for r in audit
        if r["context_review_required"] and not r["blocking_reasons"]
    }
    if (
        len(resolutions) != len(spec["resolutions"])
        or set(resolutions) != expected
        or len(expected) != settings["expected_context_cases"]
    ):
        raise ValueError(
            "Resolution IDs must cover exactly the 48 previously passing context-review cases"
        )
    if len(queries) != settings["expected_questions"] or len(
        {r["_id"] for r in queries}
    ) != len(queries):
        raise ValueError("Unexpected query scope or duplicate IDs")
    by_audit = {r["query_id"]: r for r in audit}
    cleaned = {
        r["query_id"]: r
        for r in read_jsonl(source / "false_answer_review_cleaned.jsonl")
    }
    needed = {did for r in queries for did in r["metadata"]["relevant_docs"]}
    corpus = {
        r["_id"]: r
        for r in iter_jsonl(config["data"]["human_corpus"])
        if r["_id"] in needed
    }
    inputs = [
        source / name
        for name in [
            "queries.jsonl",
            "all_cases_checked.jsonl",
            "false_answers.jsonl",
            "false_answer_review_cleaned.jsonl",
            "dataset.json",
        ]
    ]
    inputs.extend(
        [Path(settings["resolutions_file"]), Path(config["data"]["human_corpus"])]
    )
    before = {str(p): file_sha256(p) for p in inputs}
    prepared = []
    answers = []
    blocked = []
    reviewed = []
    audited = []
    for original in queries:
        row = copy.deepcopy(original)
        qid = row["_id"]
        old = by_audit[qid]
        resolution = resolutions.get(qid)
        metadata = row["metadata"]
        question = row["text"]
        gold = old["gold"]
        target = old["false_answer"]
        docs = list(metadata["docs_with_answer"])
        previous_gold = metadata["answer"]
        if resolution:
            question = resolution.get("question", question)
            gold = resolution.get("gold", gold)
            target = resolution.get("false_answer", target)
            docs = resolution.get("answer_doc_ids", docs)
            if not docs or not set(docs) <= set(metadata["relevant_docs"]):
                raise ValueError(
                    f"Resolution must use original relevant document IDs: {qid}"
                )
            # Rewritten questions have new, explicitly reviewed evidence qrels.
            # Retain the old judgments in lineage rather than claiming official parity.
            if question != row["text"] or docs != metadata["docs_with_answer"]:
                metadata["relevant_docs"] = list(docs)
            metadata["context_resolution"] = {
                "version": spec["version"],
                "method": spec["method"],
                "reason": resolution["reason"],
                "original_question": row["text"],
                "original_gold": previous_gold,
                "original_aliases": list(metadata["answer_aliases"]),
                "original_target": old["false_answer"],
                "original_answer_doc_ids": list(metadata["docs_with_answer"]),
                "original_relevant_doc_ids": list(
                    original["metadata"]["relevant_docs"]
                ),
                "question_changed": question != row["text"],
                "gold_changed": gold != previous_gold,
                "target_changed": target != old["false_answer"],
                "answer_documents_changed": docs != metadata["docs_with_answer"],
            }
        row["text"] = question
        metadata["answer"] = gold
        if gold != previous_gold:
            metadata["answer_aliases"] = full_answer_forms(gold)
        else:
            metadata["answer_aliases"] = list(
                dict.fromkeys([gold, *metadata["answer_aliases"]])
            )
        if resolution and resolution.get("answer_aliases"):
            metadata["answer_aliases"] = list(
                dict.fromkeys([gold, *resolution["answer_aliases"]])
            )
        # Include raw equivalent forms (e.g. non-breaking spaces) already supplied
        # by the source annotations, without retaining obsolete corrected golds.
        if gold == previous_gold:
            metadata["answer_aliases"] = list(
                dict.fromkeys([*metadata["answer_aliases"], *full_answer_forms(gold)])
            )
        metadata["docs_with_answer"] = list(docs)
        metadata["answer_kind"] = (
            "variants" if len(metadata["answer_aliases"]) > 1 else "single"
        )
        hdocs = []
        for did in docs:
            document = corpus[did]
            content = document["title"] + "\n" + document["text"]
            hdocs.append(
                {
                    "doc_id": did,
                    "title": document["title"],
                    "text": document["text"],
                    "aliases_in_doc": [
                        a
                        for a in metadata["answer_aliases"]
                        if contains_alias(content, a)
                    ],
                }
            )
        # If a gold has Unicode-equivalent literal source spellings, carry those
        # exact spellings forward for the existing minimal-edit stage.
        if qid in cleaned:
            old_forms = cleaned[qid].get("source_answer_forms", [])
            from evlab.generation.source_constraints import normalize_source_text

            for form in old_forms:
                if (
                    normalize_source_text(form).casefold()
                    == normalize_source_text(gold).casefold()
                ):
                    metadata["answer_aliases"] = list(
                        dict.fromkeys([*metadata["answer_aliases"], form])
                    )
            for d in hdocs:
                d["aliases_in_doc"] = [
                    a
                    for a in metadata["answer_aliases"]
                    if contains_alias(d["title"] + "\n" + d["text"], a)
                ]
        candidate = {
            "query_id": qid,
            "query": question,
            "answer": gold,
            "answer_aliases": metadata["answer_aliases"],
            "h_docs": hdocs,
        }
        checks = check_source_constraints(gold=gold, target=target, documents=hdocs)
        original_target_check = None
        if resolution:
            original_target_check = check_source_constraints(
                gold=gold,
                target=target,
                documents=[
                    {
                        "doc_id": did,
                        "title": corpus[did]["title"],
                        "text": corpus[did]["text"],
                    }
                    for did in original["metadata"]["relevant_docs"]
                ],
            )
            if not original_target_check["all_target_absent"]:
                raise ValueError(
                    f"Target occurs in an original relevant document: {qid}"
                )
        structural = is_valid_false_answer(
            target, gold, metadata["answer_aliases"], question=question
        )
        errors = []
        if not checks["all_gold_present"]:
            errors.append("full_gold_missing_in_source")
        if not checks["all_target_absent"]:
            errors.append("target_present_in_source")
        if not structural:
            errors.append("structurally_invalid_target")
        if resolution and errors:
            raise ValueError(
                f"Resolved case still fails hard conditions: {qid}: {errors}"
            )
        metadata["false_answer"] = target if not errors else None
        metadata["proposed_false_answer"] = target
        metadata["false_answer_status"] = "ready" if not errors else "needs_review"
        metadata["false_answer_source"] = (
            "source_grounded_context_resolution" if resolution else old["target_origin"]
        )
        metadata["false_answer_review_reasons"] = errors
        metadata["false_answer_source_checks"] = {target: checks}
        # The source CSV is kept as provenance, not as a second competing gold set
        # after a documented answer correction.
        if resolution and gold != previous_gold:
            metadata["answer_source"] = "source_grounded_context_resolution"
        evidence = [
            {
                "doc_id": d["doc_id"],
                "title": d["title"],
                "text": d["text"],
                "source_hash": stable_hash(corpus[d["doc_id"]]),
            }
            for d in hdocs
        ]
        if resolution:
            reviewed.append(
                {
                    "query_id": qid,
                    "question": question,
                    "gold": gold,
                    "false_answer": target,
                    "answer_aliases": metadata["answer_aliases"],
                    "source_doc_ids": docs,
                    "resolution": metadata["context_resolution"],
                    "source_constraints": checks,
                    "original_relevant_target_check": original_target_check,
                    "evidence": evidence,
                    "source_review_completed": True,
                    "independent_semantic_verifier": False,
                    "semantic_verified": False,
                }
            )
        if not errors:
            verification = deterministic_answer_receipt(candidate, target)
            verification["source_constraints"] = checks
            answers.append(
                {
                    "_id": qid,
                    "false_answer": target,
                    "source_hash": answer_source_hash(
                        question, gold, metadata["answer_aliases"]
                    ),
                    "answer_type": answer_type(gold, question),
                    "origin": metadata["false_answer_source"],
                    "verification": verification,
                    "selection": {
                        "method": (
                            "source_grounded_resolution"
                            if resolution
                            else "audited_existing_target"
                        ),
                        "selected": target,
                        "selector_model": None,
                    },
                    "parent_audit_row_hash": stable_hash(old),
                    "resolution_hash": stable_hash(resolution) if resolution else None,
                }
            )
        else:
            blocked.append(
                {
                    "query_id": qid,
                    "question": question,
                    "gold": gold,
                    "false_answer": target,
                    "reasons": errors,
                    "source_constraints": checks,
                }
            )
        audited.append(
            {
                "query_id": qid,
                "question": question,
                "gold": gold,
                "false_answer": target,
                "blocking_reasons": errors,
                "source_constraints": checks,
                "context_review_required": old["context_review_required"]
                and not bool(resolution),
                "resolution_applied": bool(resolution),
                "semantic_verified": False,
            }
        )
        prepared.append(row)
    stats = {
        "questions": len(prepared),
        "resolved_context_cases": len(reviewed),
        "question_changes": sum(r["resolution"]["question_changed"] for r in reviewed),
        "gold_corrections": sum(r["resolution"]["gold_changed"] for r in reviewed),
        "target_changes": sum(r["resolution"]["target_changed"] for r in reviewed),
        "source_assignment_changes": sum(
            r["resolution"]["answer_documents_changed"] for r in reviewed
        ),
        "ready_false_answers": len(answers),
        "source_blocked_questions": len(blocked),
        "answer_documents": sum(
            len(r["metadata"]["docs_with_answer"]) for r in prepared
        ),
        "qrels": sum(len(r["metadata"]["relevant_docs"]) for r in prepared),
        "remaining_context_reviews_among_ready": sum(
            r["context_review_required"] and not r["blocking_reasons"] for r in audited
        ),
    }
    if (
        len(reviewed) != settings["expected_context_cases"]
        or stats["remaining_context_reviews_among_ready"]
    ):
        raise ValueError("Unresolved context cases remain")
    payloads = {
        "queries.jsonl": prepared,
        "false_answers.jsonl": answers,
        "false_answer_review.jsonl": blocked,
        "context_cases_resolved.jsonl": reviewed,
        "all_cases_checked.jsonl": audited,
        "corpus/relevant.jsonl": [
            corpus[did]
            for did in sorted(
                {did for r in prepared for did in r["metadata"]["relevant_docs"]}
            )
        ],
    }
    for name, rows in payloads.items():
        write_jsonl(output / name, rows)
    (output / "qrels").mkdir(parents=True, exist_ok=True)
    with (output / "qrels/test.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows(
            (r["_id"], did, 1)
            for r in prepared
            for did in r["metadata"]["relevant_docs"]
        )
    write_json(
        output / "dataset.json",
        {
            "name": "nqopen+_source_grounded_resolved",
            "version": 1,
            "statistics": stats,
            "not_original_benchmark_parity": bool(stats["question_changes"]),
            "source_files": {name: {"sha256": value} for name, value in before.items()},
            "original_human_corpus": config["data"]["human_corpus"],
            "question_policy": "Question/gold changes are explicitly versioned; rewritten questions use reviewed evidence qrels.",
            "validation": "All selected golds occur and targets are absent in every associated original document; 48 contexts reviewed by the assistant, not an independent judge.",
            "files": {
                name: file_sha256(output / name)
                for name in [*payloads, "qrels/test.tsv"]
            },
        },
    )
    assert all(
        file_sha256(path) == hash_value for path, hash_value in before.items()
    ), "Original inputs changed"
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/nqopen-plus-resolved.yaml")
    args = parser.parse_args()
    print(json.dumps(build(load_config(args.config)), ensure_ascii=False, indent=2))
