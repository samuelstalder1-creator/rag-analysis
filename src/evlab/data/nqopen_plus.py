"""Build a reproducible Cocktail/NQ-Open subset with imported distractor targets."""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import csv
from pathlib import Path
import unicodedata

from evlab.data.nqplus import load_query_rows, load_generation_candidates
from evlab.generation.false_answer import proposal_status, answer_type
from evlab.generation.source_constraints import check_source_constraints
from evlab.generation.false_answers import (
    answer_source_hash,
    deterministic_answer_receipt,
)
from evlab.io import (
    file_sha256,
    iter_jsonl,
    read_qrels,
    stable_hash,
    write_json,
    write_jsonl,
)


def normalize_question(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()).rstrip(
        " ?"
    )


def split_answers(value: str) -> list[str]:
    return list(
        dict.fromkeys(part.strip() for part in value.split(";") if part.strip())
    )


def build_nqopen_plus(config: dict) -> dict:
    settings = config["dataset_build"]
    output = Path(config["data"]["nqplus_dir"])
    inputs = {
        "nqplus_queries": Path(settings["source_nqplus_dir"]) / "queries.jsonl",
        "cocktail_queries": Path(settings["cocktail_queries"]),
        "cocktail_qrels": Path(settings["cocktail_qrels"]),
        "nqopen_csv": Path(settings["nqopen_csv"]),
        "human_corpus": Path(config["data"]["human_corpus"]),
    }
    labels = Path(settings["source_nqplus_dir"]) / "generated_labels.jsonl"
    if labels.exists():
        inputs["nqplus_labels"] = labels
    if any(
        output.resolve() == p.resolve().parent
        or output.resolve() in p.resolve().parents
        for p in inputs.values()
    ):
        raise ValueError("nqopen+ output must be separate from its input directories")
    rows = load_query_rows(settings["source_nqplus_dir"])
    selected = [
        r for r in rows if r.get("metadata", {}).get("usable_for", {}).get("generation")
    ]
    expected = settings.get("expected_questions")
    if expected is not None and len(selected) != expected:
        raise ValueError(
            f"Expected {expected} generation questions, found {len(selected)}"
        )
    if len({str(r["_id"]) for r in selected}) != len(selected):
        raise ValueError("Duplicate source query IDs")
    if any(
        not all(
            r["metadata"]["usable_for"].get(usage) for usage in ("retrieval", "reader")
        )
        for r in selected
    ):
        raise ValueError(
            "Every nqopen+ question must support retrieval and reader evaluation"
        )
    cocktail = {}
    for row in iter_jsonl(inputs["cocktail_queries"]):
        qid = str(row["_id"])
        if qid in cocktail:
            raise ValueError(f"Duplicate Cocktail query ID: {qid}")
        cocktail[qid] = row
    index = defaultdict(list)
    with inputs["nqopen_csv"].open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"question", "best_answer", "correct_answers", "incorrect_answers"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("NQ-Open CSV lacks required question/answer columns")
        for position, row in enumerate(reader, 2):
            index[normalize_question(row["question"])].append((position, row))
    source_candidates = {
        c["query_id"]: c
        for c in load_generation_candidates(
            settings["source_nqplus_dir"], inputs["human_corpus"]
        )
    }
    prepared, imported, review = [], [], []
    for original in selected:
        row = copy.deepcopy(original)
        qid = str(row["_id"])
        if qid not in cocktail or cocktail[qid]["text"] != row["text"]:
            raise ValueError(f"Cocktail ID/text mismatch: {qid}")
        matches = index.get(normalize_question(row["text"]), [])
        if len(matches) != 1:
            raise ValueError(
                f"Expected unique NQ-Open question match for {qid}, found {len(matches)}"
            )
        position, external = matches[0]
        candidate = source_candidates[qid]
        metadata = row["metadata"]
        # Preserve the effective NQ+ gold/aliases and question fingerprints. Raw
        # external alternatives remain available without silently changing gold.
        external_correct = split_answers(external["correct_answers"])
        protected = list(
            dict.fromkeys(
                [
                    *candidate["answer_aliases"],
                    external["best_answer"],
                    *external_correct,
                ]
            )
        )
        sources = [d["title"] + "\n" + d["text"] for d in candidate["h_docs"]]
        proposals = []
        for target in split_answers(external["incorrect_answers"]):
            status = proposal_status(
                target,
                candidate["answer"],
                protected,
                candidate["query"],
                set(),
                sources,
            )
            proposals.append({"text": target, "status": status})
        document_checks = {
            p["text"]: check_source_constraints(
                gold=candidate["answer"],
                target=p["text"],
                documents=candidate["h_docs"],
            )
            for p in proposals
        }
        available = [
            p["text"]
            for p in proposals
            if p["status"] == "valid" and document_checks[p["text"]]["eligible"]
        ]
        reasons = []
        gold_checks = check_source_constraints(
            gold=candidate["answer"], target=None, documents=candidate["h_docs"]
        )
        if not gold_checks["all_gold_present"]:
            reasons.append("full_gold_missing_from_at_least_one_source")
        if not available:
            reasons.append("no_usable_imported_target")
        target = available[0] if available and not reasons else None
        metadata.update(
            nqopen={
                "csv_record": position,
                "raw": external,
                "correct_answers": external_correct,
                "incorrect_answers": split_answers(external["incorrect_answers"]),
                "match": (
                    "exact" if row["text"] == external["question"] else "normalized"
                ),
            },
            cocktail={"query_id": qid, "raw": cocktail[qid]},
            false_answer=target,
            false_answer_status="ready" if target else "needs_review",
            false_answer_source="nqopen_import",
            false_answer_candidates=proposals,
            false_answer_review_reasons=reasons,
            false_answer_source_checks=document_checks,
        )
        if target:
            verification = deterministic_answer_receipt(candidate, target)
            verification["source_constraints"] = document_checks[target]
            imported.append(
                {
                    "_id": qid,
                    "false_answer": target,
                    "source_hash": answer_source_hash(
                        candidate["query"],
                        candidate["answer"],
                        candidate["answer_aliases"],
                    ),
                    "answer_type": answer_type(candidate["answer"], candidate["query"]),
                    "verification": verification,
                    "origin": "nqopen_import",
                    "nqopen_record_hash": stable_hash(external),
                    "nqopen_csv_record": position,
                    "candidates": proposals,
                    "source_document_checks": document_checks,
                    "valid_candidates": available,
                    "selection": {
                        "method": "first_valid_imported_target",
                        "selected": target,
                        "selector_model": None,
                    },
                }
            )
        else:
            review.append(
                {
                    "query_id": qid,
                    "question": row["text"],
                    "gold": metadata["answer"],
                    "reasons": reasons,
                    "candidates": proposals,
                    "source_document_checks": document_checks,
                }
            )
        prepared.append(row)
    selected_ids = {str(row["_id"]) for row in prepared}
    qrels = [
        r for r in read_qrels(inputs["cocktail_qrels"]) if r.query_id in selected_ids
    ]
    from_qrels = defaultdict(set)
    for rel in qrels:
        if rel.relevance > 0:
            from_qrels[rel.query_id].add(rel.doc_id)
    for row in prepared:
        if from_qrels[str(row["_id"])] != set(row["metadata"]["relevant_docs"]):
            raise ValueError(f"qrels/relevant_docs mismatch: {row['_id']}")
    relevant_ids = {r.doc_id for r in qrels}
    relevant = [
        r for r in iter_jsonl(inputs["human_corpus"]) if str(r["_id"]) in relevant_ids
    ]
    if {str(r["_id"]) for r in relevant} != relevant_ids:
        raise ValueError("Missing relevant human documents")
    stats = {
        "questions": len(prepared),
        "reader_questions": len(prepared),
        "generation_questions": len(prepared),
        "answer_documents": sum(
            len(r["metadata"]["docs_with_answer"]) for r in prepared
        ),
        "qrels": len(qrels),
        "relevant_documents": len(relevant),
        "ready_false_answers": len(imported),
        "needs_review": len(review),
        "review_reasons": dict(
            Counter(reason for r in review for reason in r["reasons"])
        ),
        "proposal_statuses": dict(
            Counter(
                p["status"]
                for r in prepared
                for p in r["metadata"]["false_answer_candidates"]
            )
        ),
    }
    payloads = {
        "queries.jsonl": prepared,
        "false_answers.jsonl": imported,
        "false_answer_review.jsonl": review,
        "corpus/relevant.jsonl": relevant,
    }
    # Refuse to combine a rebuilt dataset with old overlay labels or target files.
    if (output / "generated_labels.jsonl").exists():
        raise ValueError(
            "nqopen+ output has unexpected generated_labels.jsonl; use a clean output directory"
        )
    for filename, data in payloads.items():
        write_jsonl(output / filename, data)
    (output / "qrels").mkdir(parents=True, exist_ok=True)
    with (output / "qrels/test.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows((r.query_id, r.doc_id, r.relevance) for r in qrels)
    write_json(
        output / "dataset.json",
        {
            "name": "nqopen+",
            "version": 1,
            "selection": "effective NQ+ generation-eligible questions; no target-based query filtering",
            "statistics": stats,
            "sources": {
                key: {"path": str(path), "sha256": file_sha256(path)}
                for key, path in inputs.items()
            },
            "target_validation": "deterministic rules only; semantic_verified=false",
            "retrieval_corpus": str(inputs["human_corpus"]),
            "relevant_corpus": "corpus/relevant.jsonl (relevant documents only, not the full retrieval corpus)",
            "files": {
                name: file_sha256(output / name)
                for name in [*payloads, "qrels/test.tsv"]
            },
        },
    )
    return stats
