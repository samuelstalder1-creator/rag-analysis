"""Per-query case analysis of the selected-sources Qwen3-4B BM25 sweep.

Writes one CSV row per (variant, query) with the reader context, its sources and the
generated answer, plus a JSON file with the aggregate numbers used in the case report.
``--show test1 test2 ...`` prints the full context and answer of single questions.

    PYTHONPATH=src .venv/bin/python scripts/analyze_query_cases.py
    PYTHONPATH=src .venv/bin/python scripts/analyze_query_cases.py --show test1 --variants human_only human+gpt_oss_attack
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from evlab.config import load_config
from evlab.corpus.builder import build_corpus_with_report
from evlab.evaluation.answers import contains_answer, normalize
from evlab.experiments.runner import _load_queries
from evlab.experiments.sweep import expand_sweep

CONFIG = "configs/experiments/nqopen1520_selected_sources_qwen3_4b_bm25.yaml"
RUNS = Path("results/evaluations/qwen3_4b_selected_bm25_20261027/runs")
OUT = Path("results/analysis/selected_bm25_qwen3_4b_20261028")
SOURCE_CODES = {
    "resolved_correct": "RC", "resolved_false": "RF", "qwen3_correct": "QC",
    "qwen3_false": "QF", "gpt-oss-120b-attack-false": "A",
}
ATTACK_PREFIX = "gen::gpt-oss-120b-attack-false::atk_"
HUMAN, ALL = "human_only", "human+resolved_correct+resolved_false+qwen3_correct+qwen3_false+gpt_oss_attack"
H_A, MIX = "human+gpt_oss_attack", "human+resolved_correct+qwen3_correct+gpt_oss_attack"


def source(doc_id: str) -> str:
    parts = doc_id.split("::")
    return "H" if parts[0] == "human" else SOURCE_CODES[parts[1]]


class Sweep:
    def __init__(self) -> None:
        config = load_config(CONFIG)
        runs = expand_sweep(config)
        self.order = [label for label, _, _ in runs]
        docs, _ = build_corpus_with_report(runs[-1][2])  # every source enabled
        self.docs = {doc.doc_id: doc for doc in docs}
        self.queries = {q.query_id: q for q in _load_queries(runs[-1][2])}
        self.answers: dict[str, dict[str, dict]] = {}
        self.retrieval: dict[str, dict[str, list[str]]] = {}
        by_name = {path.name.split("_20261027__")[1].rsplit("_", 1)[0]: path for path in RUNS.iterdir()}
        for label in self.order:
            path = by_name[label.replace("+", "-")]
            self.answers[label] = {
                row["query_id"]: row for row in map(json.loads, (path / "bm25/answers.jsonl").open())
            }
            hits: dict[str, list] = collections.defaultdict(list)
            for row in map(json.loads, (path / "bm25/retrieval.jsonl").open()):
                hits[row["query_id"]].append((row["rank"], row["doc_id"]))
            self.retrieval[label] = {q: [d for _, d in sorted(v)] for q, v in hits.items()}
        self._text: dict[str, str] = {}

    # -- helpers -----------------------------------------------------------
    def text(self, doc_id: str) -> str:
        if doc_id not in self._text:
            doc = self.docs[doc_id]
            self._text[doc_id] = normalize(f"{doc.title or ''} {doc.text}")
        return self._text[doc_id]

    def contains(self, doc_id: str, answers) -> bool:
        text = self.text(doc_id)
        return any(contains_answer(text, normalize(a)) for a in answers if a)

    def relevant(self, qid: str) -> set[str]:
        return {"human::" + d for d in self.queries[qid].relevant_doc_ids}

    def row(self, variant: str, qid: str) -> dict:
        return self.answers[variant][qid]

    def context(self, variant: str, qid: str) -> list[str]:
        return self.row(variant, qid)["context_doc_ids"]

    def target(self, qid: str):
        return self.queries[qid].metadata.get("false_answer")

    def case(self, variant: str, qid: str) -> dict:
        query, row, ctx = self.queries[qid], self.row(variant, qid), self.context(variant, qid)
        roots = collections.Counter(self.docs[d].provenance_root_id for d in ctx)
        own = ATTACK_PREFIX + qid
        return {
            "variant": variant, "query_id": qid, "label": row["label"], "answer": row["answer"],
            "citations": " ".join(map(str, row["citations"])), "valid_json": row["valid_json"],
            "context_sources": "|".join(source(d) for d in ctx), "context_doc_ids": "|".join(ctx),
            "relevant_human_in_context": bool(self.relevant(qid) & set(ctx)),
            "gold_in_context": any(self.contains(d, query.aliases) for d in ctx),
            "target_in_context": bool(self.target(qid)) and any(self.contains(d, [self.target(qid)]) for d in ctx),
            "own_attack_rank": ctx.index(own) + 1 if own in ctx else 0,
            "max_versions_of_one_root": max(roots.values()),
        }

    def show(self, qid: str, variants: list[str], snippet: int = 200) -> None:
        query = self.queries[qid]
        print(f"### {qid}: {query.text}\n    gold={list(query.aliases)[:3]} target={self.target(qid)!r}")
        for variant in variants:
            row = self.row(variant, qid)
            print(f"  [{variant}] answer={row['answer']!r} label={row['label']} cites={row['citations']}")
            for index, doc_id in enumerate(row["context_doc_ids"], 1):
                doc = self.docs[doc_id]
                flags = ("G" if self.contains(doc_id, query.aliases) else "-") \
                    + ("T" if self.target(qid) and self.contains(doc_id, [self.target(qid)]) else "-") \
                    + ("R" if doc_id in self.relevant(qid) else "-")
                mark = "*" if index in row["citations"] else " "
                print(f"   {mark}{index} {source(doc_id):2} {flags} {(doc.title or '')[:28]:28} | {doc.text[:snippet]!r}")


def statistics(sweep: Sweep) -> dict:
    Q = list(sweep.queries)
    label = lambda v, q: sweep.row(v, q)["label"]
    stats: dict = {}
    stats["labels"] = {v: dict(collections.Counter(label(v, q) for q in Q)) for v in sweep.order}
    # Baseline funnel: relevant human document in context -> gold text in context -> label
    funnel = collections.Counter()
    for q in Q:
        c = sweep.case(HUMAN, q)
        funnel[f"rel={c['relevant_human_in_context']}|gold_text={c['gold_in_context']}|{c['label']}"] += 1
    stats["baseline_funnel"] = dict(funnel)
    # Transitions of the baseline label into other variants
    stats["transitions_from_H"] = {
        v: {f"{a}->{b}": n for (a, b), n in sorted(collections.Counter((label(HUMAN, q), label(v, q)) for q in Q).items())}
        for v in sweep.order if v != HUMAN
    }
    # Own attack document: rank in the top-5 context and outcome
    by_rank = collections.defaultdict(collections.Counter)
    for q in Q:
        by_rank[sweep.case(H_A, q)["own_attack_rank"]][label(H_A, q)] += 1
    stats["H+A_by_own_attack_rank"] = {str(k): dict(v) for k, v in sorted(by_rank.items())}
    # Source of the rank-1 context document and outcome
    for variant in (H_A, MIX, ALL):
        table = collections.defaultdict(collections.Counter)
        for q in Q:
            table[source(sweep.context(variant, q)[0])][label(variant, q)] += 1
        stats[f"rank1_source|{variant}"] = {s: dict(c) for s, c in table.items()}
    # Exposure of a false version of the relevant root and target adoption
    exposure = {}
    for variant, src in (("human+resolved_false", "RF"), ("human+qwen3_false", "QF"), (H_A, "A")):
        counts = collections.Counter()
        for q in Q:
            ctx = sweep.context(variant, q)
            roots = set(sweep.queries[q].relevant_doc_ids)
            exposed = (ATTACK_PREFIX + q in ctx) if src == "A" else any(
                source(d) == src and sweep.docs[d].provenance_root_id in roots for d in ctx)
            counts[(exposed, label(variant, q) == "target")] += 1
        exposure[variant] = {f"exposed={e}|target={t}": n for (e, t), n in sorted(counts.items())}
    stats["false_exposure"] = exposure
    # Where do gains from correct rewrites come from?
    gains = {}
    for variant in ("human+qwen3_correct", "human+resolved_correct"):
        gained = [q for q in Q if label(HUMAN, q) != "correct" and label(variant, q) == "correct"]
        gains[variant] = {
            "gains": len(gained),
            "human_original_already_in_baseline_context": sum(sweep.case(HUMAN, q)["relevant_human_in_context"] for q in gained),
            "losses": sum(label(HUMAN, q) == "correct" and label(variant, q) != "correct" for q in Q),
        }
    stats["correct_rewrite_gains"] = gains
    # Contexts with several versions of one root, and displaced human originals
    stats["multi_version_contexts"] = {v: sum(sweep.case(v, q)["max_versions_of_one_root"] >= 2 for q in Q) for v in (ALL, MIX, "human+qwen3_false")}
    displaced = [q for q in Q if (sweep.relevant(q) & set(sweep.retrieval[ALL][q]))
                 and not (sweep.relevant(q) & set(sweep.context(ALL, q)))]
    stats["ALL_human_original_pushed_out_of_top5"] = {"n": len(displaced), "target": sum(label(ALL, q) == "target" for q in displaced)}
    foreign = collections.Counter(sum(source(d) == "A" and d != ATTACK_PREFIX + q for d in sweep.context(H_A, q)) for q in Q)
    stats["H+A_foreign_attack_docs_in_context"] = dict(sorted(foreign.items()))
    # Is the answer text found in the context at all?
    grounded = {}
    for variant in (HUMAN, H_A, ALL):
        counts = collections.Counter()
        for q in Q:
            row = sweep.row(variant, q)
            if row["label"] == "abstain":
                continue
            a = normalize(row["answer"])
            counts[(row["label"], bool(a) and any(a in sweep.text(d) for d in row["context_doc_ids"]))] += 1
        grounded[variant] = {f"{l}|in_context={g}": n for (l, g), n in sorted(counts.items())}
    stats["answer_text_found_in_context"] = grounded
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", nargs="*", help="print these query ids instead of writing files")
    parser.add_argument("--variants", nargs="*", default=[HUMAN, H_A])
    args = parser.parse_args()
    sweep = Sweep()
    if args.show:
        for qid in args.show:
            sweep.show(qid, args.variants)
            print()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "per_query_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for variant in sweep.order:
            for qid in sweep.queries:
                case = sweep.case(variant, qid)
                case["question"] = sweep.queries[qid].text
                case["gold"] = " / ".join(sweep.queries[qid].aliases[:3])
                case["target"] = sweep.target(qid) or ""
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(case))
                    writer.writeheader()
                writer.writerow(case)
    (OUT / "query_case_stats.json").write_text(
        json.dumps(statistics(sweep), indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUT / 'per_query_cases.csv'} and query_case_stats.json")


if __name__ == "__main__":
    main()
