from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Sequence

from evlab.config import load_config
from evlab.corpus.builder import build_corpus_with_report
from evlab.corpus.conditions import corpus_manifest
from evlab.data.nqplus import (
    load_generation_queries,
    load_qrels,
    load_rag_queries,
    load_retrieval_queries,
)
from evlab.evaluation.retrieval import evaluate_retrieval
from evlab.evaluation.source import evaluate_source_metrics
from evlab.io import stable_hash, write_json, write_jsonl
from evlab.retrieval.specs import RetrievalSession, resolve_retrievers
from evlab.schema import Doc, Hit, Query
from evlab.rag.config import reader_summary, resolve_reader
from evlab.rag.pipeline import answer_with_hits
from evlab.evaluation.answers import classify_answer
from evlab.llm.client import build_llm_client
from evlab.llm.cache import CachedLLMClient


def run_experiment(config_path: str | Path | dict) -> dict[str, object]:
    config = (
        dict(config_path) if isinstance(config_path, dict) else load_config(config_path)
    )
    if "sweep" in config or "variants" in config:
        raise ValueError(
            "This config defines a sweep/variants; run it with `evlab sweep <config>`"
        )
    # Resolve retrievers and reader first so config errors fail before any
    # corpus is loaded.
    run_names, retriever_specs = resolve_retrievers(config)
    reader_spec = resolve_reader(config, run_names)
    resolved = {"retrievers": retriever_specs, "reader": reader_spec}
    run_id = _run_id(config, resolved)
    output_root = Path(config.get("output", {}).get("dir", "results/runs")) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    reader = (
        CachedLLMClient(build_llm_client(reader_spec["llm"]), reader_spec["cache_path"])
        if reader_spec
        else None
    )
    docs, selection = build_corpus_with_report(config)
    docs_by_id = {doc.doc_id: doc for doc in docs}
    queries = _load_queries(config)
    evaluated_ids = {query.query_id for query in queries}
    # Only score the questions that were searched; otherwise unsearched
    # retrieval queries count as zero-recall misses when `queries` is a subset.
    qrels = [
        qrel
        for qrel in load_qrels(config["data"]["nqplus_dir"])
        if qrel.query_id in evaluated_ids
    ]
    targeted_ids = set((selection.get("targeted") or {}).get("query_ids", []))
    targeted_ids &= evaluated_ids
    top_k = int(config.get("retrieval", {}).get("top_k", 100))
    relevance_modes = tuple(
        config.get("evaluation", {}).get("relevance_modes", ["strict", "relaxed"])
    )
    cutoffs = tuple(
        int(item) for item in config.get("evaluation", {}).get("cutoffs", [10, 100])
    )

    run_summaries: dict[str, object] = {
        "run_id": run_id,
        "num_docs": len(docs),
        "num_queries": len(queries),
        "corpus": _selection_summary(selection),
        "retriever_specs": retriever_specs,
        "reader": reader_summary(reader_spec),
        "retrievers": {},
    }
    write_json(output_root / "config.json", config)
    write_json(output_root / "manifest.json", corpus_manifest(docs, config))
    write_json(output_root / "corpus_selection.json", selection)
    write_json(output_root / "resolved.json", resolved)

    session = RetrievalSession(retriever_specs, docs, queries)
    session.plan(run_names, top_k)
    for retriever_name in run_names:
        run = session.run(retriever_name, top_k)
        metrics: dict[str, float] = {}
        for mode in relevance_modes:
            for key, value in evaluate_retrieval(
                run,
                qrels,
                docs_by_id=docs_by_id,
                relevance_mode=str(mode),
                cutoffs=cutoffs,
            ).items():
                metrics[f"{mode}_{key}"] = value
        metrics.update(evaluate_source_metrics(run, docs_by_id, cutoffs=cutoffs))
        if targeted_ids and targeted_ids != evaluated_ids:
            # Same metrics on the attacked questions only, so a low query_ratio
            # does not dilute the effect on the questions that were attacked.
            targeted_run = {qid: hits for qid, hits in run.items() if qid in targeted_ids}
            targeted_qrels = [qrel for qrel in qrels if qrel.query_id in targeted_ids]
            for mode in relevance_modes:
                for key, value in evaluate_retrieval(
                    targeted_run,
                    targeted_qrels,
                    docs_by_id=docs_by_id,
                    relevance_mode=str(mode),
                    cutoffs=cutoffs,
                ).items():
                    metrics[f"targeted_{mode}_{key}"] = value
            for key, value in evaluate_source_metrics(
                targeted_run, docs_by_id, cutoffs=cutoffs
            ).items():
                metrics[f"targeted_{key}"] = value
        run_dir = output_root / str(retriever_name)
        write_jsonl(run_dir / "retrieval.jsonl", _run_rows(run))
        if reader is not None and retriever_name in reader_spec["retrievers"]:
            answers, answer_metrics = _evaluate_reader(
                reader, queries, run, docs_by_id, reader_spec
            )
            write_jsonl(run_dir / "answers.jsonl", answers)
            metrics.update(answer_metrics)
        write_json(run_dir / "metrics.json", metrics)
        run_summaries["retrievers"][str(retriever_name)] = metrics  # type: ignore[index]

    write_json(output_root / "summary.json", run_summaries)
    return run_summaries


def _selection_summary(selection: dict) -> dict:
    """Corpus selection without the per-question ID list."""
    summary = {key: value for key, value in selection.items() if key != "targeted"}
    if selection.get("targeted"):
        summary["targeted"] = {
            key: value
            for key, value in selection["targeted"].items()
            if key != "query_ids"
        }
    return summary


def _load_queries(config: dict) -> list[Query]:
    mode = str(config.get("queries", "retrieval"))
    nqplus_dir = config["data"]["nqplus_dir"]
    if mode == "retrieval":
        return load_retrieval_queries(nqplus_dir)
    if mode == "rag":
        return load_rag_queries(nqplus_dir)
    if mode == "generation":
        return load_generation_queries(nqplus_dir)
    raise ValueError(f"Unsupported query set: {mode}")


def _run_rows(run: dict[str, list[Hit]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for query_id, hits in sorted(run.items()):
        for hit in hits:
            rows.append(
                {
                    "query_id": query_id,
                    "doc_id": hit.doc_id,
                    "rank": hit.rank,
                    "score": hit.score,
                    "retriever": hit.retriever,
                }
            )
    return rows


def _run_id(config: dict, resolved: dict | None = None) -> str:
    experiment = str(config.get("experiment", "experiment"))
    payload: object = config
    if resolved and _references_files(config):
        # Retriever/reader/prompt files can change without the config changing;
        # include their resolved content so such runs get a new ID.
        payload = {"config": config, "resolved": resolved}
    return f"{experiment}_{stable_hash(payload)[:10]}"


def _references_files(config: dict) -> bool:
    reader = config.get("reader")
    if isinstance(reader, str):
        return True
    if isinstance(reader, dict) and (
        isinstance(reader.get("llm"), str) or reader.get("prompt") is not None
    ):
        return True
    definitions = dict(config.get("retriever_definitions") or {})
    if isinstance(config.get("retrievers"), dict):
        definitions.update(config["retrievers"])
    return any(isinstance(value, str) for value in definitions.values())


def _evaluate_reader(client, queries, run, docs_by_id, config):
    targets = {}
    for doc in docs_by_id.values():
        if doc.condition == "false":
            if not doc.target_answer:
                raise ValueError(
                    f"False document {doc.doc_id} has no validated target answer"
                )
            previous = targets.setdefault(doc.query_id, doc.target_answer)
            if previous != doc.target_answer:
                raise ValueError(f"Conflicting false targets for {doc.query_id}")
    rows = []
    counts = Counter()
    for query in queries:
        result = answer_with_hits(
            client,
            query=query,
            hits=run.get(query.query_id, []),
            docs_by_id=docs_by_id,
            top_k=int(config["top_k"]),
            temperature=float(config["temperature"]),
            seed=config["seed"],
            prompt=config["prompt_template"],
        )
        label = (
            classify_answer(
                result["answer"],
                gold_aliases=list(query.aliases),
                target_answer=targets.get(query.query_id),
            )
            if result["valid_json"]
            else "invalid"
        )
        doc_ids = result["context_doc_ids"]
        valid_citations = [i for i in result["citations"] if 1 <= i <= len(doc_ids)]
        result.update(
            label=label,
            gold_aliases=list(query.aliases),
            target_answer=targets.get(query.query_id),
            cited_doc_ids=[doc_ids[i - 1] for i in valid_citations],
            citations_valid=len(valid_citations) == len(result["citations"]),
        )
        counts[label] += 1
        rows.append(result)
    size = len(rows)
    metrics = {
        f"answer_{label}_rate": counts[label] / size if size else 0.0
        for label in ("correct", "target", "abstain", "other", "ambiguous", "invalid")
    }
    targeted = [row for row in rows if row["target_answer"] is not None]
    metrics["answer_target_query_count"] = len(targeted)
    metrics["answer_target_rate_on_targeted_queries"] = (
        sum(row["label"] == "target" for row in targeted) / len(targeted)
        if targeted
        else 0.0
    )
    return rows, metrics
