from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import Sequence

from evlab.config import load_config
from evlab.corpus.builder import build_corpus
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
from evlab.retrieval.bm25 import BM25Retriever
from evlab.retrieval.dense import HashDenseRetriever, SentenceTransformerRetriever
from evlab.retrieval.tfidf import TfidfRetriever
from evlab.schema import Doc, Hit, Query
from evlab.rag.pipeline import answer_with_hits
from evlab.evaluation.answers import classify_answer
from evlab.llm.client import build_llm_client
from evlab.llm.cache import CachedLLMClient


def run_experiment(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    run_id = _run_id(config)
    output_root = Path(config.get("output", {}).get("dir", "results/runs")) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    reader_config = config.get("reader", {})
    reader_enabled = reader_config.get("enabled", config.get("queries") == "rag")
    if reader_enabled and "llm" not in reader_config:
        raise ValueError("RAG evaluation requires reader.llm in the experiment config")
    reader = (
        CachedLLMClient(
            build_llm_client(reader_config["llm"]),
            reader_config.get("cache_path", "cache/reader.sqlite"),
        )
        if reader_enabled
        else None
    )
    docs = build_corpus(config)
    docs_by_id = {doc.doc_id: doc for doc in docs}
    queries = _load_queries(config)
    qrels = load_qrels(config["data"]["nqplus_dir"])
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
        "retrievers": {},
    }
    write_json(output_root / "config.json", config)
    write_json(output_root / "manifest.json", corpus_manifest(docs, config))

    for retriever_name in config.get("retrievers", ["bm25"]):
        retriever = _build_retriever(str(retriever_name), config)
        retriever.index(docs)
        run = retriever.search(queries, top_k=top_k)
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
        run_dir = output_root / str(retriever_name)
        write_jsonl(run_dir / "retrieval.jsonl", _run_rows(run))
        if reader is not None:
            answers, answer_metrics = _evaluate_reader(
                reader, queries, run, docs_by_id, reader_config
            )
            write_jsonl(run_dir / "answers.jsonl", answers)
            metrics.update(answer_metrics)
        write_json(run_dir / "metrics.json", metrics)
        run_summaries["retrievers"][str(retriever_name)] = metrics  # type: ignore[index]

    write_json(output_root / "summary.json", run_summaries)
    return run_summaries


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


def _build_retriever(name: str, config: dict):
    retrieval = config.get("retrieval", {})
    if name == "bm25":
        return BM25Retriever(
            k1=float(retrieval.get("k1", 1.5)),
            b=float(retrieval.get("b", 0.75)),
            remove_query_stopwords=bool(retrieval.get("remove_query_stopwords", True)),
        )
    if name == "tfidf":
        return TfidfRetriever()
    if name == "hash_dense":
        return HashDenseRetriever()
    if name == "minilm":
        return SentenceTransformerRetriever(
            str(retrieval.get("dense_model", "sentence-transformers/all-MiniLM-L6-v2")),
            batch_size=int(retrieval.get("dense_batch_size", 128)),
            search_batch_size=int(retrieval.get("dense_search_batch_size", 16)),
            local_files_only=bool(retrieval.get("dense_local_files_only", True)),
            cache_dir=str(retrieval.get("dense_cache_dir", "cache/embeddings")),
            device=retrieval.get("dense_device"),
            score_device=retrieval.get("dense_score_device"),
            max_seq_length=_optional_int(retrieval.get("dense_max_seq_length")),
        )
    if name == "e5":
        return SentenceTransformerRetriever(
            str(retrieval.get("dense_model", "intfloat/e5-base-v2")),
            batch_size=int(retrieval.get("dense_batch_size", 128)),
            search_batch_size=int(retrieval.get("dense_search_batch_size", 16)),
            local_files_only=bool(retrieval.get("dense_local_files_only", True)),
            cache_dir=str(retrieval.get("dense_cache_dir", "cache/embeddings")),
            device=retrieval.get("dense_device"),
            score_device=retrieval.get("dense_score_device"),
            query_prefix="query: ",
            doc_prefix="passage: ",
            max_seq_length=_optional_int(retrieval.get("dense_max_seq_length")),
        )
    raise ValueError(f"Unsupported retriever: {name}")


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


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


def _run_id(config: dict) -> str:
    experiment = str(config.get("experiment", "experiment"))
    return f"{experiment}_{stable_hash(config)[:10]}"


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
            top_k=int(config.get("top_k", 10)),
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
