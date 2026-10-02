from __future__ import annotations

from pathlib import Path
from typing import Sequence

from evlab.config import load_config
from evlab.corpus.builder import build_corpus
from evlab.corpus.conditions import corpus_manifest
from evlab.data.nqplus import load_generation_candidates, load_qrels, load_rag_queries, load_retrieval_queries
from evlab.evaluation.retrieval import evaluate_retrieval
from evlab.evaluation.source import evaluate_source_metrics
from evlab.io import stable_hash, write_json, write_jsonl
from evlab.retrieval.bm25 import BM25Retriever
from evlab.retrieval.dense import HashDenseRetriever, SentenceTransformerRetriever
from evlab.retrieval.tfidf import TfidfRetriever
from evlab.schema import Doc, Hit, Query


def run_experiment(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    run_id = _run_id(config)
    output_root = Path(config.get("output", {}).get("dir", "results/runs")) / run_id
    output_root.mkdir(parents=True, exist_ok=True)

    docs = build_corpus(config)
    docs_by_id = {doc.doc_id: doc for doc in docs}
    queries = _load_queries(config)
    qrels = load_qrels(config["data"]["nqplus_dir"])
    top_k = int(config.get("retrieval", {}).get("top_k", 100))
    relevance_modes = tuple(config.get("evaluation", {}).get("relevance_modes", ["strict", "relaxed"]))
    cutoffs = tuple(int(item) for item in config.get("evaluation", {}).get("cutoffs", [10, 100]))

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
        wanted = {str(row["query_id"]) for row in load_generation_candidates(nqplus_dir)}
        return [query for query in load_retrieval_queries(nqplus_dir) if query.query_id in wanted]
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
