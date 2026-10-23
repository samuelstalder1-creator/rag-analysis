from __future__ import annotations

import argparse
import json
from pathlib import Path

from evlab.config import load_config
from evlab.data.cocktail import load_corpus
from evlab.data.nqplus import load_generation_candidates
from evlab.data.answer_classification import split_answer_flags
from evlab.data.validate import validate_dataset_config
from evlab.experiments.runner import run_experiment
from evlab.generation.pipeline import run_generation_pipeline
from evlab.generation.false_answers import run_false_answer_generation
from evlab.generation.false_documents import run_false_document_generation
from evlab.generation.maintenance import cleanup_false_generation, generation_status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evlab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)
    data_validate = data_sub.add_parser("validate")
    data_validate.add_argument("--config", required=True)
    data_classify = data_sub.add_parser("classify-answers")
    data_classify.add_argument("--config", required=True)
    data_build = data_sub.add_parser("build-nqopen-plus")
    data_build.add_argument("--config", required=True)

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--config", required=True)
    gen_parser.add_argument("--limit", type=int, default=None)
    for command in ("generate-false-answers", "generate-false-documents"):
        stage_parser = subparsers.add_parser(command)
        stage_parser.add_argument("--config", required=True)
        stage_parser.add_argument("--limit", type=int, default=None)
        if command == "generate-false-documents":
            stage_parser.add_argument("--dry-run", action="store_true")
            stage_parser.add_argument("--retry-seed", type=int, default=0,
                                      help="fresh rewrite sampling for missing documents; keeps completed rows")
    for command in ("generation-status", "cleanup-false"):
        maintenance_parser = subparsers.add_parser(command)
        maintenance_parser.add_argument("--config", required=True)

    for command in ("repair-correct-documents", "generate-correct-documents"):
        correct_parser = subparsers.add_parser(command, help="generate checked correct document copies")
        correct_parser.add_argument("--config", required=True)
        correct_parser.add_argument("--limit", type=int, default=None)
        correct_parser.add_argument("--dry-run", action="store_true")
        correct_parser.add_argument("--retry-seed", type=int, default=0,
                                    help="fresh sampling for missing correct documents; keeps completed rows")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config")
    run_parser.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N questions (smoke test)"
    )

    sweep_parser = subparsers.add_parser(
        "sweep", help="run every variant/grid point of a sweep config"
    )
    sweep_parser.add_argument("config")
    sweep_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only build each corpus and report its composition (no retrieval)",
    )
    sweep_parser.add_argument(
        "--limit", type=int, default=None, help="evaluate only the first N questions per run (smoke test)"
    )

    corpus_parser = subparsers.add_parser(
        "corpus", help="build an experiment's corpus and print its composition"
    )
    corpus_parser.add_argument("config")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-dir", default="results/runs")

    args = parser.parse_args(argv)
    if args.command == "data" and args.data_command == "validate":
        config = load_config(args.config)
        _print(validate_dataset_config(config))
    elif args.command == "data" and args.data_command == "classify-answers":
        config = load_config(args.config)
        _print(
            split_answer_flags(
                config["data"]["nqplus_dir"],
                human_corpus=config["data"]["human_corpus"],
                reviews_file=config["answer_classification"]["reviews_file"],
            )
        )
    elif args.command == "data" and args.data_command == "build-nqopen-plus":
        from evlab.data.nqopen_plus import build_nqopen_plus
        _print(build_nqopen_plus(load_config(args.config)))
    elif args.command in {"generation-status", "cleanup-false"}:
        config = load_config(args.config)
        operation = (
            generation_status
            if args.command == "generation-status"
            else cleanup_false_generation
        )
        _print(operation(config))
    elif args.command in {"generate-false-answers", "generate-false-documents"}:
        config = load_config(args.config)
        stage = (
            run_false_answer_generation
            if args.command == "generate-false-answers"
            else run_false_document_generation
        )
        kwargs = {"limit": args.limit}
        if args.command == "generate-false-documents":
            kwargs["dry_run"] = args.dry_run
            kwargs["retry_seed"] = args.retry_seed
        kwargs["human_corpus"] = config["data"].get("human_corpus")
        _print(stage(config["generation"], config["data"]["nqplus_dir"], **kwargs))
    elif args.command in {"repair-correct-documents", "generate-correct-documents"}:
        from evlab.generation.correct_repair import run_correct_repair, run_correct_generation
        stage = run_correct_generation if args.command == "generate-correct-documents" else run_correct_repair
        _print(stage(load_config(args.config), limit=args.limit, dry_run=args.dry_run,
                     retry_seed=args.retry_seed))
    elif args.command == "generate":
        config = load_config(args.config)
        generation_config = config["generation"]
        source = str(generation_config.get("source", "candidates"))
        if source == "human_corpus":
            if "human_corpus" not in config["data"]:
                raise ValueError(
                    "generation source 'human_corpus' requires data.human_corpus in the config"
                )
            candidates = _human_corpus_generation_candidates(
                config["data"]["human_corpus"]
            )
        elif source == "candidates":
            candidates = load_generation_candidates(
                config["data"]["nqplus_dir"], config["data"].get("human_corpus")
            )
        else:
            raise ValueError(f"Unsupported generation source: {source}")
        from evlab.generation.false_answers import attach_saved_answers

        candidates = attach_saved_answers(candidates, generation_config)
        _print(run_generation_pipeline(generation_config, candidates, limit=args.limit))
    elif args.command == "run":
        if args.limit:
            config = load_config(args.config)
            config["query_limit"] = args.limit
            _print(run_experiment(config))
        else:
            _print(run_experiment(args.config))
    elif args.command == "sweep":
        from evlab.experiments.sweep import run_sweep

        result = run_sweep(args.config, dry_run=args.dry_run, limit=args.limit)
        _print({"sweep_dir": result["sweep_dir"], "runs": result["runs"]})
    elif args.command == "corpus":
        from evlab.corpus.builder import build_corpus_with_report
        from evlab.corpus.conditions import corpus_manifest
        from evlab.experiments.runner import _selection_summary

        config = load_config(args.config)
        docs, selection = build_corpus_with_report(config)
        _print(
            {
                "manifest": corpus_manifest(docs, config),
                "selection": _selection_summary(selection),
            }
        )
    elif args.command == "report":
        _print(_collect_reports(Path(args.run_dir)))


def _collect_reports(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(run_dir.glob("*/*/metrics.json")):
        rows.append(
            {
                "path": str(metrics_path),
                "metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
            }
        )
    return rows


def _human_corpus_generation_candidates(path: str) -> list[dict]:
    candidates: list[dict] = []
    for doc in load_corpus(path, origin="human", namespace="human"):
        original_id = str(
            doc.metadata.get("_id") or doc.provenance_root_id or doc.doc_id
        )
        candidates.append(
            {
                "query_id": original_id,
                "query": doc.title or original_id,
                "answer": "",
                "answer_aliases": [],
                "h_docs": [
                    {
                        "doc_id": original_id,
                        "title": doc.title,
                        "text": doc.text,
                        "aliases_in_doc": [],
                    }
                ],
            }
        )
    return candidates


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
