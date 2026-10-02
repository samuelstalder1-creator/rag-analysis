from __future__ import annotations

import argparse
import json
from pathlib import Path

from evlab.config import load_config
from evlab.data.cocktail import load_corpus
from evlab.data.nqplus import load_generation_candidates
from evlab.data.validate import validate_dataset_config
from evlab.experiments.runner import run_experiment
from evlab.generation.pipeline import run_generation_pipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evlab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data")
    data_sub = data_parser.add_subparsers(dest="data_command", required=True)
    data_validate = data_sub.add_parser("validate")
    data_validate.add_argument("--config", required=True)

    gen_parser = subparsers.add_parser("generate")
    gen_parser.add_argument("--config", required=True)
    gen_parser.add_argument("--limit", type=int, default=None)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--run-dir", default="results/runs")

    args = parser.parse_args(argv)
    if args.command == "data" and args.data_command == "validate":
        config = load_config(args.config)
        _print(validate_dataset_config(config))
    elif args.command == "generate":
        config = load_config(args.config)
        generation_config = config["generation"]
        source = str(generation_config.get("source", "candidates"))
        if source == "human_corpus":
            if "human_corpus" not in config["data"]:
                raise ValueError("generation source 'human_corpus' requires data.human_corpus in the config")
            candidates = _human_corpus_generation_candidates(config["data"]["human_corpus"])
        elif source == "candidates":
            candidates = load_generation_candidates(config["data"]["nqplus_dir"])
        else:
            raise ValueError(f"Unsupported generation source: {source}")
        _print(run_generation_pipeline(generation_config, candidates, limit=args.limit))
    elif args.command == "run":
        _print(run_experiment(args.config))
    elif args.command == "report":
        _print(_collect_reports(Path(args.run_dir)))


def _collect_reports(run_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(run_dir.glob("*/*/metrics.json")):
        rows.append({"path": str(metrics_path), "metrics": json.loads(metrics_path.read_text(encoding="utf-8"))})
    return rows


def _human_corpus_generation_candidates(path: str) -> list[dict]:
    candidates: list[dict] = []
    for doc in load_corpus(path, origin="human", namespace="human"):
        original_id = str(doc.metadata.get("_id") or doc.provenance_root_id or doc.doc_id)
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
