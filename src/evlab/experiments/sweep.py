"""Run one experiment config under several corpus/retrieval settings.

A sweep config is a normal experiment config plus ``variants`` and/or
``sweep``:

``variants``  named settings, each a mapping of dotted paths to values::

                variants:
                  human_only:
                    corpus.background.qwen_correct.ratio: 0.0
                    corpus.targeted.query_ratio: 0.0
                  false_add:
                    corpus.targeted.false.mode: add

              (a list of ``{name: ..., set: {...}}`` works too)

``sweep``     a grid; every combination of the listed values is run::

                sweep:
                  corpus.background.0.ratio: [0.0, 0.25, 0.5, 1.0]
                  corpus.background.0.mode: [add, replace]

Both together run every variant for every grid point. Numeric path segments
index lists (``corpus.background.0.ratio``). The outputs of each run go to
``output.dir`` as usual; the sweep additionally writes ``summary.csv`` and
``summary.jsonl`` (one row per run and retriever) to ``sweep_output.dir``.
"""

from __future__ import annotations

import copy
import csv
import itertools
import re
from pathlib import Path

from evlab.config import load_config
from evlab.io import stable_hash, write_json, write_jsonl


def expand_sweep(config: dict) -> list[tuple[str, dict, dict]]:
    """Return ``(label, params, run_config)`` for every variant x grid point."""
    base = {k: v for k, v in config.items() if k not in {"sweep", "variants", "sweep_output"}}
    variants = config.get("variants") or [{"name": None, "set": {}}]
    if isinstance(variants, dict):
        # Mapping form: {variant_name: {dotted.path: value, ...}}
        variants = [{"name": name, "set": dict(overrides or {})} for name, overrides in variants.items()]
    grid = config.get("sweep") or {}
    if not isinstance(grid, dict):
        raise ValueError("sweep must map dotted config paths to lists of values")
    for path, values in grid.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.{path} must be a non-empty list")
    names = [variant.get("name") for variant in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"Variant names must be unique: {names}")

    keys = list(grid)
    experiment = str(base.get("experiment", "experiment"))
    expanded = []
    for variant in variants:
        overrides = variant.get("set") or {}
        for values in itertools.product(*(grid[key] for key in keys)) if keys else [()]:
            run_config = copy.deepcopy(base)
            params: dict = {}
            if variant.get("name") is not None:
                params["variant"] = variant["name"]
            for path, value in overrides.items():
                set_path(run_config, path, copy.deepcopy(value))
            for path, value in zip(keys, values):
                set_path(run_config, path, copy.deepcopy(value))
                params[path] = value
            label = "__".join(
                [str(variant["name"])] * (variant.get("name") is not None)
                + [f"{_short(path)}={_fmt(value)}" for path, value in zip(keys, values)]
            ) or "base"
            run_config["experiment"] = f"{experiment}__{_slug(label)}"
            expanded.append((label, params, run_config))
    return expanded


def set_path(config: dict, path: str, value: object) -> None:
    parts = path.split(".")
    node: object = config
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if isinstance(node, list):
            position = int(part)
            if position >= len(node):
                raise KeyError(f"{path}: list index {position} out of range")
            if last:
                node[position] = value
            else:
                node = node[position]
        elif isinstance(node, dict):
            if last:
                node[part] = value
            else:
                if node.get(part) is None:
                    node[part] = {}
                node = node[part]
        else:
            raise KeyError(f"{path}: cannot descend into {type(node).__name__} at '{part}'")


def run_sweep(config_path: str | Path, *, dry_run: bool = False) -> dict:
    from evlab.corpus.builder import build_corpus_with_report
    from evlab.experiments.runner import _selection_summary, run_experiment

    config = load_config(config_path)
    runs = expand_sweep(config)
    sweep_name = str(config.get("experiment", Path(config_path).stem))
    out_dir = Path((config.get("sweep_output") or {}).get("dir", "results/sweeps")) / (
        f"{sweep_name}_{stable_hash(config)[:10]}"
    )
    rows: list[dict] = []
    for label, params, run_config in runs:
        if dry_run:
            docs, selection = build_corpus_with_report(run_config)
            rows.append(
                {"label": label, **params, **_flatten("corpus", _selection_summary(selection))}
            )
            continue
        summary = run_experiment(run_config)
        corpus = _flatten("corpus", summary.get("corpus", {}))
        reader = summary.get("reader") or {}
        specs = summary.get("retriever_specs", {})
        for retriever, metrics in summary["retrievers"].items():
            spec = specs.get(retriever, {})
            rows.append(
                {
                    "label": label,
                    "run_id": summary["run_id"],
                    "retriever": retriever,
                    "retriever_type": spec.get("type"),
                    "retriever_members": "+".join(spec.get("members", [])) or None,
                    "reader_model": reader.get("model") if retriever in reader.get("retrievers", []) else None,
                    **params,
                    **corpus,
                    **metrics,
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "corpus_preview" if dry_run else "summary"
    write_jsonl(out_dir / f"{name}.jsonl", rows)
    _write_csv(out_dir / f"{name}.csv", rows)
    write_json(out_dir / "sweep_config.json", config)
    return {"sweep_dir": str(out_dir), "runs": len(runs), "rows": rows}


def _flatten(prefix: str, value: object) -> dict:
    if isinstance(value, dict):
        flat: dict = {}
        for key, item in value.items():
            flat.update(_flatten(f"{prefix}.{key}", item))
        return flat
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value):
            return {prefix: value}
        flat = {}
        for index, item in enumerate(value):
            flat.update(_flatten(f"{prefix}.{index}", item))
        return flat
    return {prefix: value}


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _short(path: str) -> str:
    return path[len("corpus."):] if path.startswith("corpus.") else path


def _fmt(value: object) -> str:
    return "null" if value is None else str(value)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._=-]+", "-", text)
