"""Shared file metadata, stored once per generation output directory."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Callable, TypeVar

from evlab.config import load_config


REGISTRY_NAME = "metadata.yaml"
T = TypeVar("T")


def read_registry(output_dir: Path) -> dict:
    path = output_dir / REGISTRY_NAME
    registry = load_config(path) if path.exists() else {"version": 1, "files": {}}
    if registry.get("files") is None:
        registry["files"] = {}
    if registry.get("version") != 1 or not isinstance(registry.get("files"), dict):
        raise ValueError(f"Invalid generation registry: {path}")
    return registry


def update_registry(output_dir: Path, change: Callable[[dict], T]) -> T:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Correct and false generation can run in separate processes.
    with (output_dir / ".metadata.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = read_registry(output_dir)
        result = change(registry)
        for name in registry["files"]:
            if Path(name).name != name:
                raise ValueError(f"Registry filenames must be local: {name}")
        lines = _yaml_lines(registry)
        temporary = output_dir / ".metadata.yaml.tmp"
        temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary.replace(output_dir / REGISTRY_NAME)
        return result


def _yaml_lines(mapping: dict, indent: int = 0) -> list[str]:
    lines = []
    for key, value in mapping.items():
        prefix = " " * indent + str(key) + ":"
        if isinstance(value, dict):
            lines.append(prefix)
            lines.extend(_yaml_lines(value, indent + 2))
        else:
            lines.append(prefix + " " + json.dumps(value, ensure_ascii=False))
    return lines
