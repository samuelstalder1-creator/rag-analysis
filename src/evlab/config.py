from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        return tomllib.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError:
            return _parse_small_yaml(text)
        loaded = yaml.safe_load(text)
        return loaded or {}
    raise ValueError(f"Unsupported config format: {config_path}")


def _parse_small_yaml(text: str) -> dict[str, Any]:
    """Tiny YAML subset parser for this repo's simple configs.

    It supports nested dictionaries, lists, strings, booleans, nulls and inline
    scalar lists like [1, 5, 10]. Use PyYAML for anything more expressive.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    last_key_by_indent: dict[int, str] = {}

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            value = _parse_scalar(line[2:].strip())
            if not isinstance(parent, list):
                raise ValueError("List item without list parent. Install PyYAML for full YAML support.")
            parent.append(value)
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported YAML line: {raw_line!r}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            value = _parse_scalar(raw_value)
        else:
            value = {}
        if isinstance(parent, dict):
            parent[key] = value
        else:
            raise ValueError("Nested mapping inside list is not supported by the fallback parser.")
        last_key_by_indent[indent] = key
        if raw_value == "":
            stack.append((indent, value))
        elif raw_value == "[]":
            parent[key] = []
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
