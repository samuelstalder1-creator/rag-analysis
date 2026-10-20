from __future__ import annotations

import re
from pathlib import Path

from evlab.io import append_jsonl, iter_jsonl, write_jsonl
from evlab.generation.registry import update_registry


CORPUS_FILENAME = re.compile(
    r"^(?P<model>.+)--(?P<condition>correct|false)--tmp(?P<temperature>[\d.eE+-]+)"
    r"--s(?P<sample>\d+)--d(?P<depth>\d+)--(?P<source>candidates|human_corpus)--v1\.jsonl$"
)


def output_label(model: str, temperature: float, source: str) -> str:
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    return f"{model_slug}--tmp{temperature:g}--{source}--v1"


class GenerationOutput:
    """One corpus per model, condition, temperature, sample and source."""

    def __init__(self, output_dir: Path, config: dict, model: str, conditions: set[str]):
        self.model = model
        self.output_dir = output_dir
        self.source = str(config.get("source", "candidates"))
        temperature = float(config.get("temperature", 0.3))
        self.label = output_label(model, temperature, self.source)
        migrate_compact_files(output_dir, model=model)
        def allocate(registry: dict) -> dict:
            files = registry["files"]
            paths = {}
            for condition in sorted(conditions):
                for sample in range(int(config.get("samples_per_h", 1))):
                    metadata = {"model": model, "condition": condition, "temperature": temperature,
                                "sample_idx": sample, "generation_depth": 1,
                                "source": self.source, "prompt_version": "v1"}
                    name = next((name for name, entry in files.items()
                                 if all(entry.get(k) == v for k, v in metadata.items())), None)
                    if name is None:
                        name = _next_filename(output_dir, files)
                        files[name] = {**metadata, "document_count": 0}
                    paths[(condition, sample)] = output_dir / name
            return paths
        self.paths = update_registry(output_dir, allocate)
        resume = bool(config.get("resume", True))
        self.ids: dict[tuple[str, int], set[str]] = {}
        for key, path in self.paths.items():
            if not resume or not path.exists():
                write_jsonl(path, [])
            self.ids[key] = {str(row["_id"]) for row in iter_jsonl(path)}
        if resume:
            self._import_legacy(output_dir)
        self.update_counts()

    def update_counts(self) -> None:
        def change(registry: dict) -> None:
            for key, path in self.paths.items():
                registry["files"][path.name]["document_count"] = len(self.ids[key])
        update_registry(self.output_dir, change)

    def contains(self, condition: str, sample: int, doc_id: str) -> bool:
        return doc_id in self.ids[(condition, sample)]

    def append(self, row: dict) -> int:
        key = (str(row["condition"]), int(row["sample_idx"]))
        doc_id = str(row["provenance_root_id"])
        if doc_id in self.ids[key]:
            return 0
        append_jsonl(self.paths[key], [{"_id": doc_id, "title": row["title"], "text": row["text"]}])
        self.ids[key].add(doc_id)
        return 1

    def _import_legacy(self, output_dir: Path) -> None:
        # Keep previous runs resumable without changing their original files.
        for name in ("generated_docs.jsonl", "correct_docs.jsonl", "false_docs.jsonl"):
            path = output_dir / name
            if not path.exists():
                continue
            for row in iter_jsonl(path):
                if row.get("generator_model") != self.model:
                    continue
                source = "human_corpus" if row.get("query_id") == row.get("provenance_root_id") else "candidates"
                key = (str(row.get("condition")), int(row.get("sample_idx", 0)))
                if source == self.source and key in self.paths:
                    self.append(row)


def _next_filename(output_dir: Path, files: dict) -> str:
    index = 1
    while True:
        name = f"corpus_{index:04d}.jsonl"
        if name not in files and not (output_dir / name).exists():
            return name
        index += 1


def migrate_compact_files(output_dir: Path, *, model: str) -> dict[str, str]:
    """Rename the previous compact format while preserving document bytes."""
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    def change(registry: dict) -> dict[str, str]:
        renamed = {}
        for path in sorted(output_dir.glob("*.jsonl")):
            match = CORPUS_FILENAME.fullmatch(path.name)
            if match is None or match["model"] != model_slug:
                continue
            name = _next_filename(output_dir, registry["files"])
            count = sum(1 for _ in iter_jsonl(path))
            registry["files"][name] = {
                "model": model, "condition": match["condition"],
                "temperature": float(match["temperature"]), "sample_idx": int(match["sample"]),
                "generation_depth": int(match["depth"]), "source": match["source"],
                "prompt_version": "v1", "document_count": count,
                "id_namespace": path.stem,
            }
            path.rename(output_dir / name)
            renamed[path.name] = name
        return renamed
    return update_registry(output_dir, change)
