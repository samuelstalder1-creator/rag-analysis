from __future__ import annotations

import re
import fcntl
import json
from pathlib import Path

from evlab.io import append_jsonl, iter_jsonl, write_jsonl
from evlab.generation.registry import update_registry
from evlab.generation.manifest import (
    read_manifest,
    write_manifest,
    checked_receipt,
    manifest_path,
    append_receipt,
    read_manifest_delta,
)
from evlab.io import stable_hash


CORPUS_FILENAME = re.compile(
    r"^(?P<model>.+)--(?P<condition>correct|false)--tmp(?P<temperature>[\d.eE+-]+)"
    r"--s(?P<sample>\d+)--d(?P<depth>\d+)--(?P<source>candidates|human_corpus)--v1\.jsonl$"
)


def output_label(model: str, temperature: float, source: str) -> str:
    model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
    return f"{model_slug}--tmp{temperature:g}--{source}--v1"


class GenerationOutput:
    """One corpus per model, condition, temperature, sample and source."""

    def __init__(
        self, output_dir: Path, config: dict, model: str, conditions: set[str]
    ):
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
                    metadata = {
                        "model": model,
                        "condition": condition,
                        "temperature": temperature,
                        "sample_idx": sample,
                        "generation_depth": 1,
                        "source": self.source,
                        "prompt_version": "v1",
                    }
                    if config.get("generation_signature"):
                        metadata["generation_signature"] = config[
                            "generation_signature"
                        ]
                    name = next(
                        (
                            name
                            for name, entry in files.items()
                            if all(entry.get(k) == v for k, v in metadata.items())
                        ),
                        None,
                    )
                    if name is None:
                        name = _next_filename(output_dir, files)
                        files[name] = {**metadata, "document_count": 0}
                    paths[(condition, sample)] = output_dir / name
            return paths

        self.paths = update_registry(output_dir, allocate)
        resume = bool(config.get("resume", True))
        self.ids: dict[tuple[str, int], set[str]] = {}
        self.manifests = {}
        self.rows = {}
        self.offsets = {}
        self.manifest_offsets = {}
        for key, path in self.paths.items():
            with path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not resume or not path.exists():
                    write_jsonl(path, [])
                loaded = list(iter_jsonl(path))
                self.rows[key] = {str(row["_id"]): row for row in loaded}
                if len(self.rows[key]) != len(loaded):
                    raise ValueError(f"Duplicate document IDs in {path}")
                self.ids[key] = set(self.rows[key])
                self.manifests[key] = read_manifest(path) if resume else {}
                if not resume:
                    write_manifest(path, {})
                self.offsets[key] = path.stat().st_size
                self.manifest_offsets[key] = (
                    manifest_path(path).stat().st_size
                    if manifest_path(path).exists()
                    else 0
                )
        if resume and not config.get("generation_signature"):
            self._import_legacy(output_dir)
        self.update_counts()

    def update_counts(self) -> None:
        def change(registry: dict) -> None:
            for key, path in self.paths.items():
                with path.with_suffix(".lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    self._refresh(key)
                    registry["files"][path.name]["document_count"] = len(self.ids[key])

        update_registry(self.output_dir, change)

    def contains(
        self, condition: str, sample: int, doc_id: str, fingerprint: str | None = None
    ) -> bool:
        key = (condition, sample)
        if doc_id not in self.ids[key]:
            return False
        if fingerprint is not None:
            receipt = checked_receipt(self.rows[key][doc_id], self.manifests[key])
            if receipt.get("fingerprint") != fingerprint:
                raise ValueError(
                    f"Stale generated document {doc_id}; use a new output directory or resume: false"
                )
        return True

    def previous_texts(self, condition: str, doc_id: str) -> list[str]:
        return [
            rows[doc_id]["text"]
            for (kind, _), rows in self.rows.items()
            if kind == condition and doc_id in rows
        ]

    def append(self, row: dict) -> int:
        key = (str(row["condition"]), int(row["sample_idx"]))
        doc_id = str(row["provenance_root_id"])
        path = self.paths[key]
        # Corpus appends, manifest checkpoints and duplicate checks share a lock.
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._refresh(key)
            if doc_id in self.ids[key]:
                if row.get("validation_receipt"):
                    self.contains(
                        key[0], key[1], doc_id, row["validation_receipt"]["fingerprint"]
                    )
                return 0
            compact = {"_id": doc_id, "title": row["title"], "text": row["text"]}
            manifest = self.manifests[key]
            if row.get("validation_receipt"):
                manifest[doc_id] = {
                    **row["validation_receipt"],
                    "row_hash": stable_hash(compact),
                }
                # A crash here leaves an unused receipt, never an unvalidated corpus row.
                self.manifest_offsets[key] = append_receipt(
                    path, doc_id, manifest[doc_id]
                )
            append_jsonl(path, [compact])
            self.offsets[key] = path.stat().st_size
            self.manifests[key] = manifest
            self.rows[key][doc_id] = compact
            self.ids[key].add(doc_id)
            return 1

    def _refresh(self, key) -> None:
        """Read only rows added since this writer last held the file lock."""
        path = self.paths[key]
        if path.stat().st_size < self.offsets[key]:
            raise ValueError(f"Corpus was truncated by another writer: {path}")
        with path.open(encoding="utf-8") as handle:
            handle.seek(self.offsets[key])
            for line in handle.read().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.rows[key][str(row["_id"])] = row
                    self.ids[key].add(str(row["_id"]))
            self.offsets[key] = handle.tell()
        additions, offset = read_manifest_delta(path, self.manifest_offsets[key])
        self.manifests[key].update(additions)
        self.manifest_offsets[key] = offset

    def _import_legacy(self, output_dir: Path) -> None:
        # Keep previous runs resumable without changing their original files.
        for name in ("generated_docs.jsonl", "correct_docs.jsonl", "false_docs.jsonl"):
            path = output_dir / name
            if not path.exists():
                continue
            for row in iter_jsonl(path):
                if row.get("condition") == "false":
                    continue
                if row.get("generator_model") != self.model:
                    continue
                source = (
                    "human_corpus"
                    if row.get("query_id") == row.get("provenance_root_id")
                    else "candidates"
                )
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
                "model": model,
                "condition": match["condition"],
                "temperature": float(match["temperature"]),
                "sample_idx": int(match["sample"]),
                "generation_depth": int(match["depth"]),
                "source": match["source"],
                "prompt_version": "v1",
                "document_count": count,
                "id_namespace": path.stem,
            }
            sidecar = manifest_path(path)
            path.rename(output_dir / name)
            if sidecar.exists():
                sidecar.rename(manifest_path(output_dir / name))
            renamed[path.name] = name
        return renamed

    return update_registry(output_dir, change)
