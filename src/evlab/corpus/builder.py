from __future__ import annotations

from pathlib import Path
from glob import glob

from evlab.data.cocktail import load_corpus
from evlab.io import iter_jsonl, stable_hash
from evlab.data.nqplus import load_qrels, load_retrieval_queries
from evlab.generation.output import CORPUS_FILENAME
from evlab.generation.registry import read_registry
from evlab.generation.manifest import read_manifest, checked_receipt
from evlab.schema import Doc


def load_generated_docs(
    path: str | Path, *, query_ids_by_doc: dict[str, str] | None = None
) -> list[Doc]:
    candidate = Path(path)
    if candidate.is_dir():
        entries = read_registry(candidate)["files"]
        paths = sorted(candidate / name for name in entries)
        paths.extend(
            sorted(
                p
                for p in candidate.glob("*.jsonl")
                if CORPUS_FILENAME.fullmatch(p.name)
            )
        )
    else:
        paths = [Path(p) for p in sorted(glob(str(path)))]
    if not paths:
        raise FileNotFoundError(f"No generated corpus files found: {path}")
    docs = []
    for file in paths:
        docs.extend(_load_generated_file(file, query_ids_by_doc or {}))
    return docs


def _load_generated_file(path: Path, query_ids_by_doc: dict[str, str]) -> list[Doc]:
    docs: list[Doc] = []
    file_metadata = read_registry(path.parent)["files"].get(path.name)
    legacy_filename = CORPUS_FILENAME.fullmatch(path.name)
    if file_metadata is None and legacy_filename is not None:
        file_metadata = {
            "model": legacy_filename["model"],
            "condition": legacy_filename["condition"],
            "generation_depth": int(legacy_filename["depth"]),
            "sample_idx": int(legacy_filename["sample"]),
            "prompt_version": "v1",
        }
    manifest = read_manifest(path)
    for row in iter_jsonl(path):
        if "_id" in row:
            if file_metadata is None:
                raise ValueError(
                    f"Compact generated corpus is missing from metadata.yaml: {path}"
                )
            root = str(row["_id"])
            condition = file_metadata["condition"]
            receipt = (
                checked_receipt(row, manifest)
                if file_metadata.get("generation_signature") or condition == "false"
                else {}
            )
            mapped_query = query_ids_by_doc.get(root, root)
            if (
                file_metadata.get("source") != "human_corpus"
                and receipt.get("query_id")
                and mapped_query != root
                and receipt["query_id"] != mapped_query
            ):
                raise ValueError(f"Generated document {root} has a stale query mapping")
            docs.append(
                Doc(
                    doc_id=f"gen::{file_metadata.get('id_namespace', path.stem)}::{root}",
                    title=str(row.get("title", "") or ""),
                    text=str(row["text"]),
                    origin="generated",
                    query_id=(
                        mapped_query
                        if file_metadata.get("source") == "human_corpus"
                        else receipt.get("query_id", mapped_query)
                    ),
                    condition=condition,
                    gold_answer=receipt.get("gold_answer"),
                    target_answer=receipt.get("target_answer"),
                    metadata={"validation_receipt": receipt},
                    parent_id=root if condition == "correct" else f"{root}*",
                    seed_id=root if condition == "correct" else f"{root}*",
                    provenance_root_id=root,
                    generation_depth=int(file_metadata["generation_depth"]),
                    generator_model=file_metadata["model"],
                    prompt_version=file_metadata["prompt_version"],
                    sample_idx=int(file_metadata["sample_idx"]),
                )
            )
            continue
        if row.get("condition") == "false":
            raise ValueError(
                f"Unverified legacy false document in {path}; regenerate with validation manifests"
            )
        docs.append(
            Doc(
                doc_id=str(row["doc_id"]),
                title=str(row.get("title", "") or ""),
                text=str(row["text"]),
                origin="generated",
                query_id=_optional_str(row.get("query_id")),
                condition=row.get("condition"),  # type: ignore[arg-type]
                gold_answer=_optional_str(row.get("gold_answer")),
                target_answer=_optional_str(row.get("target_answer")),
                parent_id=_optional_str(row.get("parent_id")),
                seed_id=_optional_str(row.get("seed_id")),
                provenance_root_id=_optional_str(row.get("provenance_root_id")),
                generation_depth=int(row.get("generation_depth", 1)),
                generator_model=_optional_str(row.get("generator_model")),
                prompt_version=_optional_str(row.get("prompt_version")),
                sample_idx=(
                    int(row["sample_idx"])
                    if row.get("sample_idx") is not None
                    else None
                ),
                metadata=dict(row),
            )
        )
    return docs


def build_corpus(config: dict) -> list[Doc]:
    return build_corpus_with_report(config)[0]


def build_corpus_with_report(config: dict) -> tuple[list[Doc], dict]:
    """Build the run corpus and describe what entered it.

    Configs with ``corpus.sources`` use the ratio-based mixer
    (``evlab.corpus.mix``); older ``base``/``generated_docs``/``k_*`` configs keep
    their exact previous behaviour.
    """
    from evlab.corpus.mix import build_mixed_corpus, is_mix_config

    corpus = config.get("corpus", {}) or {}
    if corpus.get("fixed") is not None:
        return _build_fixed_corpus(config)
    if is_mix_config(corpus):
        return build_mixed_corpus(config)
    return _build_legacy_corpus(config), {"mode": "legacy"}


def query_ids_by_doc(data: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for qrel in load_qrels(data["nqplus_dir"]):
        if qrel.relevance > 0:
            previous = mapping.setdefault(qrel.doc_id, qrel.query_id)
            if previous != qrel.query_id:
                raise ValueError(
                    f"Generated document {qrel.doc_id} belongs to multiple questions"
                )
    return mapping


def verify_generated_docs(generated: list[Doc], human_docs: list[Doc], data: dict) -> None:
    """Reject generated docs whose source evidence or question annotations changed."""
    originals = {doc.root_id: doc for doc in human_docs if doc.origin == "human"}
    queries = {
        query.query_id: query for query in load_retrieval_queries(data["nqplus_dir"])
    }
    for doc in generated:
        receipt = doc.metadata.get("validation_receipt", {})
        source = originals.get(doc.root_id)
        if receipt.get("source_content_hash"):
            expected = (
                stable_hash(
                    {"doc_id": doc.root_id, "title": source.title, "text": source.text}
                )
                if source
                else None
            )
            if receipt["source_content_hash"] != expected:
                raise ValueError(f"Stale source evidence for {doc.doc_id}")
        if receipt.get("question_hash"):
            query = queries.get(doc.query_id)
            expected = (
                stable_hash(
                    {
                        "question": query.text,
                        "gold": query.answer,
                        "aliases": list(query.aliases),
                    }
                )
                if query
                else None
            )
            if receipt["question_hash"] != expected:
                raise ValueError(f"Stale question or gold annotations for {doc.doc_id}")


def _build_fixed_corpus(config: dict) -> tuple[list[Doc], dict]:
    """Human corpus plus complete generated files, added as they are.

    ``corpus.fixed`` maps ``correct`` (LLM rewrites) and ``poisoned`` (false documents;
    not named ``false`` because YAML reads that key as a boolean) to
    ``{path, manifest?, enabled?}``. Every document of an enabled file is added next to
    its human original; nothing is sampled. A document's root is the human document it
    was generated from, so relaxed relevance counts any version as the relevant one.
    """
    data = config["data"]
    docs = load_corpus(data["human_corpus"], origin="human", namespace="human")
    owners = query_ids_by_doc(data)
    report: dict = {"mode": "fixed", "sources": {}}
    for name, spec in (config["corpus"]["fixed"] or {}).items():
        if name not in {"correct", "poisoned"}:
            raise ValueError(f"corpus.fixed keys must be 'correct' or 'poisoned', got {name!r}")
        condition = "false" if name == "poisoned" else "correct"
        spec = spec or {}
        if not spec.get("enabled", True):
            report["sources"][name] = {"enabled": False, "documents": 0}
            continue
        receipts = {}
        if spec.get("manifest"):
            receipts = {str(row["doc_id"]): row.get("receipt", {}) for row in iter_jsonl(spec["manifest"])}
        added = []
        if spec.get("kind") == "attack":
            # Question + false-answer documents have no source document: the query id is
            # stored in the row and the document is its own root, so it never counts as
            # a relevant document (see r_attack@k for how often it is retrieved).
            if name != "poisoned":
                raise ValueError("corpus.fixed kind 'attack' is only valid for 'poisoned'")
            query_ids = set(owners.values())
            for row in iter_jsonl(spec["path"]):
                node = str(row["_id"])
                qid = str(row["query_id"])
                if qid not in query_ids:
                    raise ValueError(f"{spec['path']}: attack document {node} belongs to unknown query {qid}")
                receipt = receipts.get(node, {})
                added.append(Doc(
                    doc_id=f"gen::{spec.get('label', 'fixed')}-{condition}::{node}",
                    title=str(row.get("title", "") or ""), text=str(row["text"]), origin="generated",
                    query_id=qid, condition="false",
                    gold_answer=receipt.get("gold_answer"), target_answer=receipt.get("target_answer"),
                    parent_id=None, seed_id=None, provenance_root_id=node,
                    generation_depth=1, generator_model=spec.get("model"),
                    metadata={"origin": receipt.get("origin", "attack_v1"), "attack": True},
                ))
            if len({d.doc_id for d in added}) != len(added):
                raise ValueError(f"Duplicate documents in {spec['path']}")
            docs.extend(added)
            report["sources"][name] = {"enabled": True, "path": str(spec["path"]),
                                       "documents": len(added), "kind": "attack",
                                       "manual_documents": 0}
            continue
        for row in iter_jsonl(spec["path"]):
            root = str(row["_id"])
            if root not in owners:
                raise ValueError(f"{spec['path']}: document {root} is not a relevant document of any query")
            receipt = receipts.get(root, {})
            added.append(Doc(
                doc_id=f"gen::{spec.get('label', 'fixed')}-{condition}::{root}",
                title=str(row.get("title", "") or ""), text=str(row["text"]), origin="generated",
                query_id=owners[root], condition=condition,  # type: ignore[arg-type]
                gold_answer=receipt.get("gold_answer"), target_answer=receipt.get("target_answer"),
                parent_id=root, seed_id=root, provenance_root_id=root,
                generation_depth=1, generator_model=spec.get("model"),
                metadata={"origin": receipt.get("origin", "generated")},
            ))
        if len({d.doc_id for d in added}) != len(added):
            raise ValueError(f"Duplicate documents in {spec['path']}")
        docs.extend(added)
        report["sources"][name] = {
            "enabled": True, "path": str(spec["path"]), "documents": len(added),
            "manual_documents": sum(1 for d in added if d.metadata["origin"].startswith("manual")),
        }
    return docs, report


def _build_legacy_corpus(config: dict) -> list[Doc]:
    data = config["data"]
    corpus = config.get("corpus", {})
    base = str(corpus.get("base", "human"))
    docs = load_corpus(data["human_corpus"], origin="human", namespace="human")
    if base == "cocktail_mixed":
        docs.extend(
            load_corpus(
                data["cocktail_llm_corpus"],
                origin="cocktail_llm",
                namespace="cocktail_llm",
            )
        )
    elif base != "human":
        raise ValueError(f"Unsupported base corpus: {base}")

    generated_file = corpus.get("generated_docs")
    if generated_file:
        generated = select_generated_docs(
            load_generated_docs(generated_file, query_ids_by_doc=query_ids_by_doc(data)),
            corpus,
        )
        verify_generated_docs(generated, docs, data)
        docs.extend(generated)

    return docs


def select_generated_docs(generated_docs: list[Doc], corpus_config: dict) -> list[Doc]:
    k_correct = int(corpus_config.get("k_correct", 0))
    k_false = int(corpus_config.get("k_false", 0))
    depth = int(corpus_config.get("generation_depth", 1))
    selected: list[Doc] = []
    grouped: dict[tuple[str, str], list[Doc]] = {}
    for doc in generated_docs:
        if doc.generation_depth != depth:
            continue
        if not doc.query_id or not doc.condition:
            continue
        grouped.setdefault((doc.query_id, doc.condition), []).append(doc)
    for (query_id, condition), docs in grouped.items():
        limit = k_correct if condition == "correct" else k_false
        selected.extend(
            sorted(docs, key=lambda doc: (doc.sample_idx or 0, doc.doc_id))[:limit]
        )
    return selected


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
