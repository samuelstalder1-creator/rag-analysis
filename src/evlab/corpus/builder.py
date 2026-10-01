from __future__ import annotations

from pathlib import Path

from evlab.data.cocktail import load_corpus
from evlab.io import iter_jsonl
from evlab.schema import Doc


def load_generated_docs(path: str | Path) -> list[Doc]:
    docs: list[Doc] = []
    for row in iter_jsonl(path):
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
                sample_idx=int(row["sample_idx"]) if row.get("sample_idx") is not None else None,
                metadata=dict(row),
            )
        )
    return docs


def build_corpus(config: dict) -> list[Doc]:
    data = config["data"]
    corpus = config.get("corpus", {})
    base = str(corpus.get("base", "human"))
    docs = load_corpus(data["human_corpus"], origin="human", namespace="human")
    if base == "cocktail_mixed":
        docs.extend(load_corpus(data["cocktail_llm_corpus"], origin="cocktail_llm", namespace="cocktail_llm"))
    elif base != "human":
        raise ValueError(f"Unsupported base corpus: {base}")

    generated_file = corpus.get("generated_docs")
    if generated_file:
        docs.extend(select_generated_docs(load_generated_docs(generated_file), corpus))
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
        selected.extend(sorted(docs, key=lambda doc: (doc.sample_idx or 0, doc.doc_id))[:limit])
    return selected


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
