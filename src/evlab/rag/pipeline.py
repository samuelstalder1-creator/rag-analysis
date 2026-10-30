from __future__ import annotations

from evlab.rag.context import format_context
from evlab.rag.reader import read_answer
from evlab.schema import Doc, Hit, Query


def answer_with_hits(
    client,
    *,
    query: Query,
    hits: list[Hit],
    docs_by_id: dict[str, Doc],
    top_k: int = 10,
    temperature: float = 0.0,
    seed: int | None = None,
    prompt: str | None = None,
    context_order: str = "retrieval",
) -> dict:
    selected = list(hits[:top_k])
    if context_order == "reverse":
        selected.reverse()
    elif context_order != "retrieval":
        raise ValueError(f"Unknown reader context_order: {context_order}")
    context, doc_ids = format_context(selected, docs_by_id)
    result = read_answer(
        client,
        question=query.text,
        context=context,
        temperature=temperature,
        seed=seed,
        prompt=prompt,
    )
    result["query_id"] = query.query_id
    result["context_doc_ids"] = doc_ids
    result["context_order"] = context_order
    result["context_ir_ranks"] = [hit.rank for hit in selected]
    return result
