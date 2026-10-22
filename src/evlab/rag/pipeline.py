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
) -> dict:
    context, doc_ids = format_context(hits[:top_k], docs_by_id)
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
    return result
