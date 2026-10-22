from __future__ import annotations


from evlab.llm.client import LLMClient
from evlab.generation.semantic import json_object


READER_PROMPT = """Answer the question using only the numbered context.
If the context is insufficient or conflicting, answer "I don't know".
Return a concise answer value without explanations or alternative answers.
Return JSON only: {{"answer": "...", "citations": [1, 3]}}

Question: {question}

Context:
{context}
"""


def read_answer(
    client: LLMClient,
    *,
    question: str,
    context: str,
    temperature: float = 0.0,
    seed: int | None = None,
    prompt: str | None = None,
) -> dict:
    """Ask the reader LLM. ``prompt`` is a template with {question} and {context}
    (literal braces doubled); the default is ``READER_PROMPT``."""
    template = prompt or READER_PROMPT
    response = client.generate(
        template.format(question=question, context=context),
        temperature=temperature,
        seed=seed,
    )
    return parse_reader_response(response)


def parse_reader_response(response: str) -> dict:
    try:
        payload = json_object(response)
        answer = payload.get("answer")
        citations = payload.get("citations")
        if (
            not isinstance(answer, str)
            or not isinstance(citations, list)
            or any(type(i) is not int or i < 1 for i in citations)
        ):
            raise ValueError("Invalid reader answer schema")
        return {"answer": answer.strip(), "citations": citations, "valid_json": True}
    except (ValueError, TypeError):
        return {"answer": response.strip(), "citations": [], "valid_json": False}
