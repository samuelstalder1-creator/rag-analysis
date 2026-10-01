from __future__ import annotations

import json
import re

from evlab.llm.client import LLMClient


READER_PROMPT = """Answer the question using only the numbered context.
If the context is insufficient or conflicting, answer "I don't know".
Return JSON only: {{"answer": "...", "citations": [1, 3]}}

Question: {question}

Context:
{context}
"""


def read_answer(client: LLMClient, *, question: str, context: str, temperature: float = 0.0) -> dict:
    response = client.generate(READER_PROMPT.format(question=question, context=context), temperature=temperature)
    return parse_reader_response(response)


def parse_reader_response(response: str) -> dict:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return {"answer": response.strip(), "citations": [], "valid_json": False}
        payload = json.loads(match.group(0))
    return {
        "answer": str(payload.get("answer", "")).strip(),
        "citations": [int(item) for item in payload.get("citations", []) if str(item).isdigit()],
        "valid_json": True,
    }
