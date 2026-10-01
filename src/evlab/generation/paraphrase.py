from __future__ import annotations

from evlab.llm.client import LLMClient


PARAPHRASE_PROMPT = """Original Text: {text}

Rewrite the text.
Preserve all factual claims of the input.
Change wording and style.
Return only the rewritten text.
"""


def paraphrase(
    client: LLMClient,
    *,
    text: str,
    temperature: float = 0.4,
    seed: int | None = None,
) -> str:
    prompt = PARAPHRASE_PROMPT.format(text=text)
    return client.generate(prompt, temperature=temperature, seed=seed).strip()
