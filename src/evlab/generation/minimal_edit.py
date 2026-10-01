from __future__ import annotations

import re

from evlab.llm.client import LLMClient


MINIMAL_EDIT_PROMPT = """Original Text: {text}

Replace every mention of the answer ({aliases}) with "{target_answer}",
adapted to the format of each mention.
Change only what is necessary.
Keep all other facts and wording unchanged.
Return only the edited text.
"""


def deterministic_minimal_edit(text: str, aliases: list[str], target_answer: str) -> str:
    edited = text
    for alias in sorted({alias for alias in aliases if alias.strip()}, key=len, reverse=True):
        edited = re.sub(_alias_pattern(alias), target_answer, edited, flags=re.IGNORECASE)
    return edited


def minimal_edit_with_fallback(
    client: LLMClient,
    *,
    text: str,
    aliases: list[str],
    target_answer: str,
    temperature: float = 0.0,
    seed: int | None = None,
) -> str:
    edited = deterministic_minimal_edit(text, aliases, target_answer)
    if edited != text and target_answer.lower() in edited.lower():
        return edited
    prompt = MINIMAL_EDIT_PROMPT.format(text=text, aliases=", ".join(aliases), target_answer=target_answer)
    return client.generate(prompt, temperature=temperature, seed=seed).strip()


def _alias_pattern(alias: str) -> str:
    escaped = re.escape(alias.strip())
    if re.search(r"^\w", alias):
        escaped = r"\b" + escaped
    if re.search(r"\w$", alias):
        escaped = escaped + r"\b"
    return escaped
