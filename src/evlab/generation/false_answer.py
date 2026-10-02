from __future__ import annotations

import json
import re

from evlab.llm.client import LLMClient


FALSE_ANSWER_PROMPT = """Question: {question}
Correct answer: {answer}

Generate three plausible but incorrect answers.
They must answer the question, differ from the correct answer, and have the same type.
Return JSON only: {{"candidates": ["...", "...", "..."]}}
"""


def generate_false_answer(
    client: LLMClient,
    *,
    question: str,
    answer: str,
    aliases: list[str],
    temperature: float = 0.3,
    seed: int | None = None,
) -> str:
    prompt = FALSE_ANSWER_PROMPT.format(question=question, answer=answer)
    response = client.generate(prompt, temperature=temperature, seed=seed)
    try:
        candidates = parse_candidates(response)
    except (json.JSONDecodeError, ValueError):
        candidates = []
    candidates.extend(fallback_false_answers(answer, aliases))
    for candidate in candidates:
        if is_valid_false_answer(candidate, answer, aliases):
            return candidate
    raise ValueError(f"No valid false answer candidate in response: {response}")


def parse_candidates(response: str) -> list[str]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    values = payload.get("candidates", [])
    if not isinstance(values, list):
        raise ValueError("False-answer response must contain a candidates list")
    return [str(value).strip() for value in values if str(value).strip()]


def is_valid_false_answer(candidate: str, answer: str, aliases: list[str]) -> bool:
    candidate_norm = _norm(candidate)
    if not candidate_norm:
        return False
    if candidate_norm in {_norm(answer), *{_norm(alias) for alias in aliases}}:
        return False
    return _same_coarse_type(candidate, answer)


def fallback_false_answers(answer: str, aliases: list[str]) -> list[str]:
    answer = answer.strip()
    if _looks_like_year(answer):
        year = int(answer)
        return [str(year + 1), str(year - 1), str(year + 2)]
    if _is_number(answer):
        normalized = answer.replace(",", ".")
        value = float(normalized)
        if value.is_integer():
            integer = int(value)
            return [str(integer + 1), str(max(0, integer - 1)), str(integer + 2)]
        return [f"{value + 1:g}", f"{value + 0.5:g}", f"{max(0.0, value - 0.5):g}"]
    blocked = {_norm(answer), *{_norm(alias) for alias in aliases}}
    candidates = ["a different answer", "another option", "an unrelated alternative"]
    return [candidate for candidate in candidates if _norm(candidate) not in blocked]


def _same_coarse_type(left: str, right: str) -> bool:
    if _is_number(left) or _is_number(right):
        return _is_number(left) and _is_number(right)
    if _looks_like_year(left) or _looks_like_year(right):
        return _looks_like_year(left) and _looks_like_year(right)
    return True


def _is_number(text: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:[.,]\d+)?", text.strip()))


def _looks_like_year(text: str) -> bool:
    return bool(re.fullmatch(r"(1|2)\d{3}", text.strip()))


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip().rstrip(".,;:"))
