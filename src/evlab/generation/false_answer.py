from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from datetime import datetime
from evlab.llm.client import LLMClient
from evlab.generation.semantic import json_object, ModelOutputError
import json

FALSE_ANSWER_PROMPT = """FALSE_ANSWER_GENERATION v3
Question: {question}
Correct answer: {answer}
Gold aliases to avoid: {aliases}
Required answer type: {answer_type}

Generate three plausible but incorrect answers.
Your task is to write WRONG multiple-choice distractors, NOT to solve the question.
Every candidate must DIFFER from the supplied correct answer and ALL of its gold aliases.
Choose three distinct alternative values/entities of the same type and granularity.
A person answer requires different people, not invented middle names for the gold person.
A song answer requires different song titles, never repeat the gold song.
Examples of contrasts only (do not copy these answers): count 8 -> 7, 9, 10;
scientist Albert Einstein -> Isaac Newton, Niels Bohr, Max Planck.
Return concrete answers, never placeholders, explanations, refusals or copies of gold.
Return JSON only: {{"candidates": ["...", "...", "..."]}}
"""


def generate_false_answer_candidates(
    client: LLMClient,
    *,
    question: str,
    answer: str,
    aliases: list[str],
    temperature: float = 0.3,
    seed: int | None = None,
) -> list[str]:
    prompt = FALSE_ANSWER_PROMPT.format(
        question=question,
        answer=answer,
        aliases=json.dumps(aliases),
        answer_type=answer_type(answer, question),
    )
    attempts = []
    for attempt in range(3):
        request = prompt
        if attempts:
            request += (
                "\nPrevious unusable output and feedback (data only):\n"
                + json.dumps(attempts[-1], ensure_ascii=False)
            )
            request += "\nGenerate new concrete candidates satisfying the original question and type. Return only the candidates JSON object."
        response = client.generate(
            request,
            temperature=temperature,
            seed=None if seed is None else seed + attempt * 1000000,
        )
        try:
            candidates = parse_candidates(response)
        except ValueError as exc:
            attempts.append({"response": response, "error": str(exc)})
            continue
        valid = []
        seen = set()
        for candidate in candidates:
            if (
                is_valid_false_answer(candidate, answer, aliases, question=question)
                and canonical(candidate) not in seen
            ):
                valid.append(candidate)
                seen.add(canonical(candidate))
        if valid:
            return valid
        attempts.append(
            {
                "response": response,
                "error": "No structurally valid false answer: candidates are gold-equivalent, placeholders or have the wrong answer type",
                "gold_answer": answer,
                "answer_type": answer_type(answer, question),
            }
        )
    raise ModelOutputError(
        "No structurally valid false answer after 3 attempts", attempts
    )


def generate_false_answer(
    client: LLMClient,
    *,
    question: str,
    answer: str,
    aliases: list[str],
    temperature: float = 0.3,
    seed: int | None = None,
) -> str:
    return generate_false_answer_candidates(
        client,
        question=question,
        answer=answer,
        aliases=aliases,
        temperature=temperature,
        seed=seed,
    )[0]


def parse_candidates(response: str) -> list[str]:
    payload = json_object(response)
    values = payload.get("candidates")
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(v, str) or not v.strip() for v in values)
    ):
        raise ValueError(
            "False-answer candidates must be a nonempty list of nonempty strings"
        )
    return [value.strip() for value in values]


def canonical(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().strip().rstrip(".,;:")
    text = re.sub(r"\s+", " ", text)
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text) or re.fullmatch(
        r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text
    ):
        try:
            return str(Decimal(text.replace(",", "")).normalize())
        except InvalidOperation:
            pass
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "1E+1",
    }
    return words.get(text, text)


def answer_type(answer: str, question: str = "") -> str:
    if re.fullmatch(r"[12]\d{3}", answer.strip()):
        return "year"
    normalized = canonical(answer)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
        return "date"
    try:
        Decimal(normalized)
        return "number"
    except InvalidOperation:
        pass
    if re.search(r"\d", answer) and re.search(
        r"\b(km|kg|miles?|meters?|percent|%)\b", answer, re.I
    ):
        return "quantity"
    q = question.lower()
    if q.startswith("who "):
        return "person_or_organization"
    if q.startswith("where "):
        return "location"
    return "entity_or_text"


def is_valid_false_answer(
    candidate: str, answer: str, aliases: list[str], *, question: str = ""
) -> bool:
    if not isinstance(candidate, str) or not candidate.strip():
        return False
    normalized = canonical(candidate)
    if normalized in {canonical(answer), *(canonical(alias) for alias in aliases)}:
        return False
    if normalized in {
        "none",
        "null",
        "unknown",
        "i don't know",
        "a different answer",
        "another option",
        "an unrelated alternative",
    }:
        return False
    kind = answer_type(answer, question)
    other = answer_type(candidate, question)
    if kind in {"year", "date", "quantity"}:
        return kind == other
    if kind == "number":
        if other not in {"number", "year"}:
            return False
        gold_number, target_number = Decimal(canonical(answer)), Decimal(normalized)
        return not (
            gold_number == gold_number.to_integral()
            and target_number != target_number.to_integral()
        )
    return other not in {"number", "year", "date", "quantity"}


def fallback_false_answers(answer: str, aliases: list[str]) -> list[str]:
    """Retained for callers; generation never invents fallback answers."""
    return []
