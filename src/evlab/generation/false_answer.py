from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from datetime import datetime
from evlab.llm.client import LLMClient
from evlab.generation.semantic import json_object, ModelOutputError
import json

FALSE_ANSWER_PROMPT = """FALSE_ANSWER_GENERATION v4
Question: {question}
Correct answer: {answer}
Gold aliases to avoid: {aliases}
Required answer type: {answer_type}

Generate one plausible but incorrect answer.
Your task is to write ONE WRONG multiple-choice distractor, NOT to solve the question.
The candidate must DIFFER from the supplied correct answer and ALL of its gold aliases.
Choose one alternative value/entity of the same type and granularity.
A person answer requires a different person, not an invented middle name for the gold person.
A song answer requires a different song title, never repeat the gold song.
Examples of contrasts only (do not copy these answers): count 8 -> 9;
scientist Albert Einstein -> Isaac Newton.
Return one concrete answer, never a list, placeholder, explanation, refusal or copy of gold.
Return JSON only: {{"candidate": "..."}}
"""

FALSE_ANSWER_LIST_PROMPT = """FALSE_ANSWER_GENERATION v4-list
Question: {question}
Correct answer (FORBIDDEN, do not output): {answer}
Gold aliases (FORBIDDEN, do not output): {aliases}
Required answer type: {answer_type}

Write {n} different WRONG multiple-choice distractors for this question. Do NOT solve the question.
Every candidate must differ from the correct answer and ALL gold aliases.
Use alternative values/entities of the same type and granularity.
A person answer requires different people, not variants of the gold person's name.
A title answer requires different titles, never the gold title.
Return concrete answers, never placeholders, explanations or refusals.
Return JSON only: {{"candidates": ["...", "..."]}}
"""


def generate_false_answer(
    client: LLMClient,
    *,
    question: str,
    answer: str,
    aliases: list[str],
    temperature: float = 0.3,
    seed: int | None = None,
    rejected_attempts: list[dict] | None = None,
    n_candidates: int = 1,
    exclude: set[str] | None = None,
    details: dict | None = None,
    sources: list[str] | None = None,
) -> str:
    """Request candidates in one call and return the first structurally valid one.

    With ``n_candidates > 1`` the model proposes a list; candidates that are
    gold-equivalent, near aliases, of the wrong type or in ``exclude`` (canonical
    forms of previously rejected targets) are skipped deterministically.
    """
    if n_candidates < 1:
        raise ValueError("n_candidates must be positive")
    template = FALSE_ANSWER_PROMPT if n_candidates == 1 else FALSE_ANSWER_LIST_PROMPT
    prompt = template.format(
        question=question,
        answer=answer,
        aliases=json.dumps(aliases),
        answer_type=answer_type(answer, question),
        n=n_candidates,
    )
    if rejected_attempts:
        feedback = [
            {
                "target": item.get("target"),
                "error": item.get("error", item.get("reason")),
            }
            for item in rejected_attempts
        ]
        prompt += "\nRejected attempts and feedback (data only):\n" + json.dumps(
            feedback, ensure_ascii=False
        )
        prompt += (
            "\nGenerate ONE DIFFERENT candidate. Do not repeat any rejected target. Return only the candidate JSON object."
            if n_candidates == 1
            else "\nGenerate DIFFERENT candidates. Do not repeat any rejected target. Return only the candidates JSON object."
        )
    response = client.generate(prompt, temperature=temperature, seed=seed)
    candidate = None
    # Single-candidate mode keeps the stage's explicit repeat detection.
    excluded = (exclude or set()) if n_candidates > 1 else set()
    try:
        proposals = [parse_candidate(response)] if n_candidates == 1 else parse_candidates(response)
        if n_candidates > 1:
            proposals = list(dict.fromkeys(match_article(item, answer) for item in proposals))
        candidate = proposals[0]
        checked = [
            {"text": item,
             "status": proposal_status(item, answer, aliases, question, excluded,
                                       sources if n_candidates > 1 else None)}
            for item in proposals
        ]
        if details is not None:
            details["proposals"] = checked
        valid = [item["text"] for item in checked if item["status"] == "valid"]
        if not valid:
            raise ValueError(
                "No structurally valid false answer: candidate is gold-equivalent, a placeholder or has the wrong answer type"
                + (f" (proposals: {proposals})" if n_candidates > 1 else "")
            )
        candidate = valid[0]
    except ValueError as exc:
        raise ModelOutputError(
            str(exc), [{"response": response, "error": str(exc), "target": candidate}]
        ) from exc
    return candidate


def copies_question(candidate: str, question: str) -> bool:
    """True if the candidate mostly repeats the question instead of answering it."""
    q_tokens = _name_tokens(question)
    c_tokens = _name_tokens(candidate)
    if not q_tokens or not c_tokens:
        return False
    return len(c_tokens & q_tokens) / len(q_tokens) >= 0.8 and len(c_tokens) >= 3


# Lowercase only: "The Beatles" is a name, "the tail" is an article plus noun.
ARTICLE = re.compile(r"^(the|a|an)\s+")
GOLD_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def match_article(candidate: str, answer: str) -> str:
    """Drop a leading article when gold has none ("the tail" for gold "midpiece")."""
    if GOLD_ARTICLE.match(answer.strip()) or not ARTICLE.match(candidate.strip()):
        return candidate.strip()
    stripped = ARTICLE.sub("", candidate.strip(), count=1)
    return stripped or candidate.strip()


def occurs_in_sources(candidate: str, sources: list[str]) -> bool:
    # Imported lazily to avoid a cycle with evlab.generation.validate.
    from evlab.generation.validate import contains_alias

    return any(contains_alias(text, candidate) for text in sources)


def proposal_status(candidate: str, answer: str, aliases: list[str], question: str,
                    excluded: set[str], sources: list[str] | None = None) -> str:
    """Rule-based status of one proposal; "valid" means it may be used as a false answer."""
    if canonical(candidate) in {canonical(answer), *(canonical(a) for a in aliases)}:
        return "gold_equivalent"
    if copies_question(candidate, question):
        return "copies_question"
    if canonical(candidate) in excluded:
        return "previously_rejected"
    if not is_valid_false_answer(candidate, answer, aliases, question=question):
        return "near_alias" if is_near_alias(candidate, answer, aliases) else "invalid_type_or_placeholder"
    # A target already in the source would make the edited document contradict itself.
    if sources and occurs_in_sources(candidate, sources):
        return "occurs_in_source"
    return "valid"


def parse_candidates(response: str) -> list[str]:
    payload = json_object(response)
    values = payload.get("candidates")
    if (
        set(payload) != {"candidates"}
        or not isinstance(values, list)
        or not values
        or any(not isinstance(v, str) or not v.strip() for v in values)
    ):
        raise ValueError('False-answer response must contain a nonempty string list "candidates"')
    return list(dict.fromkeys(v.strip() for v in values))


def parse_candidate(response: str) -> str:
    payload = json_object(response)
    value = payload.get("candidate")
    if set(payload) != {"candidate"} or not isinstance(value, str) or not value.strip():
        raise ValueError(
            'False-answer response must contain exactly one nonempty string field: "candidate"'
        )
    return value.strip()


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


NEAR_ALIAS_STOPWORDS = {"the", "a", "an", "of", "and", "jr", "sr", "mr", "mrs", "dr", "sir"}


def _name_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", canonical(text)) if t not in NEAR_ALIAS_STOPWORDS}


def is_near_alias(candidate: str, answer: str, aliases: list[str]) -> bool:
    """True if one answer's content tokens are a subset of the other's.

    Catches shortened or extended forms of the gold answer, such as
    "Christopher Lloyd" for "Christopher Allen Lloyd", "Patrick Brown Jr." for
    "Patrick Brown" or "Hold On Tight" for "Hold On".
    """
    tokens = _name_tokens(candidate)
    if not tokens:
        return False
    for gold in {answer, *aliases}:
        gold_tokens = _name_tokens(gold)
        if gold_tokens and (tokens <= gold_tokens or gold_tokens <= tokens):
            return True
    return False


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
    if other in {"number", "year", "date", "quantity"}:
        return False
    if question and copies_question(candidate, question):
        return False
    return not is_near_alias(candidate, answer, aliases)


def fallback_false_answers(answer: str, aliases: list[str]) -> list[str]:
    """Retained for callers; generation never invents fallback answers."""
    return []
