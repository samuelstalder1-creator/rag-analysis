from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reasons: tuple[str, ...] = ()


def validate_correct_copy(text: str, *, answer_aliases: list[str]) -> ValidationResult:
    reasons: list[str] = []
    if not any(contains_alias(text, alias) for alias in answer_aliases):
        reasons.append("missing_gold_answer")
    if not text.strip():
        reasons.append("empty_text")
    return ValidationResult(not reasons, tuple(reasons))


def validate_nonempty(text: str) -> ValidationResult:
    return ValidationResult(bool(text.strip()), () if text.strip() else ("empty_text",))


NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                "seventeen", "eighteen", "nineteen", "twenty"]


def number_variants(value: str) -> list[str]:
    """The value plus its digit/word counterpart for 0-20 ("4" <-> "four")."""
    text = value.strip()
    if text.isdigit() and int(text) < len(NUMBER_WORDS):
        return [text, NUMBER_WORDS[int(text)]]
    if text.lower() in NUMBER_WORDS:
        return [text, str(NUMBER_WORDS.index(text.lower()))]
    return [text]


def expand_number_aliases(aliases: list[str]) -> list[str]:
    return list(dict.fromkeys(v for alias in aliases for v in number_variants(alias)))


def validate_false_copy(
    text: str,
    *,
    target_answer: str,
    answer_aliases: list[str],
    allow_unrelated_gold: bool = False,
) -> ValidationResult:
    reasons: list[str] = []
    # A paraphrase may write "four" for target "4"; both forms count.
    if not any(contains_alias(text, form) for form in number_variants(target_answer)):
        reasons.append("missing_target_answer")
    if not allow_unrelated_gold and any(
        contains_alias(text, alias) for alias in answer_aliases
    ):
        reasons.append("still_contains_gold_answer")
    if not text.strip():
        reasons.append("empty_text")
    return ValidationResult(not reasons, tuple(reasons))


def contains_alias(text: str, alias: str) -> bool:
    alias = alias.strip()
    if not alias:
        return False
    escaped = re.escape(alias)
    pattern = escaped
    if re.search(r"^\w", alias):
        pattern = r"\b" + pattern
    if re.search(r"\w$", alias):
        pattern = pattern + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None
