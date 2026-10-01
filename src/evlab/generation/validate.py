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


def validate_false_copy(text: str, *, target_answer: str, answer_aliases: list[str]) -> ValidationResult:
    reasons: list[str] = []
    if not contains_alias(text, target_answer):
        reasons.append("missing_target_answer")
    if any(contains_alias(text, alias) for alias in answer_aliases):
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
