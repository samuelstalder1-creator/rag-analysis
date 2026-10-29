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


def target_answer_forms(value: str) -> list[str]:
    """Outer title quotes are presentation, not part of the target's wording."""
    value = value.strip()
    forms = number_variants(value)
    if len(value) > 2 and (value[0], value[-1]) in {('"', '"'), ('\u201c', '\u201d')}:
        unquoted = value[1:-1].strip()
        if unquoted:
            forms.extend(number_variants(unquoted))
    return list(dict.fromkeys(forms))


def validate_false_copy(
    text: str,
    *,
    target_answer: str,
    answer_aliases: list[str],
    allow_unrelated_gold: bool = False,
) -> ValidationResult:
    reasons: list[str] = []
    # A paraphrase may write "four" for target "4"; both forms count.
    if not any(contains_alias(text, form) for form in target_answer_forms(target_answer)):
        reasons.append("missing_target_answer")
    if not allow_unrelated_gold and any(
        contains_alias(text, alias) for alias in answer_aliases
    ):
        reasons.append("still_contains_gold_answer")
    if not text.strip():
        reasons.append("empty_text")
    return ValidationResult(not reasons, tuple(reasons))


def alias_pattern(alias: str) -> str:
    """Match typographic variants without changing content or source offsets.

    Unicode whitespace, hyphens and quotation marks are equivalent to their
    plain-text forms. Other punctuation (including en/em dashes and minus
    signs), spelling and numeric/date formats remain significant.
    """
    alias = alias.strip()
    if not alias:
        return r"(?!)"
    equivalents = {
        **dict.fromkeys("-\u2010\u2011", "[-\u2010\u2011]"),
        **dict.fromkeys("'\u2018\u2019", "['\u2018\u2019]"),
        **dict.fromkeys('\"\u201c\u201d', '[\"\u201c\u201d]'),
    }
    pattern = "".join(
        r"\s+" if part.isspace() else
        "".join(equivalents.get(char, re.escape(char)) for char in part)
        for part in re.split(r"(\s+)", alias)
    )
    if re.search(r"^\w", alias):
        pattern = r"\b" + pattern
    if re.search(r"\w$", alias):
        pattern = pattern + r"\b"
    return pattern


def contains_alias(text: str, alias: str) -> bool:
    return re.search(alias_pattern(alias), text, flags=re.IGNORECASE) is not None
