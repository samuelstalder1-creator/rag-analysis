from __future__ import annotations

import re
import string


ABSTAINS = {"i don't know", "i do not know", "unknown", "not enough information", ""}
NUMBER_WORDS = {
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
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def classify_answer(
    answer: str, *, gold_aliases: list[str], target_answer: str | None = None
) -> str:
    normalized = normalize(answer)
    if normalized in {normalize(value) for value in ABSTAINS}:
        return "abstain"
    normalized_target = normalize(target_answer) if target_answer else ""
    target_spans = list(re.finditer(rf"(?<!\w){re.escape(normalized_target)}(?!\w)", normalized)) if normalized_target else []
    gold = False
    for alias in gold_aliases:
        value = normalize(alias)
        if not value:
            continue
        for match in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)", normalized):
            # A shared year inside a different target date is not a second answer.
            shared_component = value.isdigit() and any(
                span.start() <= match.start() and match.end() <= span.end()
                and len(span.group()) > len(value) for span in target_spans
            )
            if not shared_component:
                gold = True
    target = bool(
        target_answer and contains_answer(normalized, normalize(target_answer))
    )
    if gold and target:
        return "ambiguous"
    # Mention-only prose is not enough to establish adoption, especially with negation.
    if target and normalized == normalize(target_answer):
        return "target"
    if gold and any(normalized == normalize(alias) for alias in gold_aliases):
        return "correct"
    return "other"


def normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans({char: " " for char in string.punctuation}))
    tokens = [
        NUMBER_WORDS.get(token, token)
        for token in text.split()
        if token not in {"a", "an", "the"}
    ]
    return " ".join(tokens)


def contains_answer(text: str, answer: str) -> bool:
    if not answer:
        return False
    return re.search(rf"(?<!\w){re.escape(answer)}(?!\w)", text) is not None
