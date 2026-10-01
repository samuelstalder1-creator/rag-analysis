from __future__ import annotations

from evlab.evaluation.answers import contains_answer, normalize
from evlab.schema import Doc


def citation_supports_answer(doc: Doc, answer: str) -> bool:
    return contains_answer(normalize(doc.index_text), normalize(answer))
