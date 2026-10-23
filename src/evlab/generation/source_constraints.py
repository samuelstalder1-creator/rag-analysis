"""Check answer presence in every original source, before counterfactual editing."""

from __future__ import annotations

from datetime import date
import unicodedata

from evlab.generation.false_answer import canonical
from evlab.generation.validate import contains_alias, number_variants


def normalize_source_text(text: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", text)
        .replace("\u00ad", "")
        .replace("\u200b", "")
        .split()
    )


def full_answer_forms(answer: str) -> list[str]:
    """Equivalent full forms only: never reduce a complete date to its year."""
    normalized = normalize_source_text(answer)
    forms = number_variants(normalized)
    try:
        parsed = date.fromisoformat(canonical(normalized))
    except ValueError:
        return list(dict.fromkeys(forms))
    # English month spellings, independent of the execution host's locale.
    months = "January February March April May June July August September October November December".split()
    month = months[parsed.month - 1]
    for name in (month, month[:3]):
        forms.extend(
            [
                f"{parsed.day} {name} {parsed.year}",
                f"{parsed.day:02d} {name} {parsed.year}",
                f"{name} {parsed.day}, {parsed.year}",
                f"{name} {parsed.day:02d}, {parsed.year}",
            ]
        )
    forms.append(parsed.isoformat())
    return list(dict.fromkeys(forms))


def check_source_constraints(
    *, gold: str, target: str | None, documents: list[dict]
) -> dict:
    gold_forms = full_answer_forms(gold)
    target_forms = full_answer_forms(target) if target else []
    checked = []
    for doc in documents:
        text = normalize_source_text(doc.get("title", "") + "\n" + doc["text"])
        gold_matches = [form for form in gold_forms if contains_alias(text, form)]
        target_matches = [form for form in target_forms if contains_alias(text, form)]
        checked.append(
            {
                "doc_id": doc["doc_id"],
                "gold_present": bool(gold_matches),
                "target_absent": not target_matches,
                "gold_matches": gold_matches,
                "target_matches": target_matches,
            }
        )
    all_gold = bool(checked) and all(doc["gold_present"] for doc in checked)
    all_target_absent = (
        bool(checked)
        and bool(target_forms)
        and all(doc["target_absent"] for doc in checked)
    )
    return {
        "policy": "full_gold_present_target_absent_in_every_original_document_v1",
        "scope": "original_title_and_body",
        "normalization": "NFKC, whitespace, soft-hyphen/zero-width removal; full number/date equivalents",
        "gold_forms": gold_forms,
        "target_forms": target_forms,
        "documents": checked,
        "all_gold_present": all_gold,
        "all_target_absent": all_target_absent,
        "eligible": all_gold and all_target_absent,
    }
