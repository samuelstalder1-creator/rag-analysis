"""Typography must not hide answers or change the offsets used for editing."""

import pytest

from evlab.generation.minimal_edit import apply_spans, occurrences
from evlab.generation.output import source_identity
from evlab.generation.false_answers import deterministic_answer_receipt, valid_saved_answer
from evlab.io import stable_hash
from evlab.generation.validate import contains_alias, validate_correct_copy, validate_false_copy


@pytest.mark.parametrize("text, alias", [
    ("Written by John\u202fSmith.", "John Smith"),
    ("John\u00a0\u00a0Smith", "John Smith"),
    ("John\n\tSmith", "John  Smith"),
    ("John Smith", "John\u202fSmith"),
    ("Mercedes\u2011Benz Stadium", "Mercedes-Benz Stadium"),
    ("B-29", "B\u201029"),
    ("O\u2019Brien", "O'Brien"),
    ("\u201cThe Wedding March\u201d", '"The Wedding March"'),
])
def test_typographic_variants_are_answers(text, alias):
    assert contains_alias(text, alias)
    assert validate_correct_copy(text, answer_aliases=[alias]).ok
    assert validate_false_copy(text, target_answer=alias, answer_aliases=["unrelated"]).ok
    assert occurrences(text, [alias])


@pytest.mark.parametrize("text, alias", [
    ("John Smithson", "John Smith"),
    ("JohnSmith", "John Smith"),
    ("B\u201329", "B-29"),  # en dash
    ("B\u201429", "B-29"),  # em dash
    ("B\u221229", "B-29"),  # minus sign
    ("June 1 1977", "June 1, 1977"),
    ("after 8 p.m. ET", "after 8pm ET"),
    ('\u201cThe Wedding March.\u201d', '"The Wedding March"'),
    ("John Jones", "John Smith"),
    ("anything", "  "),
])
def test_content_differences_still_fail(text, alias):
    assert not contains_alias(text, alias)
    assert not validate_correct_copy(text, answer_aliases=[alias]).ok
    assert not occurrences(text, [alias])


def test_gold_residue_uses_same_typography_rules():
    text = "The target is Jane Doe; John\u202fSmith is also mentioned."
    result = validate_false_copy(text, target_answer="Jane Doe", answer_aliases=["John Smith"])
    assert result.reasons == ("still_contains_gold_answer",)
    assert len(occurrences(text, ["John Smith"])) == 1


@pytest.mark.parametrize("text", [
    'The song is called “The Wedding March.”',
    'The video for “The Wedding March,” was released.',
    'The song is The Wedding March.',
])
def test_outer_title_quotes_are_not_answer_content(text):
    assert validate_false_copy(text, target_answer='"The Wedding March"', answer_aliases=[]).ok


@pytest.mark.parametrize("text", ['The Wedding Marching', 'The Wedding Song', 'The Wedding, March'])
def test_quoted_target_still_requires_complete_words(text):
    assert not validate_false_copy(text, target_answer='"The Wedding March"', answer_aliases=[]).ok


def test_edit_offsets_refer_to_original_text():
    text = "Before: John\u202f\u00a0Smith, then O\u2019Brien. After."
    spans = occurrences(text, ["John Smith", "O'Brien"])
    assert len(spans) == 2
    assert all(text[s["start"]:s["end"]] == s["text"] for s in spans)
    edited, _ = apply_spans(text, spans, "Jane Doe")
    assert edited == "Before: Jane Doe, then Jane Doe. After."


def test_source_identity_preserves_resume_after_typography_change():
    original = {"doc_id": "d1", "title": "", "text": "John\u202fSmith", "aliases_in_doc": []}
    current = {**original, "aliases_in_doc": ["John Smith"]}
    assert source_identity(current, ["John Smith"]) == original
    assert current["aliases_in_doc"] == ["John Smith"]
    changed = {**current, "text": "John\u202fJones"}
    assert stable_hash(source_identity(changed, ["John Smith"])) != stable_hash(original)


def test_imported_answer_receipt_survives_derived_alias_change():
    candidate = {
        "query": "Who wrote it?", "answer": "John Smith",
        "answer_aliases": ["John Smith", "Smith"], "false_answer": "Jane Doe",
        "h_docs": [{"doc_id": "d1", "title": "", "text": "John\u202fSmith wrote it.",
                    "aliases_in_doc": ["Smith"]}],
    }
    receipt = deterministic_answer_receipt(candidate, "Jane Doe")
    candidate["false_answer_record"] = {"verification": receipt}
    candidate["h_docs"][0]["aliases_in_doc"] = ["John Smith", "Smith"]
    assert deterministic_answer_receipt(candidate, "Jane Doe") == receipt
    assert valid_saved_answer(candidate, require_semantic=False)
    candidate["h_docs"][0]["text"] += " Changed source."
    assert not valid_saved_answer(candidate, require_semantic=False)
