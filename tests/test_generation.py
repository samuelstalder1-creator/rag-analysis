from evlab.generation.false_answer import is_valid_false_answer, parse_candidates
from evlab.generation.minimal_edit import deterministic_minimal_edit
from evlab.generation.validate import validate_false_copy


def test_parse_false_answer_candidates():
    assert parse_candidates('{"candidates": ["22", "24", "25"]}') == ["22", "24", "25"]


def test_false_answer_rejects_gold_alias():
    assert not is_valid_false_answer("23", "23", ["23"])
    assert is_valid_false_answer("24", "23", ["23"])


def test_minimal_edit_replaces_aliases():
    text = "The season contained 23 episodes."
    assert deterministic_minimal_edit(text, ["23"], "24") == "The season contained 24 episodes."


def test_false_validation_rejects_old_answer():
    result = validate_false_copy("The season contained 23 episodes.", target_answer="24", answer_aliases=["23"])
    assert not result.ok
    assert "missing_target_answer" in result.reasons
    assert "still_contains_gold_answer" in result.reasons
