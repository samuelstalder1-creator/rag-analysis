import json

import pytest

from evlab.generation.paraphrase import paraphrase, paraphrase_false
from evlab.generation.pipeline import _check_document
from evlab.generation.semantic import ModelOutputError


def evidence():
    return dict(
        question="How many episodes?",
        gold="23",
        aliases=["23"],
        target="24",
        source="23 episodes aired in winter.",
        seed="24 episodes aired in winter.",
        generated="In spring, 24 episodes aired.",
        condition="false",
        stage="paraphrase",
    )


def test_false_prompt_includes_question_target_and_edited_context():
    class Client:
        def generate(self, prompt, **kwargs):
            assert "Question: How many episodes?" in prompt
            assert "Target answer for this counterfactual passage: 24" in prompt
            assert "Edited Title: Season four" in prompt
            assert "Edited Text: 24 episodes aired in winter." in prompt
            assert "reproducing every source\nclaim is not required" in prompt
            assert "Preserve all factual claims" not in prompt
            assert kwargs == {"temperature": 0.3, "seed": 42}
            return "  The season had 24 episodes.  "

    assert (
        paraphrase_false(
            Client(),
            question="How many episodes?",
            target_answer="24",
            title="Season four",
            text="24 episodes aired in winter.",
            temperature=0.3,
            seed=42,
        )
        == "The season had 24 episodes."
    )


def test_generic_paraphrase_retains_existing_contract():
    class Client:
        def generate(self, prompt, **kwargs):
            assert "Preserve all factual claims of the input." in prompt
            assert "Question:" not in prompt
            return "Rewritten."

    assert paraphrase(Client(), text="Source.") == "Rewritten."


@pytest.mark.parametrize("bad, reason", [
    ("The total was 24, although 23 were originally planned.", "gold_answer_residue"),
    ("The show ran for several years.", "missing_target_answer"),
    ("", "empty_text"),
])
def test_false_content_errors_are_retried_with_specific_feedback(bad, reason):
    calls = []

    class Client:
        def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return bad if len(calls) == 1 else "Viewers saw 24 installments over the full season."

    result = paraphrase_false(
        Client(), question="How many episodes?", target_answer="24", title="Show",
        text="A total of 24 episodes aired.", answer_aliases=["23"], max_attempts=3, seed=7,
    )
    assert result.startswith("Viewers saw 24")
    assert len(calls) == 2
    assert reason in calls[1][0]
    assert "Previous rejected passage" in calls[1][0]
    if reason == "gold_answer_residue":
        assert "Disallowed answer mentions actually found: ['23']" in calls[1][0]
        assert "at most two sentences" in calls[1][0]
    assert calls[1][1]["seed"] != calls[0][1]["seed"]


def test_repeated_gold_residue_remains_rejected():
    class Client:
        def generate(self, prompt, **kwargs):
            return "There were 24 episodes, but also 23 elsewhere."

    with pytest.raises(ModelOutputError, match="gold_answer_residue") as error:
        paraphrase_false(Client(), question="How many?", target_answer="24", title="Show",
                         text="24 episodes aired.", answer_aliases=["23"], max_attempts=3)
    assert len(error.value.attempts) == 3


def test_copied_false_passage_is_retried_with_feedback_and_new_seed():
    calls = []

    class Client:
        def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            if len(calls) == 1:
                return "The season contained 25 episodes."
            assert "previous attempt was rejected" in prompt
            return "Chicago Fire aired a total of 25 episodes during its fourth season."

    result = paraphrase_false(
        Client(), question="How many episodes?", target_answer="25",
        title="Chicago Fire (season 4)", text="The season contained 25 episodes.",
        temperature=0.3, seed=1, max_attempts=3,
    )
    assert result.startswith("Chicago Fire aired")
    assert len(calls) == 2
    assert calls[0][1]["seed"] != calls[1][1]["seed"]
    assert calls[1][1]["temperature"] > calls[0][1]["temperature"]


def test_copying_every_attempt_fails_with_responses_for_diagnosis():
    class Client:
        def generate(self, prompt, **kwargs):
            return "The season contained 25 episodes."

    with pytest.raises(ModelOutputError, match="near_duplicate_of_seed") as error:
        paraphrase_false(
            Client(), question="How many episodes?", target_answer="25", title="Season",
            text="The season contained 25 episodes.", max_attempts=3,
        )
    assert len(error.value.attempts) == 3
    assert all(a["response"] == "The season contained 25 episodes." for a in error.value.attempts)


def test_album_title_does_not_trigger_surname_replacement():
    from evlab.generation.minimal_edit import edit_all_answer_spans

    text = "Their album Hold On reached number one on the charts. It was certified on September 12."
    edited, edits = edit_all_answer_spans(text, ["Hold On"], "Moonlit Adventures", gold="Hold On")
    assert edited == text.replace("Hold On", "Moonlit Adventures")
    assert len(edits) == 1


def test_surname_body_expands_only_with_explicit_source_title():
    from evlab.generation.minimal_edit import edit_all_answer_spans

    text = "In May 2015, Brown became party leader. Brown had been an MP."
    edited, edits = edit_all_answer_spans(
        text, ["Patrick Brown"], "Michael Thompson", gold="Patrick Brown",
        context_title="Patrick Brown (politician)",
    )
    assert edited == "In May 2015, Michael Thompson became party leader. Thompson had been an MP."
    assert edits[0]["kind"] == "surname_from_title"
    with pytest.raises(ValueError, match="surname only"):
        edit_all_answer_spans(text, ["Patrick Brown"], "Michael Thompson", gold="Patrick Brown",
                              context_title="Unrelated politician")


@pytest.mark.parametrize(
    "failed_check", [None, "supports_target", "no_gold_support", "unambiguous_answer"]
)
def test_false_judge_checks_target_answerability_not_all_background_claims(
    failed_check,
):
    calls = []

    class Judge:
        provider = model = "test"

        def generate(self, prompt, **kwargs):
            payload = json.loads(prompt.split("\n")[-1])
            calls.append(payload)
            assert set(payload["checks"]) == {
                "source_supports_gold",
                "supports_target",
                "no_gold_support",
                "unambiguous_answer",
            }
            assert "unrelated" in payload["checks"]["unambiguous_answer"]
            assert payload["evidence"]["question"] == "How many episodes?"
            return json.dumps(
                {
                    **{key: key != failed_check for key in payload["checks"]},
                    "reason": "Controlled verdict; unrelated season detail is ignored.",
                }
            )

    if failed_check:
        with pytest.raises(ModelOutputError):
            _check_document(True, Judge(), **evidence())
    else:
        result = _check_document(True, Judge(), **evidence())
        assert result["answerability_verified"] is True
        assert result["validation_goal"] == "query_target_answerability"
    assert len(calls) == 1


def test_deterministic_false_receipt_does_not_claim_answerability_verification():
    result = _check_document(False, None, **evidence())
    assert result["semantic_verified"] is False
    assert result["answerability_verified"] is False
    assert result["validation_goal"] == "query_target_answerability"
