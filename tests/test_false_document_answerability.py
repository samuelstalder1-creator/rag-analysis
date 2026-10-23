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
