import json
import pytest
from evlab.generation.semantic import verify_answer


class VerifierClient:
    provider = "test"
    model = "test/verifier"

    def generate(self, prompt, **kwargs):
        if prompt.startswith("SEMANTIC_VALIDATION"):
            payload = json.loads(prompt.split("\n")[-1])
            return json.dumps(
                {
                    **{key: True for key in payload["checks"]},
                    "reason": "Reviewed fixture claims",
                }
            )
        if prompt.startswith("Select only answer-bearing"):
            return '{"spans": [0]}'
        raise AssertionError(f"Unexpected verifier prompt: {prompt}")


@pytest.fixture
def verifier(monkeypatch):
    from evlab.generation import pipeline, false_answers

    client = VerifierClient()
    monkeypatch.setattr(pipeline, "build_verifier", lambda config: client)
    monkeypatch.setattr(false_answers, "build_verifier", lambda config: client)
    return client


@pytest.fixture
def certify(verifier):
    def attach(candidate):
        target = candidate["false_answer"]
        candidate["false_answer_record"] = {
            "verification": verify_answer(verifier, candidate, target)
        }
        return candidate

    return attach
