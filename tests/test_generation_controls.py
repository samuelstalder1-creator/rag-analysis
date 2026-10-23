import json
from copy import deepcopy

import pytest

from evlab.generation.false_answer import (
    generate_false_answer,
    is_valid_false_answer,
    parse_candidate,
)
from evlab.generation.minimal_edit import deterministic_minimal_edit, edit_answer_spans
from evlab.generation.semantic import judge, verify_answer, verify_document
from evlab.generation.pipeline import run_generation_pipeline
from evlab.generation.manifest import read_manifest, manifest_path
from evlab.generation.false_answers import run_false_answer_generation
from evlab.corpus.builder import load_generated_docs
from evlab.io import read_jsonl, write_jsonl
from evlab.evaluation.answers import classify_answer
from evlab.rag.reader import parse_reader_response


@pytest.mark.parametrize(
    "response",
    [
        "[]",
        "null",
        "{}",
        '{"candidate":null}',
        '{"candidate":["24","25"]}',
        '{"candidate":"24","extra":"25"}',
        '{"candidates":["24","25"]}',
        '{"candidates":[null]}',
        '{"candidates":[{}]}',
        '{"candidates":[true]}',
    ],
)
def test_reject_malformed_candidate_shapes(response):
    with pytest.raises(ValueError):
        parse_candidate(response)


@pytest.mark.parametrize(
    "target,gold,aliases",
    [
        ("7", "2013", ["2013"]),
        ("023", "23", ["23"]),
        ("23.0", "23", ["23"]),
        ("2013-03-31", "March 31, 2013", ["March 31, 2013"]),
        ("one", "1", ["1"]),
        ("None", "Shakespeare", ["Shakespeare"]),
        ("a different answer", "Shakespeare", ["Shakespeare"]),
    ],
)
def test_reject_equivalents_wrong_types_and_placeholders(target, gold, aliases):
    assert not is_valid_false_answer(target, gold, aliases)


def test_edit_preserves_date_format_and_does_not_reedit_replacement():
    assert (
        deterministic_minimal_edit(
            "March 31, 2013 and 2013", ["2013", "March 31, 2013"], "2012"
        )
        == "March 31, 2012 and 2012"
    )
    assert deterministic_minimal_edit("Ann", ["Ann"], "Ann Marie") == "Ann Marie"
    assert deterministic_minimal_edit("Ann", ["Ann"], r"A\1") == r"A\1"


def test_question_aware_spans_preserve_unrelated_date(verifier):
    edited, edits = edit_answer_spans(
        verifier,
        text="There were 23 episodes, aired on May 23.",
        aliases=["23"],
        target_answer="24",
        question="How many episodes?",
    )
    assert edited == "There were 24 episodes, aired on May 23."
    assert edits == [{"start": 11, "end": 13, "text": "23", "replacement": "24"}]


@pytest.mark.parametrize(
    "response", ['{"spans":[]}', '{"spans":[true]}', '{"spans":[0,0]}', '{"spans":[9]}']
)
def test_bad_span_selection_fails_closed(response):
    class Client:
        def generate(self, *args, **kwargs):
            return response

    with pytest.raises(ValueError):
        edit_answer_spans(
            Client(),
            text="23 episodes",
            aliases=["23"],
            target_answer="24",
            question="How many?",
        )


@pytest.mark.parametrize("value", [False, "true", None])
def test_semantic_verification_requires_literal_true(value):
    class Client:
        model = provider = "test"

        def generate(self, *args, **kwargs):
            return json.dumps({"supports_target": value, "reason": "Not affirmative"})

    with pytest.raises(ValueError):
        judge(
            Client(),
            stage="false_seed",
            checks={"supports_target": "affirmative support"},
            evidence={},
        )


def test_negated_target_is_rejected_by_semantic_gate():
    class Client:
        model = provider = "test"

        def generate(self, prompt, **kwargs):
            payload = json.loads(prompt.split("\n")[-1])
            assert "did not have 24" in payload["evidence"]["generated"]
            return json.dumps(
                {
                    **{key: True for key in payload["checks"]},
                    "supports_target": False,
                    "reason": "Target is denied",
                }
            )

    with pytest.raises(ValueError, match="Target is denied"):
        verify_document(
            Client(),
            question="How many episodes?",
            gold="23",
            aliases=["23"],
            target="24",
            source="23 episodes",
            seed="24 episodes",
            generated="The season did not have 24 episodes.",
            condition="false",
            stage="paraphrase",
        )


def candidate():
    return {
        "query_id": "q",
        "query": "How many episodes?",
        "answer": "23",
        "answer_aliases": ["23"],
        "false_answer": "24",
        "h_docs": [
            {"doc_id": "d", "title": "Season", "text": "There were 23 episodes."}
        ],
    }


def config(tmp_path):
    return {
        "output_dir": str(tmp_path / "out"),
        "cache_path": str(tmp_path / "cache.sqlite"),
        "conditions": ["false"],
        "llm": {"provider": "abstain", "model": "test"},
    }


def document_client(monkeypatch, response="A total of 24 episodes were broadcast."):
    from evlab.generation import pipeline

    class Client:
        model = provider = "test"

        def generate(self, *args, **kwargs):
            return response

    monkeypatch.setattr(pipeline, "build_llm_client", lambda config: Client())


def test_manifest_restores_target_detects_stale_source_and_tampered_output(
    tmp_path, monkeypatch, certify
):
    document_client(monkeypatch)
    row = certify(candidate())
    cfg = config(tmp_path)
    assert run_generation_pipeline(cfg, [row]) == {"generated": 1, "rejects": 0}
    corpus = next((tmp_path / "out").glob("corpus_*.jsonl"))
    assert manifest_path(corpus).exists()
    receipt = read_manifest(corpus)["d"]
    assert receipt["edits"]["body"][0]["replacement"] == "24"
    loaded = load_generated_docs(tmp_path / "out", query_ids_by_doc={"d": "q"})[0]
    assert loaded.target_answer == "24" and loaded.gold_answer == "23"
    assert run_generation_pipeline(cfg, [row]) == {"generated": 0, "rejects": 0}
    changed = deepcopy(row)
    changed["h_docs"][0]["text"] += " It was televised."
    certify(changed)
    with pytest.raises(ValueError, match="Stale generated document"):
        run_generation_pipeline(cfg, [changed])
    rows = read_jsonl(corpus)
    rows[0]["text"] = "Tampered output."
    write_jsonl(corpus, rows)
    with pytest.raises(ValueError, match="Missing or stale validation manifest"):
        load_generated_docs(corpus)


def test_changed_target_invalidates_resume(tmp_path, monkeypatch, certify):
    document_client(monkeypatch)
    row = certify(candidate())
    cfg = config(tmp_path)
    run_generation_pipeline(cfg, [row])
    row["false_answer"] = "25"
    certify(row)
    with pytest.raises(ValueError, match="Stale generated document"):
        run_generation_pipeline(cfg, [row])


def test_retry_seed_keeps_completed_rows_and_bypasses_cached_failure(tmp_path, monkeypatch, certify):
    from evlab.generation import pipeline

    calls = []

    class Client:
        model = provider = "test"

        def generate(self, prompt, **kwargs):
            calls.append(kwargs["seed"])
            if "Across the season" in prompt:
                return ("Across this season, audiences watched a total of 24 episodes."
                        if kwargs["seed"] >= 10000000 else "There were no broadcasts.")
            return "A total of 24 episodes were broadcast."

    monkeypatch.setattr(pipeline, "build_llm_client", lambda config: Client())
    row = candidate()
    row["h_docs"].append({"doc_id": "d2", "title": "Season", "text": "Across the season, 23 episodes aired."})
    row = certify(row)
    cfg = {**config(tmp_path), "span_selection": "all", "validation": {"mode": "deterministic"}}
    assert run_generation_pipeline(cfg, [row]) == {"generated": 1, "rejects": 1}
    corpus = next((tmp_path / "out").glob("corpus_*.jsonl"))
    before = read_jsonl(corpus)[0]
    assert run_generation_pipeline(cfg, [row]) == {"generated": 0, "rejects": 0}
    assert len(calls) == 2  # Cached failed response; no new model call.
    assert run_generation_pipeline({**cfg, "retry_seed": 1}, [row]) == {"generated": 1, "rejects": 0}
    assert len(calls) == 3  # Only the rejected document is regenerated.
    assert read_jsonl(corpus)[0] == before
    assert len(read_jsonl(corpus)) == 2
    assert len(list((tmp_path / "out").glob("corpus_*.jsonl"))) == 1
    assert read_manifest(corpus)["d2"]["retry_seed"] == 1


def test_near_duplicate_and_failed_verification_never_write_document(
    tmp_path, monkeypatch, certify
):
    document_client(monkeypatch, "There were 24 episodes.")
    result = run_generation_pipeline(config(tmp_path), [certify(candidate())])
    assert result == {"generated": 0, "rejects": 1}
    assert all(
        not read_jsonl(path) for path in (tmp_path / "out").glob("corpus_*.jsonl")
    )
    rejects = read_jsonl(next((tmp_path / "out").glob("*rejects.jsonl")))
    assert rejects[0]["generated_text"] == "There were 24 episodes."
    assert "near_duplicate" in rejects[0]["reason"]


def test_unknown_verdict_never_saves_false_answer(tmp_path, monkeypatch):
    from evlab.generation import false_answers
    from test_false_answer_stages import setup_data

    root, cfg = setup_data(tmp_path)

    class Client:
        model = provider = "test"

        def generate(self, *args, **kwargs):
            return '{"candidate":"24"}'

    class Uncertain:
        model = provider = "test"

        def generate(self, *args, **kwargs):
            return '{"source_supports_gold":true,"same_type":true,"plausible":true,"incorrect":false,"reason":"Insufficient evidence"}'

    monkeypatch.setattr(false_answers, "build_llm_client", lambda config: Client())
    monkeypatch.setattr(false_answers, "build_verifier", lambda config: Uncertain())
    result = run_false_answer_generation(
        cfg, root, human_corpus=tmp_path / "human.jsonl"
    )
    assert result["rejects"] == 1
    assert not (tmp_path / "output/generated_false_answers.jsonl").exists()


def test_answer_classification_does_not_confuse_mention_with_adoption():
    assert classify_answer("Not 24", gold_aliases=["23"], target_answer="24") == "other"
    assert (
        classify_answer("23 or 24", gold_aliases=["23"], target_answer="24")
        == "ambiguous"
    )
    assert classify_answer("24", gold_aliases=["23"], target_answer="24") == "target"


@pytest.mark.parametrize(
    "response",
    [
        "[]",
        '{"answer":24,"citations":[]}',
        '{"answer":"24","citations":[true]}',
        '{"bad":1}',
    ],
)
def test_malformed_reader_response_is_recorded_as_invalid(response):
    assert parse_reader_response(response)["valid_json"] is False


def test_repeated_samples_are_rejected(tmp_path, monkeypatch, certify):
    document_client(monkeypatch)
    cfg = config(tmp_path)
    cfg["samples_per_h"] = 2
    assert run_generation_pipeline(cfg, [certify(candidate())]) == {
        "generated": 1,
        "rejects": 1,
    }
    rejects = read_jsonl(next((tmp_path / "out").glob("*rejects.jsonl")))
    assert rejects[0]["reason"] == "near_duplicate_of_other_sample"


def test_standard_abstention_is_recognized():
    assert (
        classify_answer("I don't know", gold_aliases=["23"], target_answer="24")
        == "abstain"
    )


def test_two_writers_do_not_duplicate_rows_or_lose_receipts(tmp_path):
    from evlab.generation.output import GenerationOutput

    a = GenerationOutput(tmp_path, {}, "test", {"correct"})
    b = GenerationOutput(tmp_path, {}, "test", {"correct"})

    def row(did):
        return {
            "condition": "correct",
            "sample_idx": 0,
            "provenance_root_id": did,
            "title": "",
            "text": did,
            "validation_receipt": {"fingerprint": did},
        }

    assert a.append(row("d1")) == 1
    assert b.append(row("d1")) == 0
    assert b.append(row("d2")) == 1
    a.update_counts()
    from evlab.generation.registry import read_registry

    assert read_registry(tmp_path)["files"]["corpus_0001.jsonl"]["document_count"] == 2
    assert set(read_manifest(tmp_path / "corpus_0001.jsonl")) == {"d1", "d2"}


@pytest.mark.parametrize("change", ["source", "gold"])
def test_experiment_loading_rejects_changed_source_or_gold(
    tmp_path, monkeypatch, certify, change
):
    from pathlib import Path
    from evlab.data.nqplus import load_generation_candidates
    from evlab.corpus.builder import build_corpus

    fixtures = Path(__file__).parent / "fixtures"
    queries = read_jsonl(fixtures / "nqplus/queries.jsonl")
    originals = read_jsonl(fixtures / "corpus/human.jsonl")
    write_jsonl(tmp_path / "nqplus/queries.jsonl", queries)
    write_jsonl(tmp_path / "human.jsonl", originals)
    row = load_generation_candidates(tmp_path / "nqplus", tmp_path / "human.jsonl")[0]
    row["false_answer"] = "24"
    document_client(monkeypatch)
    assert run_generation_pipeline(config(tmp_path), [certify(row)])["generated"] == 1
    cfg = {
        "data": {
            "nqplus_dir": str(tmp_path / "nqplus"),
            "human_corpus": str(tmp_path / "human.jsonl"),
        },
        "corpus": {"generated_docs": str(tmp_path / "out"), "k_false": 1},
    }
    assert any(doc.target_answer == "24" for doc in build_corpus(cfg))
    if change == "source":
        originals[0]["text"] += " A changed claim."
        write_jsonl(tmp_path / "human.jsonl", originals)
    else:
        queries[0]["metadata"].update(answer="25", answer_aliases=["25"])
        write_jsonl(tmp_path / "nqplus/queries.jsonl", queries)
    with pytest.raises(ValueError, match="Stale"):
        build_corpus(cfg)


def test_full_human_corpus_receipts_keep_relevance_mapping(tmp_path):
    from evlab.generation.output import GenerationOutput

    output = GenerationOutput(
        tmp_path,
        {"source": "human_corpus", "generation_signature": "test"},
        "test",
        {"correct"},
    )
    output.append(
        {
            "condition": "correct",
            "sample_idx": 0,
            "provenance_root_id": "doc6",
            "title": "Season",
            "text": "23 episodes",
            "validation_receipt": {"query_id": "doc6", "fingerprint": "test"},
        }
    )
    docs = load_generated_docs(tmp_path, query_ids_by_doc={"doc6": "test1"})
    assert docs[0].query_id == "test1"


def test_cache_separates_same_model_on_different_endpoints(tmp_path):
    from evlab.llm.cache import CachedLLMClient

    class Client:
        provider = "local_openai"
        model = "same-model"

        def __init__(self, base_url):
            self.base_url = base_url

        def generate(self, *args, **kwargs):
            return self.base_url

    path = tmp_path / "cache.sqlite"
    assert CachedLLMClient(Client("one"), path).generate("same prompt") == "one"
    assert CachedLLMClient(Client("two"), path).generate("same prompt") == "two"


def test_malformed_code_fence_is_invalid_reader_json():
    assert parse_reader_response("```json```")["valid_json"] is False


def test_validator_repairs_missing_reason_with_explicit_schema():
    class Client:
        provider = model = "test"

        def __init__(self):
            self.prompts = []

        def generate(self, prompt, **kwargs):
            self.prompts.append(prompt)
            assert '"required": ["supports_target", "reason"]' in prompt
            if len(self.prompts) == 1:
                return '{"supports_target":true}'
            assert "missing nonempty top-level reason string" in prompt
            return '{"supports_target":true,"reason":"The target is affirmatively supported."}'

    client = Client()
    receipt = judge(
        client,
        stage="test",
        checks={"supports_target": "affirmative support"},
        evidence={},
    )
    assert len(client.prompts) == 2
    assert client.prompts[0] != client.prompts[1]
    assert receipt["format_attempts"][0]["response"] == '{"supports_target":true}'


def test_validator_bad_format_retries_are_bounded_and_logged():
    from evlab.generation.semantic import ModelOutputError

    class Client:
        model = provider = "test"

        def __init__(self):
            self.calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return '{"supports_target":true}'

    client = Client()
    with pytest.raises(ModelOutputError) as caught:
        judge(client, stage="test", checks={"supports_target": "support"}, evidence={})
    assert client.calls == 3
    assert len(caught.value.attempts) == 3
    assert caught.value.attempts[-1]["response"] == '{"supports_target":true}'


def test_negative_semantic_verdict_is_not_retried_for_approval():
    from evlab.generation.semantic import ModelOutputError

    class Client:
        model = provider = "test"

        def __init__(self):
            self.calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return '{"supports_target":false,"reason":"The text denies the target."}'

    client = Client()
    with pytest.raises(ModelOutputError, match="semantic validation failed"):
        judge(client, stage="test", checks={"supports_target": "support"}, evidence={})
    assert client.calls == 1


def test_generator_helper_makes_exactly_one_call_and_returns_one_candidate():
    class Client:
        def __init__(self):
            self.prompts = []

        def generate(self, prompt, **kwargs):
            self.prompts.append(prompt)
            assert 'Return JSON only: {"candidate": "..."}' in prompt
            assert "Generate one plausible but incorrect answer" in prompt
            return '{"candidate":"24"}'

    client = Client()
    assert (
        generate_false_answer(
            client, question="How many episodes?", answer="23", aliases=["23"]
        )
        == "24"
    )
    assert len(client.prompts) == 1


def test_false_answer_failure_records_raw_attempts_in_reject_log(
    tmp_path, monkeypatch, verifier
):
    from test_false_answer_stages import setup_data
    from evlab.generation import false_answers

    root, cfg = setup_data(tmp_path)

    class Client:
        model = provider = "test"

        def generate(self, *args, **kwargs):
            return '{"candidate":"23"}'

    monkeypatch.setattr(false_answers, "build_llm_client", lambda config: Client())
    result = run_false_answer_generation(
        cfg, root, human_corpus=tmp_path / "human.jsonl", limit=1
    )
    assert result == {"generated": 0, "reused": 0, "rejects": 1}
    reject = read_jsonl(
        next((tmp_path / "output").glob("*false-answer-rejects.jsonl"))
    )[0]
    assert len(reject["attempts"]) == 3
    assert reject["attempts"][0]["response"] == '{"candidate":"23"}'


@pytest.mark.parametrize("relation", ["EQUIVALENT", "SUPPORTED", "UNKNOWN"])
def test_categorical_verifier_rejects_noncontradictions_even_with_positive_reason(
    relation,
):
    class Client:
        model = provider = "test"

        def generate(self, prompt, **kwargs):
            assert "CONTRADICTED = the evidence establishes" in prompt
            return json.dumps(
                {
                    "gold_support": "SUPPORTED",
                    "target_relation": relation,
                    "same_type": True,
                    "plausible": True,
                    "evidence_quote": "There were 23 episodes.",
                    "reason": "The target is factually incorrect.",
                }
            )

    with pytest.raises(ValueError, match="negative semantic verdict"):
        verify_answer(Client(), candidate(), "22")


def test_categorical_verifier_accepts_grounded_contradiction():
    class Client:
        model = provider = "test"

        def generate(self, prompt, **kwargs):
            assert '"target_relation"' in prompt
            assert '"incorrect"' not in prompt
            return json.dumps(
                {
                    "gold_support": "SUPPORTED",
                    "target_relation": "CONTRADICTED",
                    "same_type": True,
                    "plausible": True,
                    "evidence_quote": "There were 23 episodes.",
                    "reason": "The source specifies 23 episodes; 22 is a conflicting count.",
                }
            )

    receipt = verify_answer(Client(), candidate(), "22")
    assert receipt["judgment"]["target_relation"] == "CONTRADICTED"


def test_categorical_verifier_rejects_fabricated_evidence_quote():
    class Client:
        model = provider = "test"

        def generate(self, prompt, **kwargs):
            return json.dumps(
                {
                    "gold_support": "SUPPORTED",
                    "target_relation": "CONTRADICTED",
                    "same_type": True,
                    "plausible": True,
                    "evidence_quote": "The season had 23 episodes and not 22.",
                    "reason": "Conflicting episode count.",
                }
            )

    with pytest.raises(ValueError, match="unverified evidence quote"):
        verify_answer(Client(), candidate(), "22")


def test_old_inverted_boolean_output_is_not_silently_reinterpreted():
    class Client:
        model = provider = "test"

        def generate(self, prompt, **kwargs):
            return '{"source_supports_gold":true,"same_type":true,"plausible":true,"incorrect":false,"reason":"22 is factually incorrect."}'

    with pytest.raises(ValueError, match="invalid validator output"):
        verify_answer(Client(), candidate(), "22")


def test_distinct_candidates_are_checked_without_retrying_negative_target(
    tmp_path, monkeypatch, verifier
):
    from test_false_answer_stages import setup_data
    from evlab.generation import false_answers

    root, cfg = setup_data(tmp_path)
    targets = []
    prompts = []
    seeds = []

    class Generator:
        model = provider = "test"

        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            seeds.append(kwargs["seed"])
            assert "ONE WRONG multiple-choice distractor" in prompt
            if len(prompts) > 1:
                assert "Rejected attempts and feedback" in prompt
                assert "24" in prompt
            return json.dumps({"candidate": "24" if len(prompts) <= 2 else "25"})

    class Assessor:
        model = provider = "test"

        def generate(self, prompt, **kwargs):
            payload = json.loads(prompt.split("\n")[-1])
            target = payload["evidence"]["target"]
            targets.append(target)
            return json.dumps(
                {
                    "gold_support": "SUPPORTED",
                    "target_relation": "UNKNOWN" if target == "24" else "CONTRADICTED",
                    "same_type": True,
                    "plausible": True,
                    "evidence_quote": "There were 23 episodes.",
                    "reason": "Uncertain first alternative; grounded contrast for the second.",
                }
            )

    monkeypatch.setattr(false_answers, "build_llm_client", lambda config: Generator())
    monkeypatch.setattr(false_answers, "build_verifier", lambda config: Assessor())
    result = run_false_answer_generation(
        cfg, root, human_corpus=tmp_path / "human.jsonl", limit=1
    )
    assert result == {"generated": 1, "reused": 0, "rejects": 0}
    assert targets == ["24", "25"]
    assert len(prompts) == 3
    assert len(set(seeds)) == 3
    saved = read_jsonl(tmp_path / "output/generated_false_answers.jsonl")[0]
    assert saved["false_answer"] == "25"
    assert saved["verification"]["rejected_candidates"][0]["target"] == "24"
    assert (
        saved["verification"]["rejected_candidates"][1]["error"]
        == "Repeated previously rejected target"
    )
    assert saved["verification"]["generation_attempts"] == 3


def test_one_retry_budget_covers_structural_and_semantic_rejection(
    tmp_path, monkeypatch, verifier
):
    from test_false_answer_stages import setup_data
    from evlab.generation import false_answers
    from evlab.generation.semantic import ModelOutputError

    root, cfg = setup_data(tmp_path)
    cfg["false_answer_max_attempts"] = 2
    prompts = []
    verified = []

    class Generator:
        model = provider = "test"

        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            if len(prompts) == 2:
                assert "gold-equivalent" in prompt
                assert '"target": "23"' in prompt
            return json.dumps({"candidate": "23" if len(prompts) == 1 else "24"})

    def reject(verifier, candidate, target):
        verified.append(target)
        raise ModelOutputError(
            "Unknown target relation", [{"response": "UNKNOWN", "error": "uncertain"}]
        )

    monkeypatch.setattr(false_answers, "build_llm_client", lambda config: Generator())
    monkeypatch.setattr(false_answers, "verify_answer", reject)
    result = run_false_answer_generation(
        cfg, root, human_corpus=tmp_path / "human.jsonl", limit=1
    )
    assert result == {"generated": 0, "reused": 0, "rejects": 1}
    assert len(prompts) == 2 and verified == ["24"]
    assert not (tmp_path / "output/generated_false_answers.jsonl").exists()
    reject_row = read_jsonl(
        next((tmp_path / "output").glob("*false-answer-rejects.jsonl"))
    )[0]
    assert [item["stage"] for item in reject_row["attempts"]] == [
        "generation",
        "validation",
    ]
    assert "after 2 generation attempts" in reject_row["reason"]


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "3"])
def test_invalid_generation_budget_fails_before_data_or_model_access(tmp_path, budget):
    cfg = config(tmp_path)
    cfg["false_answer_max_attempts"] = budget
    with pytest.raises(ValueError, match="positive integer"):
        run_false_answer_generation(cfg, tmp_path / "missing-data")


def test_generator_does_not_retry_batch_output_inside_single_call():
    class Client:
        def __init__(self):
            self.calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return '{"candidates":["24","25","26"]}'

    client = Client()
    with pytest.raises(ValueError, match="exactly one"):
        generate_false_answer(client, question="How many?", answer="23", aliases=["23"])
    assert client.calls == 1
