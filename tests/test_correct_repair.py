from pathlib import Path
import json

import pytest

from evlab.cli import main
from evlab.generation import correct_repair
from evlab.generation.manifest import read_manifest
from evlab.generation.registry import update_registry
from evlab.io import read_jsonl, write_json, write_jsonl


def setup(tmp_path):
    originals = [
        {"_id": "d1", "title": "Season one", "text": "There were 23 episodes."},
        {"_id": "d2", "title": "Season two", "text": "There were 10 episodes."},
        {"_id": "d3", "title": "7 episodes", "text": "The series aired in winter."},
    ]
    baseline = [
        {"_id": "d1", "title": "Season one", "text": "The season aired on television."},
        {
            "_id": "d2",
            "title": "Season two",
            "text": "Ten weeks saw 10 episodes broadcast.",
        },
        {"_id": "d3", "title": "7 episodes", "text": "This was a winter series."},
    ]
    source = tmp_path / "human.jsonl"
    nq = tmp_path / "nqplus"
    old = tmp_path / "baseline/corpus_0001.jsonl"
    write_jsonl(source, originals)
    write_jsonl(old, baseline)
    update_registry(
        old.parent,
        lambda registry: registry["files"].update({old.name: {"condition": "correct"}}),
    )
    write_jsonl(
        nq / "queries.jsonl",
        [
            {
                "_id": f"q{i}",
                "text": f"How many episodes in season {i}?",
                "metadata": {
                    "answer": gold,
                    "answer_aliases": [gold],
                    "docs_with_answer": [doc["_id"]],
                    "relevant_docs": [doc["_id"]],
                    "usable_for": {"generation": True},
                },
            }
            for i, (doc, gold) in enumerate(zip(originals, ["23", "10", "7"]), 1)
        ],
    )
    return {
        "data": {"nqplus_dir": str(nq), "human_corpus": str(source)},
        "generation": {
            "repair_from": str(old),
            "output_dir": str(tmp_path / "repair"),
            "cache_path": str(tmp_path / "cache.sqlite"),
            "correct_max_attempts": 3,
            "validation": {"mode": "deterministic"},
            "llm": {"provider": "abstain", "model": "test"},
        },
    }


def test_dry_run_selects_only_lost_evidence_and_flags_unsupported_source(
    tmp_path, monkeypatch
):
    config = setup(tmp_path)
    monkeypatch.setattr(
        correct_repair,
        "build_llm_client",
        lambda cfg: pytest.fail("Dry-run must not construct a client"),
    )
    result = correct_repair.run_correct_repair(config, dry_run=True)
    assert result["checked_documents"] == 3
    assert result["missing_gold_in_generated_body"] == 2
    assert result["repairable_documents"] == 1
    assert result["needs_review"] == [
        {
            "query_id": "q3",
            "doc_id": "d3",
            "reason": "gold_alias_missing_from_original_body",
        }
    ]
    assert not (tmp_path / "repair").exists()


def test_repair_retries_retains_originals_and_resumes(tmp_path, monkeypatch):
    config = setup(tmp_path)
    old = Path(config["generation"]["repair_from"])
    before = old.read_bytes()
    prompts = []
    seeds = []

    class Client:
        model = provider = "test"

        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            seeds.append(kwargs["seed"])
            assert "There were 23 episodes." in prompt
            assert "['23']" in prompt
            assert "Question: How many episodes" in prompt
            return (
                "A televised season."
                if len(prompts) == 1
                else "A total of 23 episodes were broadcast."
            )

    monkeypatch.setattr(correct_repair, "build_llm_client", lambda cfg: Client())
    result = correct_repair.run_correct_repair(config)
    assert result["generated"] == 1 and result["rejects"] == 0
    assert len(prompts) == 2 and seeds[0] != seeds[1]
    assert "missing_gold_answer" in prompts[1]
    assert old.read_bytes() == before
    corpus = next((tmp_path / "repair").glob("corpus_*.jsonl"))
    assert read_jsonl(corpus) == [
        {
            "_id": "d1",
            "title": "Season one",
            "text": "A total of 23 episodes were broadcast.",
        }
    ]
    receipt = read_manifest(corpus)["d1"]
    assert receipt["generation_attempts"] == 2
    assert receipt["validation"]["semantic_verified"] is False
    assert receipt["validation"]["validation_goal"] == "query_answerability"
    assert receipt["validation"]["answerability_verified"] is False
    assert receipt["replaced_text"] == "The season aired on television."
    assert correct_repair.run_correct_repair(config)["reused"] == 1
    assert len(prompts) == 2
    originals = read_jsonl(config["data"]["human_corpus"])
    originals[0]["text"] += " It was popular."
    write_jsonl(config["data"]["human_corpus"], originals)
    with pytest.raises(ValueError, match="Stale generated document"):
        correct_repair.run_correct_repair(config)


def test_exhausted_budget_never_saves_invalid_copy(tmp_path, monkeypatch):
    config = setup(tmp_path)
    config["generation"]["correct_max_attempts"] = 2
    calls = []

    class Client:
        provider = model = "test"

        def generate(self, prompt, **kwargs):
            calls.append(prompt)
            return "No episode count here."

    monkeypatch.setattr(correct_repair, "build_llm_client", lambda cfg: Client())
    result = correct_repair.run_correct_repair(config)
    assert result["rejects"] == 1 and result["generated"] == 0 and len(calls) == 2
    assert not read_jsonl(next((tmp_path / "repair").glob("corpus_*.jsonl")))
    assert (
        len(read_jsonl(tmp_path / "repair/correct_repair_rejects.jsonl")[0]["attempts"])
        == 2
    )


def test_cli_audit(tmp_path, capsys):
    config = setup(tmp_path)
    path = tmp_path / "config.json"
    write_json(path, config)
    main(
        ["repair-correct-documents", "--config", str(path), "--dry-run", "--limit", "1"]
    )
    assert json.loads(capsys.readouterr().out)["selected_documents"] == 1


def test_baseline_directory_cannot_be_used_as_repair_output(tmp_path):
    config = setup(tmp_path)
    config["generation"]["output_dir"] = str(
        Path(config["generation"]["repair_from"]).parent
    )
    with pytest.raises(ValueError, match="separate"):
        correct_repair.run_correct_repair(config, dry_run=True)


def test_all_correct_selection_does_not_depend_on_old_corpus(tmp_path, monkeypatch):
    config = setup(tmp_path)
    Path(config["generation"].pop("repair_from")).unlink()
    monkeypatch.setattr(
        correct_repair, "build_llm_client", lambda cfg: pytest.fail("dry-run")
    )
    result = correct_repair.run_correct_generation(config, dry_run=True)
    assert result["scope"] == "all_answer_documents"
    assert result["checked_documents"] == result["selected_documents"] == 3
    assert result["title_only_answer_documents"] == 1
    assert result["needs_review"] == []
    assert not (tmp_path / "repair").exists()


def test_all_correct_generation_preserves_ids_titles_and_includes_title_context(
    tmp_path, monkeypatch
):
    config = setup(tmp_path)
    baseline = Path(config["generation"].pop("repair_from"))
    before = baseline.read_bytes()
    prompts = []

    class Client:
        provider = model = "test"

        def generate(self, prompt, **kwargs):
            prompts.append(prompt)
            if "Original Title: Season one" in prompt:
                return "A total of 23 episodes were broadcast."
            if "Original Title: Season two" in prompt:
                return "A total of 10 episodes were broadcast."
            assert "Original Title: 7 episodes" in prompt
            assert "The series aired in winter." in prompt
            assert "['7']" in prompt
            return "The winter series consisted of 7 episodes."

    monkeypatch.setattr(correct_repair, "build_llm_client", lambda cfg: Client())
    result = correct_repair.run_correct_generation(config)
    assert result["generated"] == 3 and result["rejects"] == 0
    assert baseline.read_bytes() == before
    corpus = next((tmp_path / "repair").glob("corpus_*.jsonl"))
    rows = read_jsonl(corpus)
    assert [row["_id"] for row in rows] == ["d1", "d2", "d3"]
    assert [row["title"] for row in rows] == ["Season one", "Season two", "7 episodes"]
    assert all(set(row) == {"_id", "title", "text"} for row in rows)
    receipts = read_manifest(corpus)
    assert receipts["d3"]["answer_in_original_body"] is False
    assert receipts["d1"]["replaced_row_hash"] is None
    assert receipts["d1"]["validation"]["semantic_verified"] is False
    assert correct_repair.run_correct_generation(config)["reused"] == 3
    assert len(prompts) == 3


def test_all_correct_flags_missing_answer_context_instead_of_inventing_it(tmp_path):
    config = setup(tmp_path)
    originals = read_jsonl(config["data"]["human_corpus"])
    originals[2]["title"] = "Winter series"
    write_jsonl(config["data"]["human_corpus"], originals)
    result = correct_repair.run_correct_generation(config, dry_run=True)
    assert result["selected_documents"] == 2
    assert result["needs_review"] == [
        {
            "query_id": "q3",
            "doc_id": "d3",
            "reason": "gold_alias_missing_from_original_title_and_body",
        }
    ]


def test_cli_all_correct_generation_audit(tmp_path, capsys):
    config = setup(tmp_path)
    config["generation"].pop("repair_from")
    path = tmp_path / "config.json"
    write_json(path, config)
    main(
        [
            "generate-correct-documents",
            "--config",
            str(path),
            "--dry-run",
            "--limit",
            "2",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["selected_documents"] == 2 and result["eligible_documents"] == 3


def test_correct_retries_copy_then_missing_answer_with_specific_feedback(
    tmp_path, monkeypatch
):
    config = setup(tmp_path)
    config["generation"].pop("repair_from")
    config["generation"]["correct_retry_temperature_step"] = 0.2
    calls = []

    class Client:
        provider = model = "test"

        def generate(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            assert "Correct answer from the source: 23" in prompt
            if len(calls) == 1:
                return "There were 23 episodes."
            if len(calls) == 2:
                assert "Reorganize the sentence structure" in prompt
                assert "Previous failed rewrite" not in prompt
                assert "All original requirements still apply" in prompt
                return "The complete season was broadcast."
            assert "A surname or pronoun alone is insufficient" in prompt
            assert "The complete season was broadcast." not in prompt
            return "A total of 23 episodes were broadcast."

    monkeypatch.setattr(correct_repair, "build_llm_client", lambda cfg: Client())
    result = correct_repair.run_correct_generation(config, limit=1)
    assert result["generated"] == 1 and result["rejects"] == 0
    assert [call[1]["temperature"] for call in calls] == pytest.approx([0.3, 0.5, 0.7])
    assert len({call[1]["seed"] for call in calls}) == 3
    corpus = next((tmp_path / "repair").glob("corpus_*.jsonl"))
    receipt = read_manifest(corpus)["d1"]
    assert [item["reason"] for item in receipt["failed_attempts"]] == [
        "near_duplicate_of_source",
        "missing_gold_answer",
    ]
    assert receipt["failed_attempts"][0]["similarity"] == 1
    assert receipt["matched_aliases"] == ["23"]
    assert receipt["temperature"] == pytest.approx(0.7)


def test_copied_source_remains_rejected_at_every_temperature(tmp_path, monkeypatch):
    config = setup(tmp_path)
    calls = []

    class Client:
        provider = model = "test"

        def generate(self, prompt, **kwargs):
            calls.append(kwargs["temperature"])
            return "There were 23 episodes."

    monkeypatch.setattr(correct_repair, "build_llm_client", lambda cfg: Client())
    result = correct_repair.run_correct_generation(config, limit=1)
    assert result["generated"] == 0 and result["rejects"] == 1
    assert len(calls) == 3
    assert not read_jsonl(next((tmp_path / "repair").glob("corpus_*.jsonl")))


@pytest.mark.parametrize("step", [-0.1, float("inf"), float("nan")])
def test_invalid_retry_temperature_step_rejected(tmp_path, step):
    config = setup(tmp_path)
    config["generation"]["correct_retry_temperature_step"] = step
    with pytest.raises(ValueError, match="correct_retry_temperature_step"):
        correct_repair.run_correct_generation(config, dry_run=True)


@pytest.mark.parametrize("failed_check", [None, "supports_gold", "unambiguous_answer"])
def test_answerability_judge_uses_question_specific_checks(failed_check):
    from evlab.generation.semantic import ModelOutputError

    class Judge:
        provider = model = "test"

        def generate(self, prompt, **kwargs):
            payload = json.loads(prompt.split("\n")[-1])
            assert set(payload["checks"]) == {
                "source_supports_gold",
                "supports_gold",
                "unambiguous_answer",
            }
            assert payload["evidence"]["validation_goal"] == "query_answerability"
            assert payload["evidence"]["question"] == "How many episodes?"
            assert "bare mention" in payload["checks"]["supports_gold"]
            assert "unrelated" in payload["checks"]["unambiguous_answer"]
            verdict = {name: name != failed_check for name in payload["checks"]}
            return json.dumps({**verdict, "reason": "Controlled test verdict"})

    # The positive case changes an irrelevant background detail. The negative
    # cases still contain the gold string, but lose or contradict its relation.
    generated = {
        None: "23 episodes aired in spring.",
        "supports_gold": "23 actors appeared in 24 episodes.",
        "unambiguous_answer": "There were 23 episodes. There were 24 episodes.",
    }[failed_check]
    evidence = {
        "question": "How many episodes?",
        "gold": "23",
        "aliases": ["23"],
        "source": "23 episodes aired in winter.",
        "generated": generated,
    }
    if failed_check:
        with pytest.raises(ModelOutputError):
            correct_repair.verify_correct_answerability(
                Judge(), evidence=evidence, stage="correct_generation"
            )
    else:
        receipt = correct_repair.verify_correct_answerability(
            Judge(), evidence=evidence, stage="correct_generation"
        )
        assert receipt["semantic_verified"] is True
