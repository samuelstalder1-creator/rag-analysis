import csv
import json
from pathlib import Path

import pytest

from evlab.cli import main
from evlab.data.nqopen_plus import build_nqopen_plus
from evlab.data.nqplus import load_generation_candidates, load_rag_queries
from evlab.generation.false_answers import (
    attach_saved_answers,
    run_false_answer_generation,
    valid_saved_answer,
)
from evlab.generation.false_documents import run_false_document_generation
from evlab.generation.manifest import read_manifest
from evlab.io import file_sha256, read_jsonl, stable_hash, write_jsonl, write_json


def fixture(tmp_path):
    source = tmp_path / "source"
    docs = [
        {"_id": "d1", "title": "Season", "text": "There were 23 episodes."},
        {"_id": "d2", "title": "Author", "text": "Tom wrote the novel."},
    ]
    queries = [
        {
            "_id": "q1",
            "text": "How many episodes?",
            "metadata": {
                "answer": "23",
                "answer_aliases": ["23"],
                "relevant_docs": ["d1"],
                "docs_with_answer": ["d1"],
                "usable_for": {"generation": True, "reader": True, "retrieval": True},
                "custom": "preserved",
            },
        },
        {
            "_id": "q2",
            "text": "Who wrote the novel?",
            "metadata": {
                "answer": "Tom",
                "answer_aliases": ["Tom"],
                "relevant_docs": ["d2"],
                "docs_with_answer": ["d2"],
                "usable_for": {"generation": False, "reader": True, "retrieval": True},
            },
        },
        {
            "_id": "q3",
            "text": "Excluded?",
            "metadata": {"usable_for": {"generation": False}},
        },
    ]
    write_jsonl(source / "queries.jsonl", queries)
    write_jsonl(
        source / "generated_labels.jsonl",
        [
            {
                "_id": "q2",
                "source_hash": stable_hash(queries[1]),
                "metadata": {
                    "usable_for": {
                        "generation": True,
                        "reader": True,
                        "retrieval": True,
                    }
                },
            }
        ],
    )
    write_jsonl(
        tmp_path / "cocktail.jsonl",
        [{"_id": r["_id"], "text": r["text"], "metadata": {}} for r in queries],
    )
    write_jsonl(tmp_path / "human.jsonl", docs)
    (tmp_path / "qrels.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td2\t1\n"
    )
    with (tmp_path / "nqopen.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "best_answer",
                "correct_answers",
                "incorrect_answers",
                "category",
            ],
        )
        w.writeheader()
        w.writerows(
            [
                dict(
                    question="How many episodes?",
                    best_answer="23",
                    correct_answers="23",
                    incorrect_answers="24",
                    category="raw preserved",
                ),
                dict(
                    question="Who wrote the novel?",
                    best_answer="Tom",
                    correct_answers="Tom; Thomas",
                    incorrect_answers="Thomas",
                    category="raw preserved",
                ),
            ]
        )
    config = {
        "data": {
            "nqplus_dir": str(tmp_path / "nqopen+"),
            "human_corpus": str(tmp_path / "human.jsonl"),
        },
        "dataset_build": {
            "source_nqplus_dir": str(source),
            "cocktail_queries": str(tmp_path / "cocktail.jsonl"),
            "cocktail_qrels": str(tmp_path / "qrels.tsv"),
            "nqopen_csv": str(tmp_path / "nqopen.csv"),
            "expected_questions": 2,
        },
    }
    gen = {
        "false_answer_source": "nqopen_import",
        "false_answers_file": str(tmp_path / "nqopen+/false_answers.jsonl"),
        "output_dir": str(tmp_path / "output"),
        "cache_path": str(tmp_path / "cache.sqlite"),
        "span_selection": "all",
        "skip_missing_false_answers": True,
        "validation": {"mode": "deterministic"},
        "llm": {"provider": "abstain", "model": "test/import"},
        "progress_every": 0,
    }
    return config, gen


def test_build_preserves_scope_gold_raw_information_and_qrels(tmp_path):
    config, gen = fixture(tmp_path)
    stats = build_nqopen_plus(config)
    assert (
        stats["questions"] == 2
        and stats["ready_false_answers"] == 1
        and stats["needs_review"] == 1
    )
    root = Path(config["data"]["nqplus_dir"])
    rows = read_jsonl(root / "queries.jsonl")
    assert [r["_id"] for r in rows] == ["q1", "q2"]
    assert rows[0]["metadata"]["custom"] == "preserved"
    assert rows[0]["metadata"]["answer_aliases"] == ["23"]
    assert rows[0]["metadata"]["nqopen"]["raw"]["category"] == "raw preserved"
    assert rows[1]["metadata"]["false_answer"] is None
    assert rows[1]["metadata"]["false_answer_candidates"] == [
        {"text": "Thomas", "status": "gold_equivalent"}
    ]
    assert len(load_rag_queries(root)) == 2
    assert len(load_generation_candidates(root, config["data"]["human_corpus"])) == 2
    assert (root / "qrels/test.tsv").read_text() == (tmp_path / "qrels.tsv").read_text()
    assert len(read_jsonl(root / "corpus/relevant.jsonl")) == 2
    snapshots = {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }
    build_nqopen_plus(config)
    assert snapshots == {
        p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
    }


def test_build_rejects_ambiguous_or_missing_match_without_partial_output(tmp_path):
    config, _ = fixture(tmp_path)
    path = tmp_path / "nqopen.csv"
    path.write_text(path.read_text() + path.read_text().splitlines()[1] + "\n")
    with pytest.raises(ValueError, match="unique NQ-Open"):
        build_nqopen_plus(config)
    assert not Path(config["data"]["nqplus_dir"]).exists()


def test_expected_scope_guard(tmp_path):
    config, _ = fixture(tmp_path)
    config["dataset_build"]["expected_questions"] = 3
    with pytest.raises(ValueError, match="Expected 3"):
        build_nqopen_plus(config)


def test_imported_targets_generate_documents_without_answer_model_calls(
    tmp_path, monkeypatch
):
    config, gen = fixture(tmp_path)
    build_nqopen_plus(config)
    root = config["data"]["nqplus_dir"]
    human = config["data"]["human_corpus"]
    before = file_sha256(gen["false_answers_file"])
    calls = []

    class Client:
        provider = "test"
        model = "test/import"

        def generate(self, prompt, **kwargs):
            assert prompt.startswith("FALSE_EVIDENCE_GENERATION")
            assert "Target answer for this counterfactual passage: 24" in prompt
            calls.append(prompt)
            return "The season consisted of 24 episodes in total."

    from evlab.generation import pipeline

    monkeypatch.setattr(pipeline, "build_llm_client", lambda cfg: Client())
    dry = run_false_document_generation(gen, root, human_corpus=human, dry_run=True)
    assert dry["total_questions"] == 2 and dry["ready_questions"] == 1
    assert dry["blocked_query_ids"] == ["q2"]
    assert calls == [] and not (tmp_path / "output").exists()
    result = run_false_document_generation(gen, root, human_corpus=human)
    assert result["generated"] == 1 and result["skipped_questions"] == 1
    assert len(calls) == 1
    assert file_sha256(gen["false_answers_file"]) == before
    corpus = next((tmp_path / "output").glob("corpus_*.jsonl"))
    receipt = read_manifest(corpus)["d1"]
    assert (
        receipt["target_answer"] == "24"
        and receipt["false_answer_origin"] == "nqopen_import"
    )
    assert receipt["validation"]["semantic_verified"] is False
    with pytest.raises(ValueError, match="read-only"):
        run_false_answer_generation(gen, root, human_corpus=human)
    assert file_sha256(gen["false_answers_file"]) == before


def test_stale_sources_and_semantic_mode_cannot_use_import_receipts(tmp_path):
    config, gen = fixture(tmp_path)
    build_nqopen_plus(config)
    root = config["data"]["nqplus_dir"]
    human = config["data"]["human_corpus"]
    candidates = attach_saved_answers(load_generation_candidates(root, human), gen)
    assert valid_saved_answer(candidates[0], require_semantic=False)
    assert not valid_saved_answer(candidates[0], require_semantic=True)
    docs = read_jsonl(human)
    docs[0]["text"] += " New source fact."
    write_jsonl(human, docs)
    result = run_false_document_generation(gen, root, human_corpus=human, dry_run=True)
    assert result["ready_questions"] == 0


def test_cli_build_and_dry_run(tmp_path, capsys):
    config, gen = fixture(tmp_path)
    config["generation"] = gen
    path = tmp_path / "config.json"
    write_json(path, config)
    main(["data", "build-nqopen-plus", "--config", str(path)])
    assert json.loads(capsys.readouterr().out)["questions"] == 2
    main(["generate-false-documents", "--config", str(path), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["ready_questions"] == 1


def test_reader_uses_imported_target_in_human_baseline_and_rejects_conflicts(
    tmp_path, monkeypatch
):
    from evlab.experiments import runner
    from evlab.schema import Doc

    config, _ = fixture(tmp_path)
    build_nqopen_plus(config)
    queries = load_rag_queries(config["data"]["nqplus_dir"])
    monkeypatch.setattr(
        runner,
        "answer_with_hits",
        lambda *args, **kwargs: {
            "answer": "24" if kwargs["query"].query_id == "q1" else "Tom",
            "valid_json": True,
            "context_doc_ids": [],
            "citations": [],
        },
    )
    reader_config = {
        "top_k": 1,
        "temperature": 0.0,
        "seed": 13,
        "prompt_template": "test",
    }
    rows, metrics = runner._evaluate_reader(None, queries, {}, {}, reader_config)
    assert rows[0]["target_answer"] == "24" and rows[0]["label"] == "target"
    assert rows[1]["target_answer"] is None and rows[1]["label"] == "correct"
    assert metrics["answer_target_query_count"] == 1
    assert metrics["answer_target_rate_on_targeted_queries"] == 1
    conflict = Doc(
        doc_id="false::d1",
        title="Season",
        text="25 episodes",
        origin="generated",
        query_id="q1",
        condition="false",
        target_answer="25",
    )
    with pytest.raises(ValueError, match="Conflicting false targets"):
        runner._evaluate_reader(
            None, queries, {}, {conflict.doc_id: conflict}, reader_config
        )
