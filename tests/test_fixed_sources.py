from pathlib import Path

import pytest

from evlab.corpus.builder import build_corpus_with_report
from evlab.generation.output import GenerationOutput
from evlab.io import write_jsonl


def test_independent_generated_sources_preserve_duplicates_and_filter_targets(tmp_path):
    fixtures = Path(__file__).parent / "fixtures"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    write_jsonl(dataset / "queries.jsonl", [{"_id": "test1", "text": "How many episodes?"}])
    (dataset / "qrels").mkdir()
    (dataset / "qrels/test.tsv").write_text("query-id\tcorpus-id\tscore\ntest1\tdoc6\t1\n")
    write_jsonl(dataset / "answers.jsonl", [{"_id": "test1", "answer": "23", "false_answer": "25"}])
    fixed = {}
    for name, target in [("first", "25"), ("second", "25"), ("old", "24")]:
        out = GenerationOutput(tmp_path / name, {}, "test", {"false"})
        out.append({"condition": "false", "sample_idx": 0, "provenance_root_id": "doc6",
                    "title": "Chicago Fire", "text": f"Season four has {target} episodes.",
                    "validation_receipt": {"query_id": "test1", "target_answer": target}})
        fixed[name] = {"kind": "generated", "condition": "false",
                       "path": str(tmp_path / name / "corpus_0001.jsonl"),
                       "compatible_only": True}
    attack = tmp_path / "attack.jsonl"
    write_jsonl(attack, [{"_id": "atk_test1", "query_id": "test1", "text": "25 episodes"}])
    fixed["gpt_attack"] = {"kind": "attack", "condition": "false", "path": str(attack)}
    config = {"data": {"nqplus_dir": str(dataset),
                       "human_corpus": str(fixtures / "corpus/human.jsonl")},
              "corpus": {"fixed": fixed}}
    docs, report = build_corpus_with_report(config)
    generated = [d for d in docs if d.origin == "generated"]
    assert len(generated) == 3
    assert len({d.doc_id for d in docs}) == len(docs)
    assert sum(d.root_id == "doc6" for d in generated) == 2
    assert report["sources"]["old"]["exclusions"][0]["reason"] == "incompatible_false_target"
    assert next(d for d in generated if d.metadata.get("attack")).root_id == "atk_test1"
    config["corpus"]["include_human"] = False
    standalone, standalone_report = build_corpus_with_report(config)
    assert len(standalone) == 3
    assert all(doc.origin == "generated" for doc in standalone)
    assert standalone_report["human_documents"] == 0
    assert standalone_report["include_human"] is False
    fixed["old"]["compatible_only"] = False
    with pytest.raises(ValueError, match="incompatible_false_target"):
        build_corpus_with_report(config)


def test_named_fixed_source_requires_condition():
    fixtures = Path(__file__).parent / "fixtures"
    config = {"data": {"nqplus_dir": str(fixtures / "nqplus"),
                       "human_corpus": str(fixtures / "corpus/human.jsonl")},
              "corpus": {"fixed": {"untyped": {"path": "unused"}}}}
    with pytest.raises(ValueError, match="needs condition"):
        build_corpus_with_report(config)
