from pathlib import Path

import pytest

from evlab.experiments.fingerprint import changed_inputs, compute_fingerprints
from evlab.experiments.runner import run_experiment
from evlab.io import read_jsonl
from evlab.rag.config import resolve_reader
from evlab.rag.pipeline import answer_with_hits
from evlab.schema import Doc, Hit, Query


class Reader:
    model = provider = "test"

    def generate(self, prompt, **kwargs):
        self.prompt = prompt
        return '{"answer": "23", "citations": [1]}'


def test_reverse_only_reorders_selected_hits_and_renumbers_context():
    docs = {str(i): Doc(str(i), f"Title {i}", f"Body {i}", "human") for i in range(1, 5)}
    hits = [Hit("q", str(i), i, 5-i, "bm25") for i in range(1, 5)]
    reader = Reader()
    result = answer_with_hits(reader, query=Query("q", "Question"), hits=hits,
                              docs_by_id=docs, top_k=3, context_order="reverse")
    assert result["context_doc_ids"] == ["3", "2", "1"]
    assert result["context_ir_ranks"] == [3, 2, 1]
    assert "[1] Title 3\nBody 3" in reader.prompt
    assert "Title 4" not in reader.prompt
    assert [hit.rank for hit in hits] == [1, 2, 3, 4]
    assert answer_with_hits(reader, query=Query("q", "Question"), hits=hits,
                            docs_by_id=docs, top_k=3)["context_doc_ids"] == ["1", "2", "3"]


def test_order_is_validated_and_fingerprinted(tmp_path):
    config = {"queries": "rag", "data": {"nqplus_dir": str(tmp_path)},
              "reader": {"llm": {"provider": "abstain"}, "context_order": "reverse"}}
    reverse = resolve_reader(config, ["bm25"])
    config["reader"]["context_order"] = "retrieval"
    normal = resolve_reader(config, ["bm25"])
    assert "inputs.reader" in changed_inputs(compute_fingerprints(config, [], normal),
                                              compute_fingerprints(config, [], reverse))
    config["reader"]["context_order"] = "typo"
    with pytest.raises(ValueError, match="context_order"):
        resolve_reader(config, ["bm25"])


def test_runner_keeps_ir_and_metrics_identical_and_maps_reversed_citations(tmp_path, monkeypatch):
    from evlab.experiments import runner

    fixtures = Path(__file__).parent / "fixtures"
    monkeypatch.setattr(runner, "build_llm_client", lambda cfg: Reader())
    config = {"experiment": "order", "queries": "rag",
              "data": {"nqplus_dir": str(fixtures / "nqplus"),
                       "human_corpus": str(fixtures / "corpus/human.jsonl")},
              "corpus": {"fixed": {}, "include_human": True}, "retrievers": ["bm25"],
              "retrieval": {"top_k": 3}, "evaluation": {"cutoffs": [1, 3]},
              "reader": {"llm": {"provider": "abstain"}, "top_k": 2,
                         "cache_path": str(tmp_path / "reader.sqlite")},
              "output": {"dir": str(tmp_path / "runs")}}
    runs = []
    for order in ["retrieval", "reverse"]:
        config["reader"]["context_order"] = order
        summary = run_experiment(config)
        directory = tmp_path / "runs" / summary["run_id"]
        runs.append((summary, read_jsonl(directory / "bm25/retrieval.jsonl"),
                     read_jsonl(directory / "bm25/answers.jsonl")))
    assert runs[0][1] == runs[1][1]
    for key, value in runs[0][0]["retrievers"]["bm25"].items():
        if not key.startswith("answer_"):
            assert runs[1][0]["retrievers"]["bm25"][key] == value
    for normal, reverse in zip(runs[0][2], runs[1][2]):
        assert normal["context_doc_ids"] == reverse["context_doc_ids"][::-1]
        assert reverse["cited_doc_ids"] == [normal["context_doc_ids"][-1]]
    assert runs[0][0]["run_id"] != runs[1][0]["run_id"]
