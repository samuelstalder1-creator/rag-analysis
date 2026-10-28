"""Regression tests for cache keys, run status, fingerprints and resumption."""

import json
import shutil
from pathlib import Path

import pytest

from evlab.experiments import runner
from evlab.experiments.fingerprint import (
    RunConflictError,
    changed_inputs,
    corpus_content_hash,
)
from evlab.io import read_jsonl, write_json
from evlab.llm.cache import CachedLLMClient
from evlab.schema import Doc

FIXTURES = Path(__file__).parent / "fixtures"


def make_doc(doc_id: str, text: str) -> Doc:
    return Doc(doc_id=doc_id, title="", text=text, origin="human", dataset="d", query_id="q", condition="correct")


# --- LLM cache -------------------------------------------------------------

class FakeClient:
    provider = "fake"
    model = "m"

    def __init__(self, max_tokens=None):
        self.max_tokens = max_tokens
        self.calls = 0

    def request_config(self):
        return {"max_tokens": self.max_tokens}

    def generate(self, prompt, *, temperature=0.0, seed=None):
        self.calls += 1
        return f"limit={self.max_tokens}"


def test_llm_cache_separates_max_tokens(tmp_path):
    path = tmp_path / "cache.sqlite"
    assert CachedLLMClient(FakeClient(256), path).generate("q") == "limit=256"
    assert CachedLLMClient(FakeClient(8192), path).generate("q") == "limit=8192"
    # same settings still hit the cache
    again = FakeClient(256)
    assert CachedLLMClient(again, path).generate("q") == "limit=256"
    assert again.calls == 0


def test_llm_cache_key_unchanged_without_request_settings(tmp_path):
    from evlab.io import stable_hash

    class Plain:
        provider = "p"
        model = "m"

        def generate(self, prompt, *, temperature=0.0, seed=None):
            return "answer"

    path = tmp_path / "cache.sqlite"
    CachedLLMClient(Plain(), path).generate("q", seed=1)
    legacy_key = stable_hash({"provider": "p", "endpoint": None, "model": "m", "prompt": "q",
                              "temperature": 0.0, "seed": 1})
    import sqlite3
    with sqlite3.connect(path) as conn:
        assert conn.execute("select count(*) from llm_cache where cache_key = ?", (legacy_key,)).fetchone()[0] == 1


# --- Dense embedding cache -------------------------------------------------

def make_dense(tmp_path, **overrides):
    np = pytest.importorskip("numpy")
    from evlab.retrieval.dense import SentenceTransformerRetriever

    class Encoder:
        def __init__(self):
            self.encoded: list[str] = []

        def encode(self, texts, **kwargs):
            self.encoded.extend(texts)
            return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

    retriever = SentenceTransformerRetriever.__new__(SentenceTransformerRetriever)
    retriever.model_name = "m"
    retriever.batch_size = 8
    retriever.normalize = False
    retriever.encoder = Encoder()
    retriever.cache_dir = tmp_path
    retriever.doc_prefix = ""
    retriever.max_seq_length = None
    for key, value in overrides.items():
        setattr(retriever, key, value)
    return retriever


def test_dense_prefix_cache_is_not_reused_after_text_change(tmp_path):
    first = make_dense(tmp_path)
    first.index([make_doc("d1", "short")])
    second = make_dense(tmp_path)
    second.index([make_doc("d1", "a much longer changed text")])
    assert second.encoder.encoded == ["a much longer changed text"]
    assert float(second.vectors[0][0]) == len("a much longer changed text")


def test_dense_prefix_cache_reused_for_identical_prefix(tmp_path):
    make_dense(tmp_path).index([make_doc("d1", "alpha")])
    grown = make_dense(tmp_path)
    grown.index([make_doc("d1", "alpha"), make_doc("d2", "beta")])
    assert grown.encoder.encoded == ["beta"]
    assert len(grown.vectors) == 2


def test_dense_prefix_cache_respects_embedding_settings(tmp_path):
    make_dense(tmp_path).index([make_doc("d1", "alpha")])
    changed = make_dense(tmp_path, doc_prefix="passage: ")
    changed.index([make_doc("d1", "alpha"), make_doc("d2", "beta")])
    assert changed.encoder.encoded == ["passage: alpha", "passage: beta"]


# --- Fingerprints ----------------------------------------------------------

def test_corpus_hash_covers_text_not_only_ids():
    same_ids = corpus_content_hash([make_doc("a", "one")])
    assert same_ids != corpus_content_hash([make_doc("a", "two")])
    assert corpus_content_hash([make_doc("a", "x"), make_doc("b", "y")]) == corpus_content_hash(
        [make_doc("b", "y"), make_doc("a", "x")]
    )


def test_changed_inputs_reports_differing_parts():
    old = {"evaluator_version": 1, "inputs": {"corpus": {"content_sha256": "a"}, "reader": None}}
    new = {"evaluator_version": 1, "inputs": {"corpus": {"content_sha256": "b"}, "reader": None}}
    assert changed_inputs(old, new) == ["inputs.corpus"]
    assert changed_inputs(old, old) == []


# --- Runner lifecycle ------------------------------------------------------

def runner_config(tmp_path, dataset: Path) -> dict:
    return {
        "experiment": "lifecycle",
        "queries": "rag",
        "data": {"nqplus_dir": str(dataset), "human_corpus": str(FIXTURES / "corpus/human.jsonl")},
        "corpus": {"base": "human"},
        "retrievers": ["bm25"],
        "retrieval": {"top_k": 3},
        "evaluation": {"cutoffs": [1, 3]},
        "reader": {"llm": {"provider": "abstain"}, "cache_path": str(tmp_path / "reader.sqlite"), "top_k": 3},
        "output": {"dir": str(tmp_path / "runs")},
    }


class FlakyReader:
    model = provider = "test"

    def __init__(self, fail_on_call=None):
        self.calls = 0
        self.fail_on_call = fail_on_call

    def generate(self, prompt, **kwargs):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("server stopped")
        return json.dumps({"answer": "I don't know", "citations": []})


def run_dir(tmp_path) -> Path:
    return next((tmp_path / "runs").iterdir())


def test_failed_rerun_leaves_no_stale_completion_files(tmp_path, monkeypatch):
    dataset = tmp_path / "nqplus"
    shutil.copytree(FIXTURES / "nqplus", dataset)
    config = runner_config(tmp_path, dataset)

    monkeypatch.setattr(runner, "build_llm_client", lambda cfg: FlakyReader())
    runner.run_experiment(dict(config))
    root = run_dir(tmp_path)
    assert json.loads((root / "run_status.json").read_text())["status"] == "complete"
    assert (root / "summary.json").exists() and (root / "bm25/metrics.json").exists()

    # Pretend the first attempt died after one answer, then fail the next attempt too.
    answers = read_jsonl(root / "bm25/answers.jsonl")
    (root / "bm25/answers.jsonl").write_text(json.dumps(answers[0]) + "\n", encoding="utf-8")
    (tmp_path / "reader.sqlite").unlink()
    monkeypatch.setattr(runner, "build_llm_client", lambda cfg: FlakyReader(fail_on_call=1))
    with pytest.raises(RuntimeError, match="server stopped"):
        runner.run_experiment(dict(config))
    status = json.loads((root / "run_status.json").read_text())
    assert status["status"] == "failed" and "server stopped" in status["error"]
    assert not (root / "summary.json").exists()
    assert not (root / "bm25/metrics.json").exists()
    assert len(read_jsonl(root / "bm25/answers.jsonl")) == 1


def test_rerun_resumes_stored_answers(tmp_path, monkeypatch):
    dataset = tmp_path / "nqplus"
    shutil.copytree(FIXTURES / "nqplus", dataset)
    config = runner_config(tmp_path, dataset)

    monkeypatch.setattr(runner, "build_llm_client", lambda cfg: FlakyReader(fail_on_call=2))
    with pytest.raises(RuntimeError):
        runner.run_experiment(dict(config))
    root = run_dir(tmp_path)
    assert len(read_jsonl(root / "bm25/answers.jsonl")) == 1

    (tmp_path / "reader.sqlite").unlink()  # prove the answer comes from the run folder, not the cache
    resumed = FlakyReader()
    monkeypatch.setattr(runner, "build_llm_client", lambda cfg: resumed)
    summary = runner.run_experiment(dict(config))
    total = summary["num_queries"]
    assert resumed.calls == total - 1
    assert len(read_jsonl(root / "bm25/answers.jsonl")) == total
    assert json.loads((root / "run_status.json").read_text())["status"] == "complete"


def test_changed_input_data_in_same_folder_is_rejected(tmp_path, monkeypatch):
    dataset = tmp_path / "nqplus"
    shutil.copytree(FIXTURES / "nqplus", dataset)
    config = runner_config(tmp_path, dataset)
    monkeypatch.setattr(runner, "build_llm_client", lambda cfg: FlakyReader())
    runner.run_experiment(dict(config))
    root = run_dir(tmp_path)
    before = (root / "summary.json").read_text()

    queries = dataset / "queries.jsonl"
    queries.write_text(queries.read_text().replace("chicago", "springfield", 1), encoding="utf-8")
    with pytest.raises(RunConflictError):
        runner.run_experiment(dict(config))
    assert (root / "summary.json").read_text() == before  # old results untouched


# --- Metrics and status ----------------------------------------------------

def test_context_recall_counts_only_the_first_k_raw_hits():
    from evlab.evaluation.retrieval import context_recall_at_k, recall_at_k
    from evlab.schema import Hit

    docs = {
        "a1": Doc(doc_id="a1", title="", text="t", origin="human", dataset="d", query_id="q",
                  condition="correct", provenance_root_id="A"),
        "a2": Doc(doc_id="a2", title="", text="t", origin="generated", dataset="d", query_id="q",
                  condition="correct", provenance_root_id="A"),
        "r": Doc(doc_id="r", title="", text="t", origin="human", dataset="d", query_id="q",
                 condition="correct", provenance_root_id="R"),
    }
    hits = [Hit(query_id="q", doc_id=doc_id, rank=rank, score=1.0, retriever="x")
            for rank, doc_id in enumerate(["a1", "a2", "r"], start=1)]
    relevant = {"R": 1}
    assert recall_at_k(relevant, hits, 2, docs, "relaxed") == 1.0  # merged roots reach rank 3
    assert context_recall_at_k(relevant, hits, 2, docs, "relaxed") == 0.0
    assert context_recall_at_k(relevant, hits, 3, docs, "relaxed") == 1.0


def test_generation_status_works_with_imported_targets():
    from evlab.config import load_config
    from evlab.generation.maintenance import generation_status

    config_path = Path(__file__).parents[1] / "configs/archive/generation/nqopen1520-deepseek-r1-false.yaml"
    if not (Path(__file__).parents[1] / "data/nqopen1520/answers.jsonl").exists():
        pytest.skip("local dataset not available")
    status = generation_status(load_config(config_path))
    assert status["prepared_nqplus"]["generation_questions"] > 0


# --- Inventory -------------------------------------------------------------

def test_inventory_marks_status_and_smoke(tmp_path):
    from evlab.experiments.inventory import build_inventory

    def folder(name, *, summary=False, answers=0, query_limit=None):
        root = tmp_path / name
        (root / "bm25").mkdir(parents=True)
        write_json(root / "config.json", {"experiment": name, "reader": {"llm": {}},
                                          **({"query_limit": query_limit} if query_limit else {})})
        (root / "bm25/retrieval.jsonl").write_text(
            "".join(json.dumps({"query_id": f"q{i}", "doc_id": "d"}) + "\n" for i in range(4)))
        (root / "bm25/answers.jsonl").write_text("".join('{"query_id": "q"}\n' for _ in range(answers)))
        if summary:
            write_json(root / "summary.json", {"num_queries": 4})
        return root

    folder("done", summary=True, answers=4)
    folder("partial", answers=2)
    folder("limited", summary=True, answers=2, query_limit=2)
    (tmp_path / "bare").mkdir()
    write_json(tmp_path / "bare/config.json", {"experiment": "bare"})
    rows = {row["experiment"]: row for row in build_inventory(tmp_path)}
    assert rows["done"]["status"] == "complete" and rows["done"]["purpose"] == "full"
    assert rows["partial"]["status"] == "incomplete"
    assert rows["partial"]["retrievers"]["bm25"]["retrieval_queries"] == 4
    assert rows["limited"]["purpose"] == "smoke"
    assert rows["bare"]["status"] == "prepared"
