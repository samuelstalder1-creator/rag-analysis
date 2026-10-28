"""Every shipped experiment config must resolve its retrievers and reader."""

from pathlib import Path

import pytest

from evlab.config import load_config
from evlab.experiments.sweep import expand_sweep
from evlab.rag.config import resolve_reader
from evlab.retrieval.specs import resolve_retrievers

ROOT = Path(__file__).parents[1]
CONFIGS = sorted((ROOT / "configs/experiments").rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_experiment_config_resolves(path, monkeypatch):
    monkeypatch.chdir(ROOT)  # configs reference prompt/retriever files relatively
    config = load_config(path)
    for _label, _params, run_config in expand_sweep(config):
        names, _specs = resolve_retrievers(run_config)
        resolve_reader(run_config, names)
