# Historical configurations

Run commands from the repository root. Paths in these YAML files remain
repository-relative. This folder is excluded from the active experiment list,
but archived experiment configs are still checked by the config-resolution tests.

- `experiments/`: 20 older BM25/TF-IDF, dense, hybrid, ratio and pilot experiment configs.
- `generation/`: eight historical generation configs, including the deprecated
  Qwen3.5-2B correct source, DeepSeek jobs and deterministic pilot.

The two `nqopen1520_all_sources_*_bm25.yaml` configs and their evaluation launch
profiles/services were removed rather than archived. The selected 17-variant
Qwen experiment stays in `configs/experiments/`.

Archived output settings document the original destinations. An intentional new
reproduction should use a distinct experiment name/output directory. Historical
results are under `results/archive/`; their saved config snapshots retain old paths.
Use `data/generated/relocations.json` to locate relocated datasets.
