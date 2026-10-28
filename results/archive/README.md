# Historical results

- `runs/`: completed older retrieval/RAG runs, including TF-IDF, E5, fusion,
  corpus-wide duplication, contamination ratios and base1520 comparisons.
- `sweeps/`: their sweep summaries/config snapshots and retained corpus previews.
- `analysis/`: the older base1520 report and the dated repository-audit inventory.
- `reports/`: earlier standalone analysis summaries.

These files retain their original contents, run IDs and configuration snapshots.
Old snapshot paths are historical, not instructions to recreate deleted folders.
`data/generated/relocations.json` records the dataset/run directory moves.

The completed selected Qwen evaluation and its analysis remain outside the archive.
The two removed all-source export directories held only audits/previews after
previous partial-run cleanup; no completed results were removed in this migration.

Inspect this archive with `evlab inventory --root results/archive`. The older
base1520 analysis script is under `scripts/archive/` and points here by default.
