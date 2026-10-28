# Generated datasets

The resolved Qwen3.5-2B datasets were renamed on 2026-10-28:

| Previous directory | Current directory | Documents |
|---|---|---:|
| `nqopen-plus-resolved-correct` | `deprecated/nqopen-plus-resolved-qwen35-2b-correct-v1` | 1,805 |
| `nqopen-plus-resolved-false` | `nqopen-plus-resolved-qwen35-2b-false-v1` | 1,663 |

These names identify the original resolved NQOpen+ source dataset and the actual
generator, `Qwen/Qwen3.5-2B`. They are not datasets generated specifically for the
1,520-question subset. That evaluation selects 1,670 correct and 1,657 compatible
false documents from these files.

All dataset files, validation receipts, generation signatures and IDs are
unchanged. The correct generator config is under `configs/archive/generation/`;
the false config remains in `configs/generation/`. Existing cache filenames are
retained to preserve cached requests.

The correct corpus is deprecated but remains an input to the historical
17-variant selected sweep. The active experiment points to its deprecated path;
no historical condition or result was silently removed. Older `base-1520` and
`qwen35-2b-all` corpora now live in `data/generated/archive/`.

Historical run snapshots and audits retain their original paths. The directory
mapping in `relocations.json` locates those inputs without rewriting the snapshots.
`scripts/analyze_selected_rag.py` uses this mapping when checking input hashes.
Active experiment configs use the new paths; changing an experiment config's
paths produces a new run identity. No completed evaluations were rerun or renamed.

## Generation configuration cleanup

Removed configurations without retained output datasets, versioned output history
or references from saved run configurations:

- `pilot-gemini.yaml`, `pilot-local.yaml`, `pilot-ollama.yaml`, `pilot-validated.yaml`
- `nqopen-plus-false.yaml`
- `nqopen1520-attack-qwen3-4b.yaml`
- `qwen35-2b-correct-candidates.yaml`, `qwen35-2b-correct-repair.yaml`

This inventory describes retained repository artifacts; it cannot establish that
an untracked pilot was never run on another computer.

`pilot-deterministic.yaml` is archived because historical completed experiments
reference `data/generated/false-deterministic`, even though that output directory
is not present in this checkout. The DeepSeek generation configs are also archived:
their outputs existed in Git history and the launchers still support these jobs.
The Qwen3.5 configs sharing `qwen35-2b-all` are archived with references to that
existing dataset's archive location.
