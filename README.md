# rag-analysis

Controlled retrieval and RAG experiments with human, correct, false and attack evidence.

## Active experiment

The single active experiment is
`configs/experiments/nqopen1520_selected_sources_qwen3_4b_bm25.yaml`:
17 corpus variants, 1,520 questions, BM25 retrieval and the
`Qwen/Qwen3-4B-Instruct-2507` reader (temperature 0, seed 13, top-five context).

| Item | Location |
|---|---|
| Experiment config | `configs/experiments/nqopen1520_selected_sources_qwen3_4b_bm25.yaml` |
| Completed evaluation | `results/evaluations/qwen3_4b_selected_bm25_20261027/` |
| Analysis | `results/analysis/selected_bm25_qwen3_4b_20261028/report.md` |
| Evaluation profile | `scripts/profiles/qwen3_4b_selected.env` |
| Historical configs | `configs/archive/` |
| Historical results | `results/archive/` |
| Dataset locations and renames | `data/generated/README.md` |

The two all-source 32-variant experiment configs, their launch profiles and
systemd evaluation units were removed. Their remaining audit/preview-only exports
were deleted; the completed 17-variant evaluation remains intact.

## Setup and checks

```bash
uv sync --group dev
uv run pytest -q
uv run evlab sweep configs/experiments/nqopen1520_selected_sources_qwen3_4b_bm25.yaml --dry-run
```

The dry run builds corpus variants and reports their composition; it makes no
model requests. GPU serving is optional (`uv sync --extra serve --group dev`).
Dense retrieval and Gemini are optional extras (`dense`, `gemini`).

## Run the selected evaluation

```bash
EVAL_CHECK_ONLY=1 bash scripts/run_evaluation.sh qwen3_4b_selected
bash scripts/run_evaluation.sh qwen3_4b_selected
```

The launcher checks the model server, runs a two-question smoke test and starts
the configured sweep. Existing complete exports are preserved by the COMPLETE
marker. A renamed input path changes the config identity for newly started runs;
the stored historical run directories keep their original names and snapshots.

The selected systemd unit remains in
`scripts/systemd/qwen3-4b-selected-bm25-evaluation.service`. These are templates
for the Linux GPU host; no installed services are changed by repository cleanup.

## Generated evidence

| Source | Location | Use |
|---|---|---|
| Human originals | `data/cocktail/nq/corpus/human.jsonl` | Shared input |
| Qwen3.5-2B correct | `data/generated/deprecated/nqopen-plus-resolved-qwen35-2b-correct-v1/` | Deprecated; retained for historical sweep reproduction |
| Qwen3.5-2B false | `data/generated/nqopen-plus-resolved-qwen35-2b-false-v1/` | Current retained source |
| Qwen3-4B correct/false | `data/generated/nqopen1520-qwen3-4b-instruct-2507-{correct,false}-v1/` | Current sources |
| GPT-OSS-120B attacks | `data/generated/nqopen1520-attack-together-gpt-oss-120b-v1/` | Current attack source |

The completed selected evaluation includes the deprecated correct corpus. Its
`resolved_correct` source key and the recorded results are preserved. Deprecation
marks its status for future work; it does not remove a condition from that
historical 17-variant design. To exclude it in a future experiment, explicitly
disable that source and use a new experiment name.

Six generation configs remain active under `configs/generation/`: Qwen3-4B
correct/false, source-grounded Together GPT-OSS correct/false, GPT-OSS attacks
and resolved Qwen3.5-2B false. Other retained
configs are in `configs/archive/generation/`, including the deprecated correct
corpus, deterministic pilot, older Qwen3.5 jobs and DeepSeek generation.

For Qwen3-4B generation, start the server and run the stages in another terminal:

```bash
bash scripts/run_qwen3_4b_generation.sh serve
bash scripts/run_qwen3_4b_generation.sh smoke
bash scripts/run_qwen3_4b_generation.sh all
```

Attack generation and its read-only preview use the retained Together config:

```bash
uv run evlab generate-attack-documents --config configs/generation/nqopen1520-attack-together.yaml --dry-run
```

Actual Together generation requires `TOGETHER_API_KEY` in the environment.
No API keys belong in configuration files. Existing generation cache paths are
retained to preserve completed requests.

### Source-grounded GPT-OSS correct and false documents

The new configs are
`configs/generation/nqopen1520-together-gpt-oss-120b-{correct,false}.yaml`.
They use `openai/gpt-oss-120b` via Together on the same `nqopen1520` base.
Correct generation paraphrases the human title/body while preserving the gold
answer. False generation edits that source to the existing validated false
target and then paraphrases it. This is separate from the attack pipeline, which
generates passages from the question and target alone.

```bash
bash scripts/run_together_generation.sh dry-run
# Set TOGETHER_API_KEY in this terminal, then inspect a five-item pilot:
bash scripts/run_together_generation.sh smoke
# Resume the pilot and process the rest:
bash scripts/run_together_generation.sh all
```

`correct` and `false` can also be run separately. The correct limit counts source
documents; the false limit counts questions. Both jobs retain accepted outputs
and use separate versioned directories/caches:
`data/generated/nqopen1520-together-gpt-oss-120b-{correct,false}-v1/`.
Each request has a 4,096-token budget and each document at most three attempts.
Checks are deterministic, matching the existing Qwen experiments; receipts do
not claim semantic verification. Rejected documents remain logged for review.
No local model server or GPU is required.

Corpus files, receipts and generation signatures are byte-preserved during
relocation. `data/generated/relocations.json` maps old paths to current locations
so historical audits can still verify their original input hashes.

## Analysis, inventory and metric interpretation

```bash
PYTHONPATH=src .venv/bin/python scripts/analyze_selected_rag.py
uv run evlab inventory
uv run evlab inventory --root results/archive
uv run evlab report
```

The default report and inventory cover `results/evaluations`. Use an explicit
root for archived experiments. Smoke runs are distinguished from full runs.

Runs record status, input fingerprints and evaluator version. The reader keeps
completed per-question answers for compatible interrupted runs. Caches include
request/embedding configuration and content checks. A directory containing
answers without a completion record is not a completed experiment.

`recall@k` collapses source roots before the cutoff; `context_recall@k` evaluates
the actual first k retrieved documents. Relaxed relevance includes poisoned
versions of relevant roots and does not establish factual accuracy. Answer
correctness and false-target adoption use normalized exact matching; citation
range validity does not establish entailment. Historical exports retain their
original evaluator definitions.

## Archive

`configs/archive/` contains historical experiment/generation configs. All paths
inside them are interpreted relative to the repository root. Historical run
snapshots in `results/archive/` are unchanged and may record old input paths.

The older base1520 analysis is available at
`scripts/archive/analyze_base1520_rag.py`; its defaults point at archived configs
and runs. Archived launch documentation is under `docs/archive/`.

The core data builders, sparse/dense retrievers and model adapters remain
available. Only the active configuration surface was narrowed. No completed
historical experiment results were deleted.
