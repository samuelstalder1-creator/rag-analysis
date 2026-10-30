# rag-analysis

Controlled retrieval and RAG experiments with human, correct, false and attack evidence.

## Active experiment

The completed reference experiment is
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

## Core 12-variant Top-15 evaluation (2026-10-30)

`configs/experiments/nqopen1520_core12_sources_qwen3_4b_bm25_top15.yaml`
defines the new evaluation: all 1,520 questions, BM25 (`k1=1.5`, `b=0.75`,
query stopwords removed), retrieval top-15 and Qwen3-4B-Instruct-2507 reader
top-15 (temperature 0, seed 13, unchanged `reader_v1.txt`). Metrics use cutoffs
1, 5, 10 and 15, retaining strict/relaxed relevance and the relaxed recall alias.
Full source files are added without document-size balancing or a shared-question
subset. Existing false-target compatibility checks remain enabled.

All 12 variants include the human corpus. In the table, Q+/Q- denote Qwen3-4B
correct/false rewrites, G+/G- denote the new source-grounded GPT-OSS-120B
correct/false rewrites, and A denotes the existing question-and-target GPT-OSS
attack documents.

| Variants | Purpose |
|---|---|
| H | Human baseline |
| H+Q+, H+G+ | Correct evidence by generator |
| H+Q-, H+G- | False evidence by generator |
| H+A | Attack baseline |
| H+Q++Q-, H+G++G- | Conflicting evidence from the same generator |
| H+Q++G-, H+G++Q- | Conflicting evidence across generators |
| H+Q++A, H+G++A | Correct evidence against the same attack |

```bash
# Validate inputs and all twelve corpus combinations without model requests:
uv run evlab sweep configs/experiments/nqopen1520_core12_sources_qwen3_4b_bm25_top15.yaml --dry-run
EVAL_CHECK_ONLY=1 bash scripts/run_evaluation.sh qwen3_4b_core12_top15
# On the GPU host, check/start the server, smoke-test, then run all variants:
bash scripts/run_evaluation.sh qwen3_4b_core12_top15
```

Results go to `results/evaluations/qwen3_4b_core12_bm25_top15_20261030/`;
the reader uses a separate cache under `cache/reader_nqopen1520_core12_qwen3_4b_bm25_top15_20261030.sqlite`.
The launcher smoke-tests two questions in the last variant (H+G++A) before
starting the complete sweep. The historical configuration and results retain
their own paths.

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

Answer matching treats Unicode whitespace, hyphens (U+2010/U+2011) and curly
quotation marks as their plain-text equivalents, including when finding gold
residue and answer spans. Original document text and edit offsets are preserved.
Other punctuation, spelling, numbers and date formats still have to match.
For false targets supplied as quoted titles, the outer double quotes are optional;
the full title wording must still match. This also handles punctuation placed
inside closing quotes. Gold-alias and edit-span matching keep their exact wording.
Resume fingerprints keep the original literal `aliases_in_doc` annotation for
source identity; validation uses the current matching rules. This lets existing
corpora resume after a typography fix while still detecting source text changes.

False generation can edit a title-only answer while keeping the original body
as context. The generated body must still contain the target. Missing targets
and new gold mentions trigger corrective feedback within the existing three
attempts; they are never accepted just to complete a run. `exclude_doc_ids` puts
documents on review hold without changing fingerprints or random seed positions.
The GPT-OSS false configuration holds `doc40783` (`test1150`), whose question
about court powers is paired with the gold answer "Justices".

`all` runs Correct and False concurrently as two API jobs, using their separate
caches/output directories. Each job remains sequential internally. Smoke tests
remain sequential for inspection. The launcher waits for both jobs, reports
either failure, and stops its child processes when interrupted. Together's
existing rate-limit retries still apply; the speedup depends on API throughput.

Retry documents still missing after rejection with fresh sampling:

```bash
bash scripts/run_together_generation.sh all --retry-seed 1
```

Increment the retry seed for subsequent rounds. Accepted documents retain their
fingerprints and are skipped. A different retry seed avoids cached failed
responses; validation rules and imported targets remain unchanged. Source/input
failures may remain blocked regardless of sampling.

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

## Human-length attacks v2 and reader context order

The new attack generation config is
`configs/generation/nqopen1520-attack-together-human-length.yaml`.
It uses the mean body word count of all 104,194 human documents (86.9369 words),
a target of 87 words, and a strictly enforced +/-10% band of 79–95 words per
body. Word counting uses whitespace splitting for both human and attack texts;
titles are excluded. The mean is a target, not a guarantee of an identical
corpus-wide average. Out-of-band generations are rejected and retried, never
truncated. The audit reports the resulting mean. Only the question, false target
and aggregate length limits reach the generator; no human passage or gold answer
is included. The reference file hash and derived statistics are stored with the
versioned generation signature/receipts. Attack v1 keeps its historical validation.

The experiment
`configs/experiments/nqopen1520_core12_attack_v2_qwen3_4b_bm25_top15_order.yaml`
uses the same 12 corpus conditions with attack v2 in the attack conditions,
and two reader orders for each condition (24 runs, 36,480 answers):

- `reader.context_order: retrieval`: BM25 ranks 1 through 15.
- `reader.context_order: reverse`: the same selected hits in ranks 15 through 1.

Ordering is applied after selecting the IR top-15, immediately before context
formatting. The reader's citation numbers are assigned in the presented order;
`answers.jsonl` records `context_order`, `context_doc_ids`, `context_ir_ranks`
and the mapped `cited_doc_ids`. Retrieval exports and strict/relaxed recall
retain the original BM25 ranks. Reader order is included in run fingerprints.
Output/cache paths are separate from the earlier experiment. Runs with omitted
`context_order` retain their previous retrieval-order configuration identity.

Run from the repository root with `TOGETHER_API_KEY` available for generation:

```bash
uv run evlab generate-attack-documents --config configs/generation/nqopen1520-attack-together-human-length.yaml --dry-run
# Generate a five-question pilot, then resume to the full dataset:
uv run evlab generate-attack-documents --config configs/generation/nqopen1520-attack-together-human-length.yaml --limit 5
uv run evlab generate-attack-documents --config configs/generation/nqopen1520-attack-together-human-length.yaml
# Require full coverage and valid lengths before evaluating:
uv run python scripts/audit_attack_corpus.py --config configs/generation/nqopen1520-attack-together-human-length.yaml
uv run evlab sweep configs/experiments/nqopen1520_core12_attack_v2_qwen3_4b_bm25_top15_order.yaml --dry-run
# On the GPU host:
bash scripts/run_evaluation.sh qwen3_4b_core12_attack_v2_order
```

The attack-v2 data must be generated before the experiment's corpus dry-run or
reader evaluation can run. The previous Top-15 context audit used attack v1;
it is not a measurement of the newly generated attack-v2 prompts.

For missing attack documents, use a fresh sampling batch while retaining all
accepted rows and the original corpus signature:

```bash
uv run evlab generate-attack-documents --config configs/generation/nqopen1520-attack-together-human-length.yaml --retry-seed 1
```

Increase `--retry-seed` for subsequent batches. The seed selects a new,
non-overlapping block of generation attempt seeds, bypassing cached failed
responses; receipts and rejection records retain the retry batch number.
