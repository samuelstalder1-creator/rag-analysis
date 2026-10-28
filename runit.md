# Run It

Short runbook for the Linux machine with GPU. The project runs config-driven
retrieval and generation experiments over NQ+ evidence data.

## Data Layout

Place or symlink the required data here:

```text
data/nqplus/clean/
data/cocktail/nq/corpus/human.jsonl
data/cocktail/nq/corpus/llama-2-7b-chat-tmp0.2.jsonl
```

## Setup On Linux

```bash
uv sync --extra dense --group dev          # add --extra serve on the GPU server (vLLM)
uv run pytest
```

## Validate Data

```bash
uv run evlab data validate --config configs/data/nqplus.yaml
```

## E5 Retrieval

Cache E5 locally once:

```bash
uv run python scripts/cache_e5_base_v2.py
```

CPU configs:

```bash
uv run evlab run configs/experiments/rq1_dense_human_e5.yaml
uv run evlab run configs/experiments/rq1_dense_mixed_e5.yaml
```

GPU configs for the 48 GB VRAM machine:

```bash
uv run evlab run configs/experiments/rq1_dense_human_e5_gpu.yaml
uv run evlab run configs/experiments/rq1_dense_mixed_e5_gpu.yaml
```

The GPU configs use CUDA for both embedding and dense score computation.

## vLLM Server

The setup command above installs vLLM from the project dependencies.

Start Qwen3.5-2B as an OpenAI-compatible server:

```bash
export LOCAL_LLM_API_KEY=local

VLLM_USE_FLASHINFER_SAMPLER=0 uv run vllm serve Qwen/Qwen3.5-2B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
```

The sampler setting bypasses the FlashInfer compilation failure seen with the
server's CUDA 12.0 compiler (`Unknown option --compress-mode=size`).

Check that the server is reachable:

```bash
curl http://localhost:8000/v1/models
```

Keep this process running while generation runs in another terminal.
If generation logs `Connection refused`, restart vLLM and rerun the same
generation command. The configs retry transient server errors and `resume: true`
skips rows already written to disk.

## Qwen3.5-2B Generation

Run these commands from the repository root, in another terminal while vLLM is
running. Set the environment in every terminal used to start generation:

```bash
export LOCAL_LLM_API_KEY=local
export PYTHONUNBUFFERED=1
mkdir -p logs
```

Prepare eligibility labels first (this writes only `generated_labels.jsonl`):

```bash
uv run evlab data classify-answers --config configs/data/nqplus.yaml
uv run evlab generation-status --config configs/generation/qwen35-2b-all.yaml
```

Review list, conflict, and unresolved cases before expanding the eligible set.
The current generator uses reader-eligible single answers or equivalent answer
variants. An answer in the title is allowed; false-document generation edits it.

False evidence has two independent stages. First save false answers in
`data/generated/qwen35-2b-all/generated_false_answers.jsonl`; then generate documents from those saved
answers. Correct-only generation does not require the false-answer stage.

### Without an LLM Judge (deterministic validation)

`validation.mode: deterministic` skips the semantic judge in both stages. Only
rule-based checks run: answer type, gold and near-alias exclusion (e.g.
"Christopher Lloyd" for "Christopher Allen Lloyd"), target present in the
document, no extra gold mentions after paraphrasing, and near-duplicate limits.
Span selection for the minimal edit still uses the generator model.

```bash
uv run evlab generate-false-answers --config configs/generation/pilot-deterministic.yaml --limit 15
uv run evlab generate-false-documents --config configs/generation/pilot-deterministic.yaml --limit 15
```

Receipts carry `mode: deterministic` and `semantic_verified: false`; registry
entries carry `validation_mode: deterministic`. Semantic runs refuse these
answers, so judged and unjudged data cannot be mixed by accident.
With `skip_missing_false_answers: true` (set in the pilot config), the document
stage skips questions whose answer stage failed and reports them as
`skipped_questions` instead of aborting; without it, missing answers stop the run. Retries raise
the temperature by `false_answer_retry_temperature_step`. With
`false_answer_candidates_per_call: 5` the generator proposes five distractors per call
and the first one passing the rule-based checks is used; proposals equal to gold or
to earlier rejects are skipped.
Each saved answer keeps the whole pool: `candidates` (every proposal with its
rule-based `status` and `attempt`), `valid_candidates`, and `selection`
(`method: first_valid_in_model_order`, `selector_model: null`). A later LLM
selection step is meant to choose from `valid_candidates` and update
`selection`; until then `false_answer` is the first valid proposal.
Further proposal rules: a candidate that repeats the question (`copies_question`)
or already occurs in a source document (`occurs_in_source`, which would make the
edited document contradict itself) is not valid; a lowercase leading article is
dropped when gold has none. `span_selection: all` (pilot config) replaces every
gold-alias occurrence by rule instead of asking the model which spans to edit; this
can over-edit unrelated mentions (e.g. "order 3" next to "3 lines"). With
`span_selection: llm`, failed selections keep the raw model response in the reject log.
For personal-name gold answers the rule-based edit also replaces "First Last"
(e.g. a title "Christopher Lloyd" for gold "Christopher Allen Lloyd") and standalone
surnames ("Brown was elected", "Presley's label") with the target (surname); a body
that mentions only the surname is rejected. Number words 0-20 count as the same
value for the target check and the gold-residue check ("4" = "four"). `llm.max_tokens` caps
responses, and HTTP 4xx errors now include the server's explanation.

### Step 1: Generate False Answers

Run in the foreground:

```bash
uv run evlab generate-false-answers \
  --config configs/generation/qwen35-2b-false-answers.yaml
```

Alternatively, start the same step in the background:

```bash
nohup uv run evlab generate-false-answers \
  --config configs/generation/qwen35-2b-false-answers.yaml \
  > logs/qwen35_2b_false_answers.log 2>&1 &
echo $! > logs/qwen35_2b_false_answers.pid
```

The command generates one false answer per eligible question and saves each
success immediately in `generated_false_answers.jsonl` inside the configured
output directory. Source `queries.jsonl` is never written. Existing valid false answers are reused.
Deleted answers are not automatically restored from archived legacy documents.
All source documents for one question use the same saved false answer.

Wait for step 1 to finish before starting step 2. A run can finish with rejected
answers, so check readiness using the same validation as the document generator:

```bash
uv run python - <<'PYTHON'
from evlab.data.nqplus import load_generation_candidates
from evlab.generation.false_answer import is_valid_false_answer
from evlab.generation.false_answers import attach_saved_answers
from evlab.config import load_config

config = load_config("configs/generation/qwen35-2b-false-answers.yaml")
queries = attach_saved_answers(
    load_generation_candidates(config["data"]["nqplus_dir"], include_text=False),
    config["generation"],
)
missing = [q["query_id"] for q in queries
           if not isinstance(q.get("false_answer"), str)
           or not is_valid_false_answer(q["false_answer"], q["answer"], q["answer_aliases"])]
print(f"Valid false answers: {len(queries) - len(missing)}/{len(queries)}")
print(f"Missing or invalid: {len(missing)}")
if missing:
    print("First affected questions:", ", ".join(missing[:10]))
raise SystemExit(bool(missing))
PYTHON
```

If answers are missing, inspect the answer reject log and rerun step 1 after
addressing the cause. Successful answers are retained.

### Step 2: Generate False Documents

Once all required false answers are available:

```bash
nohup uv run evlab generate-false-documents \
  --config configs/generation/qwen35-2b-false.yaml \
  > logs/qwen35_2b_false_generation.log 2>&1 &
echo $! > logs/qwen35_2b_false_generation.pid
```

This derives candidates from source queries plus `generated_labels.jsonl`, joins
saved answers from `generated_false_answers.jsonl` in the output directory, then loads
source documents from the configured original human corpus. It edits the body and, if the gold answer occurs there,
the title. It never generates a new false answer. Missing or invalid saved
answers stop this stage before model calls or corpus writes. Copies that fail
validation are recorded in a document reject log instead of the corpus.

To test a small prefix, pass the same `--limit`, for example `--limit 5`, to both
stage commands. The full document run requires answers for the full candidate set.

### Correct Documents Only

```bash
nohup uv run evlab generate \
  --config configs/generation/qwen35-2b-correct.yaml \
  > logs/qwen35_2b_correct_generation.log 2>&1 &
echo $! > logs/qwen35_2b_correct_generation.pid
```

This paraphrases all documents in `data/cocktail/nq/corpus/human.jsonl` and
preserves their titles. It can run independently of false-answer generation.

### Both Document Conditions

After step 1, generate correct and false documents for the NQ+ candidate subset:

```bash
nohup uv run evlab generate \
  --config configs/generation/qwen35-2b-all.yaml \
  > logs/qwen35_2b_generation.log 2>&1 &
echo $! > logs/qwen35_2b_generation.pid
```

This all-conditions config uses `candidates` for both conditions. Use the
correct-only config above to paraphrase the full human corpus.

The helper defaults to correct-only generation and never starts the false-answer
stage automatically:

```bash
bash scripts/run_qwen35_2b_all.sh
```

To also start vLLM for that correct-only run:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 START_VLLM=1 bash scripts/run_qwen35_2b_all.sh
```

To explicitly request both false-answer and document generation in one job:

```bash
CONFIG=configs/generation/qwen35-2b-all.yaml FALSE_ANSWERS_FIRST=1 \
  bash scripts/run_qwen35_2b_all.sh
```

The helper prints its timestamped log paths and PID file. `FALSE_ANSWERS_FIRST`
defaults to `0`; it must be set to `1` to run the answer stage in the helper.

### Monitor

Follow the log for the stage you started:

```bash
tail -f logs/qwen35_2b_false_answers.log
tail -f logs/qwen35_2b_false_generation.log
tail -f logs/qwen35_2b_correct_generation.log
```

Inspect corpus sizes and the shared registry:

```bash
wc -l data/generated/qwen35-2b-all/corpus_*.jsonl
cat data/generated/qwen35-2b-all/metadata.yaml
```

Inspect available answer and document reject logs:

```bash
ls data/generated/qwen35-2b-all/*rejects.jsonl
```

Reject logs are created when rejects occur. Progress is printed to the console
and captured in the stage logs; no `progress.json` is written. Corpus document
counts are updated in `metadata.yaml` periodically and when generation stops.

### Stop and Resume

Choose the command for the active stage:

```bash
kill "$(cat logs/qwen35_2b_false_answers.pid)"
kill "$(cat logs/qwen35_2b_false_generation.pid)"
kill "$(cat logs/qwen35_2b_correct_generation.pid)"
```

Rerun the same start command to resume that stage. Step 1 keeps valid answers
already saved in `generated_false_answers.jsonl`. Step 2 uses `resume: true` to skip document IDs
already present in the registered corpus. Neither resume step chooses new false
answers for completed questions. Use `>>` instead of `>` to append to an existing
log when resuming.

If logs show `Connection refused`, restart vLLM, check `/v1/models`, and resume
the affected stage. The configs retry transient server errors.

To regenerate documents from scratch, set `resume: false` in the selected
document-generation config. This resets only its selected corpus files and
document reject log; saved false answers in `generated_false_answers.jsonl` stay unchanged.
Legacy correct documents can be imported when resuming; use the original
temperature because old metadata-rich rows did not record it. Legacy false
answers and false documents are not automatically imported.

### Outputs

```text
data/nqplus/clean/queries.jsonl                         # unchanged source
data/nqplus/clean/generated_labels.jsonl                # derived classification
data/generated/qwen35-2b-all/generated_false_answers.jsonl # false targets
data/generated/qwen35-2b-all/corpus_*.jsonl             # generated documents
data/generated/qwen35-2b-all/metadata.yaml              # shared file metadata
data/generated/qwen35-2b-all/*false-answer-rejects.jsonl # answer failures
data/generated/qwen35-2b-all/*--rejects.jsonl            # document failures
cache/llm_qwen35_2b.sqlite                              # cached model responses
logs/                                                 # stage logs and PIDs
```

Each corpus row contains only `_id`, `title`, and `text`. Files get simple unique
names such as `corpus_0001.jsonl`. The shared `metadata.yaml` identifies each
file's model, condition, temperature, sample, depth, source, prompt version, and
document count. Its `answer_generations` section tracks the answer stage,
including the generated answer-file path and the affected query IDs.

## Inventory and False-Generation Reset

The original dataset lives under `data/cocktail/nq/`. Its supplied Llama corpus
is a Cocktail baseline, separate from our Qwen generated outputs. Prepared NQ+
questions are a filtered, annotated subset of the original Cocktail questions.

Show current source counts, generated coverage, saved false answers, and missing
outputs without starting generation:

```bash
uv run evlab generation-status --config configs/generation/qwen35-2b-all.yaml
```

The report checks IDs against the restored original human corpus. It also checks
registry counts, missing files, duplicate IDs, and empty generated text. Correct
copies in the full human-corpus output also cover their NQ+ candidate source IDs.

To delete all derived false answers/documents from this output directory and
prepared dataset:

```bash
uv run evlab cleanup-false --config configs/generation/qwen35-2b-all.yaml
uv run evlab generation-status --config configs/generation/qwen35-2b-all.yaml
```

This clears saved false-answer fields, false corpus files, false-generation
reject entries, their metadata, and the configured generation SQLite cache.
Correct documents, correct reject entries, and original Cocktail files are
preserved. Repeating cleanup is safe. No legacy false answers/documents are
restored when generation resumes; step 1 must be started explicitly.

## Corpus-Mix Experiments (Retrieval + RAG)

Configs: `configs/experiments/mix/` (details in README, "Corpus Mixes" and
"Retrievers and Reader in the Config").

### 1. Update code and data

```bash
git pull
uv sync --extra dense --group dev          # add --extra serve on the GPU server (vLLM)
uv run pytest -q
```

`data/` is not in git. The mix configs need these on the server:

```text
data/nqplus/clean/queries.jsonl
data/cocktail/nq/corpus/human.jsonl
data/generated/false-deterministic/     # corpus_0001.jsonl + .manifest + metadata.yaml + generated_false_answers.jsonl
data/generated/qwen35-2b-all/           # corpus_0002.jsonl + metadata.yaml
```

Copy a missing folder from the laptop (run there, from the repo root; keep
corpus, manifest and metadata.yaml together):

```bash
rsync -av data/generated/false-deterministic data/generated/qwen35-2b-all \
  USER@SERVER:PATH/rag-analysis/data/generated/
```

### 2. Check what each run will contain (seconds, no retrieval)

```bash
uv run evlab corpus configs/experiments/mix/template.yaml
uv run evlab sweep configs/experiments/mix/variants_false_correct.yaml --dry-run
```

### 3. Retrieval-only experiments (CPU, no vLLM needed)

```bash
export PYTHONUNBUFFERED=1
mkdir -p logs
nohup uv run evlab run configs/experiments/mix/compare_retrievers.yaml \
  > logs/compare_retrievers.log 2>&1 &
nohup uv run evlab sweep configs/experiments/mix/sweep_false_ratio.yaml \
  > logs/sweep_false_ratio.log 2>&1 &
nohup uv run evlab sweep configs/experiments/mix/sweep_correct_ratio.yaml \
  > logs/sweep_correct_ratio.log 2>&1 &
```

Each BM25 + TF-IDF run takes about 1 minute (laptop: 67 s for BM25, TF-IDF
and two hybrids, ~1.7 GB RAM). Several sweeps can run in parallel if memory allows.

### 4. RAG experiments (reader = Qwen3.5-2B on vLLM)

Start vLLM as in "vLLM Server" above (`Qwen/Qwen3.5-2B`, port 8000,
`--max-model-len 4096`) and check `curl http://localhost:8000/v1/models`.
The reader preset is `configs/readers/qwen35-2b.yaml`; change `base_url`
there if vLLM runs on another host or port.

Smoke test first: 5 questions per run, the answers should be short JSON with citations:

```bash
export LOCAL_LLM_API_KEY=local
uv run evlab sweep configs/experiments/mix/rag_false_correct.yaml --limit 5
head -3 results/runs/rag_false_correct__false_add_*/bm25/answers.jsonl
```

Look for `"valid_json": true`. If you see `400` errors mentioning the context
length, lower `reader.top_k`. Then run the full sweep:

```bash
nohup uv run evlab sweep configs/experiments/mix/rag_false_correct.yaml \
  > logs/rag_false_correct.log 2>&1 &
echo $! > logs/rag_false_correct.pid
```

That is 4 variants x 2 reader retrievers x 2'172 questions, about 17k reader
calls. Answers are cached in `cache/reader.sqlite`, so a restarted sweep only
calls the LLM for new prompts.

### 5. Results

```bash
tail -f logs/rag_false_correct.log
ls results/sweeps/                      # one folder per sweep: summary.csv / summary.jsonl
column -s, -t < results/sweeps/rag_false_correct_*/summary.csv | less -S
```

Per run: `results/runs/<experiment>__<variant>_<hash>/` with `run_status.json`
(`running`/`complete`/`failed`), `summary.json`, `manifest.json` (with input
fingerprints), `corpus_selection.json`, `resolved.json` and per retriever
`metrics.json`, `retrieval.jsonl`, `answers.jsonl`. A rerun with the same config
resumes unfinished reader answers; `uv run evlab inventory` shows which runs are
complete, partial or smoke tests. Copy results to the laptop with
`rsync -av USER@SERVER:PATH/rag-analysis/results/ results/`.

## Other Useful Runs

```bash
uv run evlab run configs/experiments/rq1_displacement.yaml
uv run evlab run configs/experiments/rq1_dense_human_minilm.yaml
uv run evlab run configs/experiments/rq1_dense_mixed_minilm.yaml
uv run evlab report --run-dir results/runs
```
