# rag-analysis

NQ+ Evidence Laundering Lab: a config-driven Python project for controlled retrieval and RAG experiments with human, AI-generated, correct and false evidence.

## Setup

```bash
uv sync
uv run evlab --help
uv run pytest
```

Optional extras:

```bash
uv sync --extra yaml      # PyYAML for full YAML support
uv sync --extra gemini    # Gemini API generation
uv sync --extra dense     # sentence-transformers dense retrieval
```

## Data

Default configs expect the NQ+ artefacts inside this repo:

```text
data/nqplus/clean/
data/cocktail/nq/corpus/human.jsonl
data/cocktail/nq/corpus/llama-2-7b-chat-tmp0.2.jsonl
```

Copy or symlink the source artefacts into those paths before running validation or experiments.

## First Run

```bash
uv run evlab data validate --config configs/data/nqplus.yaml
uv run evlab run configs/experiments/rq1_displacement.yaml
uv run evlab run configs/experiments/rq1_dense_human_minilm.yaml
uv run evlab run configs/experiments/rq1_dense_mixed_minilm.yaml
```

## Generation

Local OpenAI-compatible endpoint, e.g. vLLM or llama.cpp server:

```bash
uv run evlab generate-false-answers --config configs/generation/pilot-local.yaml --limit 5
uv run evlab generate --config configs/generation/pilot-local.yaml --limit 5
```

Gemini:

```bash
export GEMINI_API_KEY=...
uv run evlab generate-false-answers --config configs/generation/pilot-gemini.yaml --limit 5
uv run evlab generate --config configs/generation/pilot-gemini.yaml --limit 5
```

All generation calls are cached in SQLite under `cache/llm.sqlite`.

False-evidence generation has two separate stages:

```bash
# Step 1: save false answers under data/generated/, keeping queries unchanged.
uv run evlab generate-false-answers --config configs/generation/qwen35-2b-false-answers.yaml

# Step 2: read those answers and create false corpus documents.
uv run evlab generate-false-documents --config configs/generation/qwen35-2b-false.yaml
```

Step 1 is implemented in `src/evlab/generation/false_answers.py`; step 2 has its
own entry point in `src/evlab/generation/false_documents.py`. Each successful
answer is saved atomically to
`data/generated/qwen35-2b-all/generated_false_answers.jsonl`, keyed by query ID
and a fingerprint of the question and gold answer. Source `queries.jsonl` stays
unchanged. Valid saved answers are reused
without replacement, even if document generation uses a different model. Every
source document belonging to the question uses that same saved false answer.

Document generation never calls the false-answer generator. Missing or invalid
saved answers cause an error before any document model calls or corpus writes.
The generic `evlab generate` command also requires saved answers when its
conditions include `false`; correct-only generation is independent of step 1.
`--limit` selects the same candidate prefix in both stages.

False answers and false documents are never automatically restored from legacy
backups. Fresh answers are generated only when you explicitly run step 1.
Answer-generation provenance is tracked in the shared `metadata.yaml` under
`answer_generations`; corpus rows keep their three-field structure. Failed
answer attempts have a separate reject log. Neither stage writes progress JSON.
The background helper defaults to the correct-only config and skips false-answer
generation. To explicitly run both stages and both document conditions, use
`CONFIG=configs/generation/qwen35-2b-all.yaml FALSE_ANSWERS_FIRST=1 bash scripts/run_qwen35_2b_all.sh`.

Generated corpus rows contain only `_id`, `title`, and `text`; `_id` is the
original human document ID. Corpus files use simple unique names such as
`corpus_0001.jsonl`. One shared `metadata.yaml` in each output directory tracks
all corpus files, for example:

```yaml
version: 1
files:
  corpus_0001.jsonl:
    model: "Qwen/Qwen3.5-2B"
    condition: "false"
    temperature: 0.3
    sample_idx: 0
    generation_depth: 1
    source: "candidates"
    prompt_version: "v1"
    document_count: 204
```

Each model, condition, temperature, sample, source, depth, and prompt version has
its own registered file. Registry updates are locked so separate correct and
false runs can share the same output directory. Counts update with progress and
when generation stops. Correct copies preserve the source title. Reject logs
remain separate from corpus rows; progress is printed to the console.

With `resume: true`, matching registered files are reused and completed documents
are skipped. Previous metadata-rich filenames are automatically renamed and
registered without changing their contents. Correct rows from old `generated_docs.jsonl` and `correct_docs.jsonl` can also
be imported; legacy false rows are ignored.
Use the original temperature when importing those legacy files, which did not
record it. Existing blank legacy titles remain blank.

Experiment `corpus.generated_docs` accepts a file, directory, or glob. Directory
loading includes the corpora listed in its shared `metadata.yaml`; the registry
provides generation metadata, and relevance annotations in NQ+ `queries.jsonl` restore question mappings. Documents
receive unique internal IDs per corpus file while keeping their original ID as
the provenance root. Renamed corpora retain their previous internal ID namespace.

False documents retain the source title. If it contains a gold-answer alias,
generation replaces that alias with the same false answer used in the body and
rejects titles that still contain a gold-answer alias. An answer in the source
title is therefore not an exclusion criterion for generation candidates;
reader eligibility and the `single`/`variants` answer requirement still apply.
Resuming a run keeps documents already written; title handling applies to newly
generated documents.

## Generation Inventory and Cleanup

`data/cocktail/nq/` contains the original Cocktail dataset: `queries.jsonl`,
`qrels/test.tsv`, and the supplied human and Llama corpora under `corpus/`.
Prepared answer annotations and generation candidates live separately under
`data/nqplus/clean/`. Our Qwen outputs live under `data/generated/`.

Inspect what exists and what is missing without starting a model:

```bash
uv run evlab generation-status --config configs/generation/qwen35-2b-all.yaml
```

The report compares document IDs against the current original human corpus and
prepared generation candidates. It includes original question/document counts,
saved false-answer counts, per-file coverage, duplicate IDs, empty text, registry
count mismatches, and unregistered files. Candidate correct-copy coverage can be
satisfied by the full human-corpus generation; a second candidate-only file is
not needed to cover those source IDs. ID coverage does not certify text quality
or establish that old outputs were produced from the current source version.

Delete derived false-generation artifacts and reset their metadata:

```bash
uv run evlab cleanup-false --config configs/generation/qwen35-2b-all.yaml
```

Cleanup removes `generated_false_answers.jsonl`, registered false corpora,
legacy false-document copies, false-generation reject
records, and answer-generation metadata. It also clears the configured generation
SQLite cache, which may contain false answers and paraphrased false text. Correct
corpora, correct reject records, source queries, generated eligibility labels,
and the original Cocktail dataset are retained.
The operation is repeatable; the status command shows the resulting inventory.

## Answer Classification

Split the former `answer_list_or_conflict` flag using the question structure,
answer annotations, and source-backed reviews:

```bash
uv run evlab data classify-answers --config configs/data/nqplus.yaml
```

`data/nqplus/clean/queries.jsonl` is the only prepared query file required.
Reader queries, generation candidates, and relevance annotations are derived
from source metadata and optional `generated_labels.jsonl` when needed; source titles and text come from the original
human corpus. No separate RAG, candidate, report, or prepared qrels files are
required. Candidates follow the order of eligible rows in `queries.jsonl`.

The classification command writes `data/nqplus/clean/generated_labels.jsonl`
and never changes `queries.jsonl`. Loaders join these labels by query ID and
reject stale labels if the source record changed. Reviewed decisions in
`configs/data/answer_classification_reviews.json` include source document IDs and
quotes; the command verifies both the answer strings and source quotes before
changing the data.

- `answer_is_list` identifies an answer set or a question with multiple slots.
- `answer_conflict` identifies reviewed incompatible answer annotations for the
  same requested fact, or a documented contradiction in an answer list.
- `answer_review_required` marks cases where neither question structure nor
  checked evidence establishes whether the answers are variants, a list, or a
  conflict. Such cases have `answer_kind: unresolved`.

The list and conflict flags can both be true. A false flag means that the
condition was not detected; for unresolved cases it does not establish its
absence. List detection does not certify that every annotated member is correct.
Distinct dates or names alone are insufficient to establish a conflict, since
they may refer to different periods, roles, or scopes.

Classification reasons, evidence, flags, and eligibility are stored in
`generated_labels.jsonl`. Entries with
`metadata.flags.answer_review_required: true` still need review. Add
evidence-backed decisions to `configs/data/answer_classification_reviews.json`
and rerun the command to resolve them. `evlab generation-status` reports counts
directly from the current queries and generated files. Only reader-eligible
`single`/`variants` questions enter the current generation pipeline; lists,
conflicts, and unresolved cases remain excluded. Answer strings and gold answers
are preserved by classification; resolving a conflict requires a separate
correction rather than silently choosing one annotation.

## Dense Retrieval

MiniLM is supported with:

```bash
uv sync --extra dense
uv run evlab run configs/experiments/rq1_dense_human_minilm.yaml
uv run evlab run configs/experiments/rq1_dense_mixed_minilm.yaml
```

Embeddings are cached under `cache/embeddings`. The mixed MiniLM run reuses the Human-only embedding prefix and only encodes the Cocktail-Llama half.

E5 is configured in `configs/experiments/rq1_dense_mixed_e5.yaml`, but requires `intfloat/e5-base-v2` to be available locally or `dense_local_files_only: false` with network access.

To store E5 inside the repo-local `models/` directory:

```bash
uv sync --extra dense --extra yaml --group dev
uv run python scripts/cache_e5_base_v2.py
uv run evlab run configs/experiments/rq1_dense_mixed_e5.yaml
```
