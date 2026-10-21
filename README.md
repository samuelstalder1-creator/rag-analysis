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

False evidence has two explicit stages. Both read the current prepared query
annotations and original human evidence; neither changes source queries.

```bash
uv run evlab generate-false-answers --config configs/generation/pilot-validated.yaml --limit 5
uv run evlab generate-false-documents --config configs/generation/pilot-validated.yaml --limit 5
```

The same candidate prefix is selected by `--limit` in both stages. Start with a
small reviewed pilot. Stage 1 requires concrete string candidates, rejects known
answer aliases and equivalent numeric/date forms, and checks answer type. There
are no synthetic fallback answers. A semantic verifier must affirm source support
for the gold answer, target plausibility, matching type, and target incorrectness.
False-answer assessment uses categorical `gold_support` and `target_relation`
verdicts rather than an ambiguous `incorrect` boolean. Only `SUPPORTED` gold and
`CONTRADICTED` target are accepted, with a verbatim source quote containing a gold
alias. `EQUIVALENT`, `SUPPORTED` targets and `UNKNOWN` are rejected. A reason string
never overrides the category.

Each generator call requests exactly one answer: `{"candidate": "..."}`. A question
has one shared generation budget, configured under `generation` with
`false_answer_max_attempts: 3` (three total attempts, including the first call).
Malformed output, structural rejection, semantic rejection, and a repeated rejected
target each consume one attempt. The next call includes the rejected targets and
failure feedback and uses a different seed. Repeated rejected targets are not
sent to the verifier again. The first verified answer is saved immediately; all
previous failed attempts are retained in its receipt. If the budget is exhausted,
no answer is saved and the complete attempt history goes into the reject log.

Malformed validator output has its own separate limit of three format attempts.
A well-formed negative or uncertain verdict is final for that candidate; the
pipeline may generate a different candidate within the question's remaining
budget. There is no nested generator retry loop. Transport retries remain governed
by `llm.max_retries`. Rerunning unchanged inputs reuses cached model responses.


Accepted targets are saved atomically in `generated_false_answers.jsonl`, keyed
by question and gold-answer fingerprint, with evidence hashes and verifier
receipts. One saved target is reused across all human documents for that question,
including when the document generator changes. Changed evidence or validator code
invalidates saved verification. Document generation checks every selected answer
before any model calls or corpus writes. It never generates its own target.

Stage 2 asks the verifier to select answer-bearing occurrence indices in source
text and titles. Only selected spans change. A single replacement pass preserves
full-date/year granularity and leaves unrelated occurrences alone. A semantic
seed check rejects changes to unrelated claims or broad identity substitutions
that alter other facts. The generated paraphrase must affirm the target, cease
supporting gold as the answer, preserve the seed's other claims, and remain
relevant. Unrelated mentions of a gold string are allowed when the semantic check
confirms they do not answer the question. Correct paraphrases undergo factual
preservation and relevance checks too, including full-human-corpus generation.

Generation uses the configured model as verifier by default. To use a separate
verifier, add this under `generation` (all normal LLM settings are supported):

```yaml
validation:
  max_similarity: 0.95
  llm:
    provider: local_openai
    base_url: http://localhost:8001/v1
    model: your-verifier-model
```

Semantic validation cannot be disabled. These judgments are operational checks,
not proofs of factual correctness; calibrate the chosen verifier against manual
reviews. Character-sequence similarity rejects copies at or above the configured
threshold relative to the source, false seed, or another sample for the same
source and condition. It is not corpus-wide semantic deduplication.

Each model, condition, temperature, sample, source, prompt/config signature has a
registered `corpus_0001.jsonl`-style file in shared `metadata.yaml`. Corpus rows
remain exactly `_id`, `title`, and `text`, using the original human document ID.
Each corpus has an append-only `corpus_0001.manifest` sidecar with source and row
hashes, question/gold/target linkage, selected edits, seed text, and validation
receipts. Keep corpus, manifest, and registry together. Writes use a per-corpus
lock; manifest receipts are written before corpus rows.

Resume compares fingerprints including source text/title, question, gold aliases,
false-answer receipt, prompts, model settings, and validator code. Changed inputs
raise a stale-output error. Use a new output directory or explicitly set
`resume: false` to regenerate the matching corpus. Changed generation settings
allocate a separate registered corpus. Existing corpora are retained; controlled
generation never imports unverified legacy rows. Loading false corpora without
matching manifest receipts fails. Old correct corpora remain loadable as legacy
artifacts but are not certified by the new checks.

The generic `evlab generate` command still supports correct-only generation and
requires verified saved answers for false conditions. Corpus paths accept a file,
directory, or glob. When old and new generations coexist, point an experiment at
the intended corpus files so older copies do not enter it accidentally.

All model calls are cached in SQLite. Reject logs include the stage, source,
edits, generated text when available, and failure reason. No model is invoked by
`generation-status`.

## RAG Reader Evaluation

`queries: rag` now runs the reader after retrieval and requires an explicit reader
configuration. Retrieval-only experiments retain their current behavior.

```yaml
queries: rag
reader:
  top_k: 10
  cache_path: cache/reader.sqlite
  llm:
    provider: local_openai
    base_url: http://localhost:8000/v1
    model: your-reader-model
```

Each retriever's output includes `answers.jsonl`: answer, parse validity, numbered
citations, context/cited document IDs, citation-range validity, gold aliases,
validated target, and classification. Manifest receipts restore targets; conflicting
targets for one question fail evaluation. Exact normalized concise answers are
classified as correct/target; mixed gold-and-target mentions are ambiguous,
mention-only or negated prose is other, malformed JSON is invalid, and abstentions
are separate. This deliberately conservative classifier does not infer semantic
adoption from prose. Citation-range validity does not certify citation support.
Metrics include category rates over all evaluated questions and target adoption
among questions with a selected false target.

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
