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

## Corpus Mixes: What Goes Into a Run

A `corpus.sources` block lets one config control exactly which generated
documents enter the retrieval corpus, and how many. Templates and ready-made
sweeps live in `configs/experiments/mix/`:

| Config | What it varies |
| --- | --- |
| `template.yaml` | one run, every option documented |
| `sweep_correct_ratio.yaml` | share of the corpus with a correct Qwen paraphrase (0–100 %), `add` vs `replace` |
| `sweep_false_ratio.yaml` | share of attackable questions that receive false documents, `add` vs `replace` |
| `variants_false_correct.yaml` | human only / correct only / false only / false replace / false + correct / false in a 50 % AI corpus |
| `rag_false_correct.yaml` | same variants with the reader (answer adoption of the false target) |

```bash
uv run evlab corpus configs/experiments/mix/template.yaml          # composition only, no retrieval
uv run evlab run    configs/experiments/mix/template.yaml          # one run
uv run evlab sweep  configs/experiments/mix/variants_false_correct.yaml --dry-run   # compositions of all variants
uv run evlab sweep  configs/experiments/mix/variants_false_correct.yaml             # all runs + summary.csv
```

### Config

```yaml
corpus:
  seed: 13
  sources:                          # named pools; listing a source adds nothing by itself
    qwen_correct:  {path: data/generated/qwen35-2b-all}
    false_det:     {path: data/generated/false-deterministic}
    cocktail_llama:
      path: data/cocktail/nq/corpus/llama-2-7b-chat-tmp0.2.jsonl
      origin: cocktail_llm
  background:                       # corpus-wide, sampled per human document
    qwen_correct: {ratio: 0.5, mode: add, scope: corpus, k: 1}
  targeted:                         # aimed at the evaluated questions, sampled per question
    query_ratio: 0.3
    false_docs:   {source: false_det,    k: 1, mode: add}
    correct_docs: {source: qwen_correct, k: 1, mode: add}
```

- **`background`**: `ratio` is the share of human documents (those the source
  covers) whose generated copy enters. `scope: relevant` limits the pool to
  the qrel documents of the evaluated questions. `background` may be a list
  of entries or a mapping keyed by source name.
- **`targeted`**: a question is *attackable* if the false source has at least
  one document for it (or, without `false_docs`, a correct copy). `query_ratio`
  is the share of attackable questions that get targeted documents. Only
  documents that carry the answer (`docs_with_answer`) are targeted. With `k: 0`
  the question set stays the same but no documents are added. This is the
  baseline for the `targeted_*` metrics.
- **`mode`**: `add` keeps the human document next to the copy (the corpus
  grows). `replace` removes the human document (the corpus size stays the
  same; for false documents only the false version remains).
- **`k`**: copies per document, limited by the samples generated
  (`samples_per_h`). Missing copies are reported as `k_shortfall`, not invented.
- In `add` mode, generated share = ratio / (1 + ratio): ratio 1.0 gives a 50 %
  generated corpus. In `replace` mode, ratio equals the share. Every run
  reports the actual `generated_share`.
- **Nested sampling**: units are ranked by a seeded hash. With the same seed,
  ratio 0.25 selects a subset of ratio 0.5, so a sweep adds contamination step
  by step instead of drawing new samples. Change `seed` (or an entry's `salt`)
  for an independent draw.
- Generated doc IDs inside a mix are `gen::<source name>::...`, so two
  generation folders cannot collide. Config errors (unknown keys or sources,
  ratio outside 0–1, mixing with the legacy `base`/`generated_docs`/`k_*`
  keys) fail before anything is loaded.

YAML reads a bare `false:` key as a boolean, which is why the targeted keys
are named `false_docs` / `correct_docs`.

### Outputs

Each run directory additionally contains `corpus_selection.json` (counts per
source, pool sizes, shortfalls and the list of targeted question IDs);
`summary.json` carries the same summary without the ID list. When only some
evaluated questions are targeted, every retrieval and source metric is also
reported on the targeted questions alone with the prefix `targeted_`
(e.g. `targeted_false@10`, `targeted_strict_recall@10`).

Retrieval metrics are computed only over the questions that were evaluated
(`queries`). Before this change, runs with `queries: generation` or `rag`
counted all other retrieval questions as zero-recall misses.

### Sweeps

A sweep config is a normal experiment config plus `variants` (named
overrides) and/or `sweep` (a grid). Paths are dotted. Numeric segments index
lists:

```yaml
variants:
  human_only: {}
  false_replace:
    corpus.targeted.false_docs.k: 1
    corpus.targeted.false_docs.mode: replace
sweep:
  corpus.background.qwen_correct.ratio: [0.0, 0.25, 0.5, 1.0]
```

Every variant runs at every grid point. Each run is a regular run under
`output.dir` (with its own config-hash run ID). The sweep writes
`summary.csv` / `summary.jsonl` (one row per run and retriever: parameters,
corpus composition and all metrics) to
`sweep_output.dir/<experiment>_<hash>/`. `--dry-run` writes
`corpus_preview.csv` instead. Loaded sources are cached within a sweep. In
`add` mode the generated part grows as a prefix, so dense embedding caches
are reused between steps.

## Retrievers and Reader in the Config

### Retrievers

`retrievers` lists what runs; each entry gets its own metrics, `retrieval.jsonl`
and (optionally) reader answers. `retriever_definitions` says how each one is
configured:

```yaml
retrievers: [bm25, tfidf, bm25_tfidf_rrf]   # only BM25: [bm25]
retriever_definitions:
  bm25:   {type: bm25, k1: 1.5, b: 0.75, remove_query_stopwords: true}
  tfidf:  {type: tfidf}
  bm25_tfidf_rrf:
    type: hybrid
    members: [bm25, tfidf]        # any retrievers, incl. e5/minilm or other definitions
    fusion: rrf                   # rrf | weighted
    rrf_k: 60
    weights: {bm25: 1.0, tfidf: 1.0}
    depth: 100                    # candidates per member (default retrieval.top_k)
    normalization: minmax         # weighted only: minmax | zscore | none
retrieval:
  top_k: 100
```

- Types: `bm25`, `tfidf`, `e5`, `minilm`, `dense` (any sentence-transformers
  `model`), `hash_dense` (tests) and `hybrid`. Several definitions of one type
  are allowed (e.g. `bm25_k1_09: {type: bm25, k1: 0.9}`).
- A built-in name without a definition uses the shared `retrieval` block
  (`k1`, `b`, `dense_model`, `dense_device`, ...), so older configs still run
  unchanged.
- A definition can be a file path (`bm25: configs/retrievers/bm25.yaml`).
  `retrievers` can also be a mapping `name -> definition`, in which case every
  entry runs.
- `rrf` fuses by rank: `sum w / (rrf_k + rank)`. `weighted` sums per-query
  normalised scores; a document missing from a member's list scores 0 for
  that member.
- Each base retriever is indexed once per run, at the largest depth needed.
  Hybrids reuse those runs, so `[bm25, tfidf, hybrid]` costs about the same
  as `[bm25, tfidf]`. Hybrid members do not have to be in `retrievers`
  themselves.
- Unknown names, keys, fusion methods or cycles fail before the corpus is
  built. The resolved specs (with all defaults) go to `resolved.json` and
  `summary.json`.

`configs/experiments/mix/compare_retrievers.yaml` runs BM25, TF-IDF, RRF and a
weighted 0.7/0.3 fusion on the same poisoned corpus.

### Reader (answer LLM)

```yaml
reader:
  llm: configs/readers/qwen35-2b.yaml   # preset, or inline {provider, base_url, model, ...}
  temperature: 0.0
  seed: 13
  top_k: 5                              # documents in the context
  prompt: prompts/reader_v1.txt         # template with {question} and {context}
  retrievers: [bm25_tfidf_rrf]          # which retrievers feed the reader (default: all)
  cache_path: cache/reader.sqlite
```

- Providers: `local_openai` (vLLM/OpenAI-compatible: `base_url`, `model`,
  `api_key_env`, `max_tokens`, `timeout`, `max_retries`, `retry_delay`),
  `ollama`, `gemini` (`model`, `api_key_env`) and `abstain`.
  Presets are in `configs/readers/` (`qwen35-2b.yaml` matches the vLLM
  server in `runit.md`, which serves a 4096-token context: keep `top_k` and
  `max_tokens` within it).
- The reader runs when `queries: rag` or `reader.enabled: true`. With
  `reader.retrievers` you can compute retrieval metrics for every retriever
  but spend LLM calls only on the ones you name.
- `prompt` is a file path or an inline template; literal braces are written
  as `{{ }}`. Without it, the built-in prompt (= `prompts/reader_v1.txt`) is
  used.
- `summary.json` and the sweep CSV record provider, model, temperature,
  `top_k` and a prompt hash. If the reader, prompt or a retriever definition
  comes from a file, the run ID includes the file contents, so editing a
  preset or prompt produces a new run directory instead of overwriting.
- To compare reader models, add sweep variants such as
  `reader.llm: configs/readers/gemini.yaml`.

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
that alter other facts. The false-document prompt receives the question, target
answer, edited title and edited body. Its generated paraphrase must support the
target as the unambiguous answer to that exact question and cease supporting gold
as the answer. Unrelated background details may be omitted; preserving every seed
claim is not required. The optional semantic judge checks source support for gold,
seed and generated support for the target, absence of gold support and conflicting
answers. Deterministic validation remains rule-based and records
`answerability_verified: false` with `validation_goal: query_target_answerability`. Unrelated mentions of a gold string are allowed when the semantic check
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

## NQOpen+ Subset and Imported False Targets

`data/nqopen+/` combines the **same 1,614 effective generation-eligible NQ+ questions**
with Cocktail IDs/qrels and complete local NQ-Open CSV records. Question selection
is fixed before target validation; review cases remain available for RAG testing.
Original gold answers, aliases and IDs are retained. No answer-generator call is
needed in this workflow.

```bash
uv run evlab data build-nqopen-plus --config configs/data/nqopen-plus.yaml
uv run evlab data validate --config configs/data/nqopen-plus.yaml
uv run evlab generate-false-documents --config configs/generation/nqopen-plus-false.yaml --dry-run
uv run evlab generate-false-documents --config configs/generation/nqopen-plus-false.yaml --limit 5
```

The builder joins only unique matching question texts and checks Cocktail ID/text
identity and qrel consistency. The current subset contains 1,614 RAG/generation
queries, 1,819 answer-bearing documents and 1,933 relevant documents/qrels.
`queries.jsonl` retains raw `correct_answers`, `incorrect_answers`, CSV fields and
source metadata. `qrels/test.tsv` contains the subset's original qrels;
`corpus/relevant.jsonl` is a convenient relevant-document extract. Retrieval still
uses the full **104,194-document human corpus**, referenced by the configs and
`dataset.json`; the extract is not a replacement background corpus.

`false_answers.jsonl` contains **1,129 usable imported targets**, with provenance
and deterministic validation receipts. **485 questions need review**; they remain
in `queries.jsonl` and are listed in `false_answer_review.jsonl`. Imported targets
are checked against all supplied correct answers, legacy type/alias rules and
source occurrences. Some `incorrect_answers` actually equal gold (112 proposals),
while other rejected entries may be legitimate targets blocked by conservative
format rules. Eleven questions lack a matching gold alias in their sources;
review reasons overlap. Rule acceptance does not establish semantic incorrectness.
No target is silently invented or replaced with an LLM fallback.

The nqopen+ document workflow now additionally enforces source constraints before
model calls: the **complete gold answer must occur in every assigned original
document**, and the false target must occur in none. Title and body are checked
together after Unicode/whitespace normalization, with equivalent full date and
number forms. A year-only alias does not establish a complete date. These checks
are also applied on resume to existing target receipts. The 1,129/485 counts above
describe the initial import; source-constrained readiness must be read from the
current `--dry-run` or a rebuilt dataset.

The separate `false_answer_review_cleaned.jsonl` retains all 485 reviewed rows and
now records `validation.source_constraints.documents` per source ID. Of these,
455 satisfy both presence constraints; 30 are blocked because at least one source
lacks the complete gold answer. Every cleaned target is absent from its original
sources under this check. Context/gold review notes remain independent of these
lexical constraints. This cleaned review file is not a replacement for the
receipt-bearing target file and is not automatically activated.

The false-document config reads this target file directly using
`false_answers_file` and writes to `data/generated/nqopen-plus-false/`.
`skip_missing_false_answers: true` reports and skips blocked targets, without
changing the dataset's 1,614-question evaluation scope. Fingerprints detect stale
questions, sources and validation code. Rebuild after such changes. The answer
model command refuses to overwrite imported targets. Previous generated corpora
and saved answers remain separate.

RAG configs:

```bash
# Human-only baseline: all 1,614 queries; no generated corpus required.
uv run evlab run configs/experiments/nqopen-plus-human.yaml --limit 5
# After generating false documents: human / false-add / false-replace.
uv run evlab sweep configs/experiments/mix/nqopen-plus-rag.yaml --limit 5
```

Both configs use BM25 and the configured Qwen reader. Imported ready targets are
also available to reader evaluation in the human-only baseline; review cases have
no selected target. A generated document with a different target is rejected.
Target adoption is scored on queries with selected targets; actual attack coverage
is separately reported by corpus selection. Remove `--limit` for all questions.

## Source-Grounded Resolution of the 48 Context Cases

A separate revision in `data/nqopen+/resolved/` resolves the 48 context-review
cases using unchanged relevant source documents. It contains all 1,614 queries,
1,548 ready targets and 66 source-blocked cases. **43 questions were refined,
five gold answers corrected, eight targets changed and five evidence assignments
changed.** This is a versioned revision, not unchanged original-benchmark parity.
Original questions, answers and qrels are retained in per-case lineage. Changed
questions use reviewed evidence qrels (1,926 total); 1,814 answer documents remain.

The `context_cases_resolved.jsonl` artifact includes each original/corrected pair,
its full source text and presence checks. Gold occurs in every assigned source;
the false target is absent from both assigned and all originally relevant sources.
The assistant reviewed answer claims in context; no independent judge is claimed.
For example, the Star Wars passages support a novelization preceding the film,
while the South Korea passage counts all Games rather than only Winter Games.

```bash
uv run python scripts/resolve_nqopen_context_cases.py --config configs/data/nqopen-plus-resolved.yaml
uv run evlab data validate --config configs/data/nqopen-plus-resolved.yaml
uv run evlab generate-false-documents --config configs/generation/nqopen-plus-resolved-false.yaml --dry-run
uv run evlab generate-false-documents --config configs/generation/nqopen-plus-resolved-false.yaml --limit 5
```

The resolved false-document config retries passages that exceed the 0.95
similarity limit up to three times (`false_document_max_attempts: 3`). Each
retry adds rewrite feedback and changes the sampling seed and temperature.
Copies that still fail are rejected, with the attempted responses retained in
the reject log. The revised rewrite prompt uses a new cache key, so previously
cached copied passages do not require clearing the cache.

To retry missing/rejected documents with fresh rewrite sampling, keep the same
config and `resume: true`, and increment `--retry-seed` for each fresh retry:

```bash
uv run evlab generate-false-documents \
  --config configs/generation/nqopen-plus-resolved-false.yaml \
  --retry-seed 1
```

Completed documents are skipped and missing documents are appended to the same
corpus. The retry seed is recorded in new document receipts and does not
invalidate existing fingerprints. Reusing a retry seed reuses its cache entries.
Reject logs retain historical failures; a reject record alone does not mean the
document is still missing. Input/span problems need correction separately;
changing the rewrite seed does not fix them. `--dry-run` reports input readiness,
including blocked questions, rather than the number of missing output rows.

For surname-only bodies, the editor expands the first surname to the complete
target name only when the original title explicitly identifies the gold person.
Without that title context, the body remains rejected. Album titles such as
`Hold On` do not trigger surname replacement of ordinary words such as `on`.

The corrected values are authored in
`configs/data/nqopen-plus-context-resolutions.json`. Rebuild refuses missing cases,
unsupported document reassignment, absent gold or targets present in sources.
Ready-target receipts are regenerated against the revised question/gold/aliases.
Use `configs/experiments/nqopen-plus-resolved-human.yaml` for the RAG baseline or
`configs/experiments/mix/nqopen-plus-resolved-rag.yaml` for human/add/replace runs.
New generation outputs are isolated under `data/generated/nqopen-plus-resolved-false/`.

## Generate All Correct Answer Documents

Retry rejected/missing correct documents with fresh sampling while preserving
completed rows in the registered corpus:

```bash
uv run evlab generate-correct-documents \
  --config configs/generation/nqopen-plus-resolved-correct.yaml \
  --retry-seed 1
```

Keep `resume: true`; increment the retry seed for each fresh round. The same
option is available for `repair-correct-documents`. Existing reject logs are
historical and may include documents that succeeded in a later round.

Generate one new correct copy per original answer-bearing document in the NQ+
generation subset, independent of the old 104k generated corpus:

```bash
uv run evlab generate-correct-documents --config configs/generation/qwen35-2b-correct-candidates.yaml --dry-run
uv run evlab generate-correct-documents --config configs/generation/qwen35-2b-correct-candidates.yaml --limit 5
uv run evlab generate-correct-documents --config configs/generation/qwen35-2b-correct-candidates.yaml
```

Scope is `docs_with_answer` for generation-eligible questions, not every qrel or
reader question. The current audit covers 1,819 source documents for 1,614 questions:
1,802 have a gold alias in the original title/body and are selected; 17 are flagged
for review. Of the selected sources, 21 contain an answer alias only in the title.
The generator receives the question, gold answer, original title and body. It may expand an unambiguous
shortened subject from the title, but must preserve the original answer claim and
include an answer mention in its body without inventing relations.

The current output directory is `data/generated/qwen35-2b-correct-candidates-v3/`.
The previous pilot remains in `data/generated/qwen35-2b-correct-candidates/`. Registered
`corpus_*.jsonl` files contain only `_id`, `title`, and `text`, with the **original
human IDs and titles**. Successful rows can be manually substituted by `_id` in a
copy of the old 104k corpus. The pipeline does not merge, replace or modify that
file. Review the audit, reject log and coverage before substituting rows; failed
or unsupported cases produce no replacement. `--limit` counts documents, not
questions. Resume skips matching accepted copies. Up to three attempts are made
per document. Prompt v3 prioritizes answerability for the exact question and
requires different wording and sentence structure. Unrelated background details
may be omitted; preserving every source claim is not required. Retries use specific feedback for copying versus missing answer forms,
without repeating the failed rewrite as a copying anchor. Temperatures rise from
0.3 to 0.5 to 0.7 (`correct_retry_temperature_step: 0.2`, capped at 1.0).
The threshold is unchanged; exhausted cases remain rejected. Default checks cover
gold-alias presence, nonempty output and the
near-duplicate threshold. These rules cannot verify answerability; receipts mark
`answerability_verified: false`. With a calibrated semantic verifier
(`validation.mode: semantic`, `validation.llm`), checks assess source support, the
correct answer-bearing relation and absence of conflicting answers to the same
question. Unrelated factual drift is not a rejection reason. Both modes record
`validation_goal: query_answerability`.

## Repair Correct Copies Missing Gold Answers

Regenerate only existing correct copies that lack every annotated gold alias in
their body. The repair starts from the original human body with a prompt that
explicitly preserves an answer mention and its claim. Matching correct copies are
not regenerated. Cases where the original body also lacks a gold alias are flagged
for review rather than filled with an unsupported answer.

```bash
uv run evlab repair-correct-documents --config configs/generation/qwen35-2b-correct-repair.yaml --dry-run
uv run evlab repair-correct-documents --config configs/generation/qwen35-2b-correct-repair.yaml --limit 5
uv run evlab repair-correct-documents --config configs/generation/qwen35-2b-correct-repair.yaml
```

The current audit covers 1,819 generation-source documents, not the full 104k
corpus: 270 generated bodies lack a gold alias; 238 have one in the original body
and can be repaired automatically, while 32 need review. This is a literal alias
audit, not a semantic finding that all 270 lost the answer claim.

Outputs go to `data/generated/qwen35-2b-correct-repair/`, with the normal compact
corpus, manifest, registry, audit and separate reject log. This is a replacement
subset, not a new full 104k corpus. Use it to replace the corresponding original
copies by source ID rather than adding both versions as independent evidence.
The baseline and source queries are never overwritten. Resume skips matching
successful repairs; each document has up to `correct_max_attempts: 3` attempts.
The default deterministic checks require a gold alias and sufficient rewording;
receipts remain `semantic_verified: false`. Set `validation.mode: semantic` and
configure a calibrated `validation.llm` for the same query-answerability checks.

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
