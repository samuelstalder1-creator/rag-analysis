# NQ+ Evidence Laundering Lab Blueprint

This repo implements the first executable version of the NQ+ blueprint from the thesis notes.

## Scope

- Load and validate NQ+ input from the MVP output.
- Build human, Cocktail-mixed and generated corpus conditions.
- Run sparse retrieval with BM25 and TF-IDF.
- Provide a dependency-light dense smoke retriever plus optional sentence-transformer retrieval.
- Generate correct and false evidence with local LLMs or Gemini through one client interface.
- Cache LLM calls in SQLite.
- Measure retrieval, source/displacement and answer labels.
- Expose everything through `uv run evlab ...`.

## Current Build Level

Implemented:

- `evlab data validate`
- `evlab run`
- `evlab generate`
- NQ+ loader
- corpus builder
- BM25, TF-IDF, hash dense, optional MiniLM/E5
- MiniLM full-corpus dense runs with embedding cache
- local OpenAI-compatible LLM client
- Ollama client
- Gemini client
- generation prompts and validation
- RAG context/reader primitives
- strict vs. relaxed retrieval evaluation
- source metrics
- tests and fixtures
- dense embedding cache with prefix reuse for mixed corpora

Since the first blueprint: semantic (LLM) validation of false answers and documents,
the NQ-Open+ build (`evlab data build-nqopen-plus`), reader sweeps with ratio/variant
grids, run status and input fingerprints (see README, "Run integrity") and a run
inventory (`evlab inventory`) are in place.

Still intentionally minimal:

- Context mitigations (MMR, text deduplication, one document per root) were removed
  as unused; reintroduce them together with the experiment that needs them.
- Preparing NQ+ from raw Cocktail + NQ-Open stays in the MVP repository.
- `recall@k` merges roots before the cutoff; use `context_recall@k` for what the
  reader actually receives.

## First Commands

```bash
uv sync
uv run pytest
uv run evlab data validate --config configs/data/nqplus.yaml
uv run evlab run configs/experiments/rq1_displacement.yaml
uv run evlab run configs/experiments/rq1_dense_human_minilm.yaml
uv run evlab run configs/experiments/rq1_dense_mixed_minilm.yaml
```

## Generation Commands

Local vLLM or llama.cpp server:

```bash
uv run evlab generate --config configs/generation/pilot-local.yaml --limit 10
```

Ollama:

```bash
uv run evlab generate --config configs/generation/pilot-ollama.yaml --limit 10
```

Gemini:

```bash
export GEMINI_API_KEY=...
uv sync --extra gemini
uv run evlab generate --config configs/generation/pilot-gemini.yaml --limit 10
```
