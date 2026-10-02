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

Default configs expect the NQ+ artefacts from the MVP next to this repo:

```text
../MT_MVP/data/nqPlus/clean/
../MT_MVP/data/cocktail/nq/corpus/human.jsonl
../MT_MVP/data/cocktail/nq/corpus/llama-2-7b-chat-tmp0.2.jsonl
```

You can also copy or symlink the files into `data/` and update `configs/data/nqplus.yaml`.

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
uv run evlab generate --config configs/generation/pilot-local.yaml --limit 5
```

Gemini:

```bash
export GEMINI_API_KEY=...
uv run evlab generate --config configs/generation/pilot-gemini.yaml --limit 5
```

All generation calls are cached in SQLite under `cache/llm.sqlite`.

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
