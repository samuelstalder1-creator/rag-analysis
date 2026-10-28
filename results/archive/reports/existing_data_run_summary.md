# Existing Data Run Summary

Run date: 2026-10-01

## Validation

```text
retrieval_queries: 2248
rag_queries: 2172
generation_candidates: 1360
qrels: 2808
human_docs: 104194
cocktail_llm_docs: 104194
```

## Completed Runs

| Run | Corpus | Retriever | strict nDCG@10 | strict Recall@10 | strict Recall@100 | LLM@10 | Generated before Human |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rq1_human_55f1330e4f` | Human | BM25 | 0.6452 | 0.7794 | 0.9096 | 0.0000 | 0.0000 |
| `rq1_human_55f1330e4f` | Human | TF-IDF | 0.5105 | 0.6736 | 0.8772 | 0.0000 | 0.0000 |
| `rq1_dense_human_minilm_c2756ac6df` | Human | MiniLM | 0.8321 | 0.9400 | 0.9898 | 0.0000 | 0.0000 |
| `rq1_displacement_dd6aef1637` | Human + Cocktail-Llama | BM25 | 0.4870 | 0.7128 | 0.8799 | 0.4924 | 0.6268 |
| `rq1_displacement_dd6aef1637` | Human + Cocktail-Llama | TF-IDF | 0.3862 | 0.5833 | 0.8277 | 0.4887 | 0.5387 |
| `rq1_dense_mixed_minilm_bea831608a` | Human + Cocktail-Llama | MiniLM | 0.6719 | 0.9040 | 0.9806 | 0.4874 | 0.4771 |

## Relaxed Metrics for Mixed Corpus

| Retriever | relaxed nDCG@10 | relaxed Recall@10 | relaxed Recall@100 |
| --- | ---: | ---: | ---: |
| BM25 | 0.6553 | 0.7889 | 0.8990 |
| TF-IDF | 0.5228 | 0.6799 | 0.8492 |
| MiniLM | 0.8390 | 0.9454 | 0.9832 |

## Blocked Runs

`rq3_correct_multiplication` and `rq4_false_consensus` were started, but both require:

```text
data/generated/pilot/generated_docs.jsonl
```

That file is not existing data yet. It must be produced by:

```bash
uv run evlab generate --config configs/generation/pilot-local.yaml --limit 100
```

or:

```bash
uv run evlab generate --config configs/generation/pilot-gemini.yaml --limit 100
```

After that, `rq3` and `rq4` can be run.

`rq1_dense_mixed_e5` was started, but `intfloat/e5-base-v2` is not available in the local HuggingFace cache and outgoing traffic is disabled. To run E5, download/cache the model once or set `dense_local_files_only: false` in the config when network access is available.

## Notes

- Fixed relaxed retrieval evaluation: human and generated copies are deduplicated by original document before scoring, so nDCG stays <= 1.
- Optimized TF-IDF with posting lists and stopword filtering so full existing-data runs complete.
- Added dense MiniLM runs for Human and Mixed corpora.
- Added dense embedding cache and prefix reuse: the mixed MiniLM run reuses Human embeddings and only encodes the Cocktail-Llama half.
