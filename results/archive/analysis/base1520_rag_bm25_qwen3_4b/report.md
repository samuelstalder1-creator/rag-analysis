# Base1520 BM25 RAG results

Updated: 2026-10-26 19:08:55 UTC

Status: 4/4 variants complete. Final comparison.

Each completed variant covers all 1,520 nqopen1520 questions. Reader: Qwen/Qwen3-4B-Instruct-2507; temperature 0; top 5 BM25 documents as context.

| Variant | Correct answers | Accuracy | False-target answers | Abstentions | Relaxed recall@5 |
|---|---:|---:|---:|---:|---:|
| human_only | 757/1520 | 49.80% | 4 (0.26%) | 19.87% | 73.46% |
| human_correct | 887/1520 | 58.36% | 4 (0.26%) | 14.08% | 82.18% |
| human_false | 445/1520 | 29.28% | 494 (32.50%) | 14.41% | 87.37% |
| human_correct_false | 726/1520 | 47.76% | 253 (16.64%) | 11.58% | 89.24% |

## Paired answer changes

- **human_correct vs human_only:** +8.55 percentage points; 158 questions become correct, 28 become incorrect (net +130).
- **human_false vs human_only:** -20.53 percentage points; 39 questions become correct, 351 become incorrect (net -312).
- **human_correct_false vs human_only:** -2.04 percentage points; 145 questions become correct, 176 become incorrect (net -31).

## Effect of combining correct and false evidence

Adding correct rewrites to the poisoned corpus increases accuracy by 18.49 percentage points and reduces false-target answers from 32.50% to 16.64%.

Adding poisoned documents to the correct-rewrite corpus reduces accuracy by 10.59 percentage points. Correct rewrites mitigate the observed degradation, but the combined variant remains below the human-only baseline.

## Retrieval and answer interpretation

| Variant | Strict recall@5 | Correct rewrites in top 5 (mean) | False docs in top 5 (mean) | Valid answer JSON |
|---|---:|---:|---:|---:|
| human_only | 73.46% | 0.000 | 0.000 | 1520/1520 |
| human_correct | 70.71% | 0.961 | 0.000 | 1520/1520 |
| human_false | 70.87% | 0.000 | 1.080 | 1520/1520 |
| human_correct_false | 66.10% | 0.889 | 1.020 | 1520/1520 |

Relaxed recall counts a human document, its correct rewrite, or its poisoned version as the same relevant source. It therefore measures source retrieval, not factual correctness. Strict recall counts only the human originals. `rewrite@5` and `poison@5` are document counts, not percentages.

False-target answers can occur even in the human-only baseline; compare the attacked variants against that baseline before attributing changes to poisoned evidence.

## Data provenance and limits

- Fixed corpora: `data/generated/base-1520/corpus_correct.jsonl` and `corpus_false_complete.jsonl`, with 1,670 documents each. They are distinct from the separate Qwen generation directories.
- The sweep configuration labels corpus generation as Qwen3.5-2B and records eight manually written false-document supplements. The reader is Qwen3-4B; this report does not infer corpus generation provenance from the reader model.
- Answer correctness follows the repository alias-based classifier; deterministic document checks and valid JSON do not establish factual validity.
- This is a single run on one fixed question set; no uncertainty estimates or broader generalization claims are made.

## Run artifacts

- human_only: `results/runs/base1520_rag_bm25__human_only_2e30296248/` (summary, metrics, per-question answers and retrieval).
- human_correct: `results/runs/base1520_rag_bm25__human_correct_a9a6bfb836/` (summary, metrics, per-question answers and retrieval).
- human_false: `results/runs/base1520_rag_bm25__human_false_45ed9c0e86/` (summary, metrics, per-question answers and retrieval).
- human_correct_false: `results/runs/base1520_rag_bm25__human_correct_false_0bbc40dc0f/` (summary, metrics, per-question answers and retrieval).
