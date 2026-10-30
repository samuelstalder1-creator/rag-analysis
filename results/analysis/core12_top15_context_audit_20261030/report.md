# Core12 Top-15 context audit (2026-10-30)

All 18,240 actual BM25 Top-15 reader prompts fit within the configured 8,192-token server context. Counts include the official Qwen chat template and a 256-token output reservation. No documents were truncated and no model inference was requested.

Largest prompt: 4’168 tokens; total with output budget: 4’424; remaining: 3’768.

| Variant | Median prompt | P95 prompt | Maximum prompt | Maximum + 256 | Over 8192 |
|---|---:|---:|---:|---:|---:|
| human_only | 1938 | 2604 | 4141 | 4397 | 0 |
| human+qwen3_correct | 1884 | 2514 | 4074 | 4330 | 0 |
| human+gpt_oss_correct | 1895 | 2498 | 4026 | 4282 | 0 |
| human+qwen3_false | 1880 | 2527 | 4165 | 4421 | 0 |
| human+gpt_oss_false | 1953 | 2619 | 4147 | 4403 | 0 |
| human+gpt_oss_attack | 1960 | 2569 | 4168 | 4424 | 0 |
| human+qwen3_correct+qwen3_false | 1825 | 2470 | 4106 | 4362 | 0 |
| human+gpt_oss_correct+gpt_oss_false | 1897 | 2539 | 4085 | 4341 | 0 |
| human+qwen3_correct+gpt_oss_false | 1901 | 2536 | 4088 | 4344 | 0 |
| human+gpt_oss_correct+qwen3_false | 1832 | 2452 | 4103 | 4359 | 0 |
| human+qwen3_correct+gpt_oss_attack | 1899 | 2492 | 4109 | 4365 | 0 |
| human+gpt_oss_correct+gpt_oss_attack | 1903 | 2501 | 4106 | 4362 | 0 |

The server limit is set by `scripts/run_qwen3_4b_generation.sh --max-model-len 8192`; the output budget comes from `configs/readers/qwen3-4b-instruct-2507.yaml`. This audit assumes the GPU host uses the same tokenizer/chat template and unmodified experiment inputs.

Detailed counts and selected document IDs: `prompts.jsonl`; tokenizer file hashes and statistics: `summary.json`; reproduction script: `audit.py` (run from the repository root with `PYTHONPATH=src`).
