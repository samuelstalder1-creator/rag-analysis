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
uv sync --extra dense --extra yaml --group dev
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

Install vLLM in the Linux environment if it is not available yet:

```bash
uv add vllm
```

Start Qwen3.5-2B as an OpenAI-compatible server:

```bash
export LOCAL_LLM_API_KEY=local

uv run vllm serve Qwen/Qwen3.5-2B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 4096
```

Check that the server is reachable:

```bash
curl http://localhost:8000/v1/models
```

Keep this process running while generation runs in another terminal.

## Qwen3.5-2B Generation

This runs all generation candidates and writes progress continuously, so it can
be resumed. Output is stored under `data/generated/qwen35-2b-all`.

If a vLLM/OpenAI-compatible server is already running on port `8000`:

```bash
bash scripts/run_qwen35_2b_all.sh
```

To also start vLLM in the background:

```bash
START_VLLM=1 bash scripts/run_qwen35_2b_all.sh
```

Monitor:

```bash
tail -f logs/qwen35_2b_generation_*.log
wc -l data/generated/qwen35-2b-all/generated_docs.jsonl
wc -l data/generated/qwen35-2b-all/rejects.jsonl
```

Stop:

```bash
kill "$(cat logs/qwen35_2b_generation.pid)"
```

Resume:

```bash
bash scripts/run_qwen35_2b_all.sh
```

Reset a failed run that produced only rejects:

```bash
rm -f data/generated/qwen35-2b-all/generated_docs.jsonl
rm -f data/generated/qwen35-2b-all/rejects.jsonl
bash scripts/run_qwen35_2b_all.sh
```

Main outputs:

```text
data/generated/qwen35-2b-all/generated_docs.jsonl
data/generated/qwen35-2b-all/rejects.jsonl
cache/llm_qwen35_2b.sqlite
logs/qwen35_2b_generation.pid
```

## Other Useful Runs

```bash
uv run evlab run configs/experiments/rq1_displacement.yaml
uv run evlab run configs/experiments/rq1_dense_human_minilm.yaml
uv run evlab run configs/experiments/rq1_dense_mixed_minilm.yaml
uv run evlab report --run-dir results/runs
```
