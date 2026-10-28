#!/usr/bin/env bash
# Resume the fresh 32-combination BM25 + DeepSeek reader evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-local}"
export MAX_MODEL_LEN=16384
# A 16k context needs 2.25 GiB of KV cache on this L4; 90% leaves only 2.15 GiB.
export GPU_MEMORY_UTILIZATION=0.92
config=configs/experiments/nqopen1520_all_sources_deepseek_bm25.yaml

server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

server_ready() {
  .venv/bin/python - <<'PY'
import json
import urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5) as response:
        models = json.load(response)['data']
    assert any(m['id'] == 'deepseek-ai/DeepSeek-R1-0528-Qwen3-8B' for m in models)
except Exception:
    raise SystemExit(1)
PY
}

if ! server_ready; then
  echo "$(date -u --iso-8601=seconds): starting DeepSeek evaluation server"
  bash scripts/run_deepseek_generation.sh serve &
  server_pid=$!
  deadline=$((SECONDS + 1200))
  until server_ready; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo 'DeepSeek server exited before becoming ready' >&2
      wait "$server_pid"
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo 'DeepSeek server did not become ready within 20 minutes' >&2
      exit 1
    fi
    sleep 5
  done
fi

# Check real RAG prompts with every source before committing to the full sweep.
echo "$(date -u --iso-8601=seconds): checking two answers with all sources"
.venv/bin/python - <<'PY'
from evlab.config import load_config
from evlab.experiments.sweep import expand_sweep
from evlab.experiments.runner import run_experiment
config = load_config('configs/experiments/nqopen1520_all_sources_deepseek_bm25.yaml')
_, _, smoke = expand_sweep(config)[-1]
smoke['experiment'] += '__smoke'
smoke['query_limit'] = 2
smoke['output']['dir'] = 'results/evaluations/deepseek_bm25_20261027/smoke'
summary = run_experiment(smoke)
if summary['retrievers']['bm25']['answer_invalid_rate']:
    raise SystemExit('DeepSeek reader smoke check produced invalid answers; inspect smoke answers.jsonl')
print('DeepSeek reader smoke check passed', flush=True)
PY

echo "$(date -u --iso-8601=seconds): running BM25 + DeepSeek evaluation"
.venv/bin/python -m evlab.cli sweep "$config"
touch results/evaluations/deepseek_bm25_20261027/COMPLETE
echo "$(date -u --iso-8601=seconds): evaluation complete"
