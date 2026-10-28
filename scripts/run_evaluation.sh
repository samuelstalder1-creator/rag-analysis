#!/usr/bin/env bash
# Run one reader-evaluation sweep against a local vLLM server.
#
# Usage: bash scripts/run_evaluation.sh <profile>
#   <profile> names scripts/profiles/<profile>.env, which defines
#   CONFIG, SERVE_SCRIPT, MODEL_ID and EVAL_DIR (and may export server settings).
#
# The run is resumable: finished answers are kept (see output.resume) and the
# COMPLETE marker is only written after the whole sweep succeeded.
# Set EVAL_CHECK_ONLY=1 to validate the profile without starting anything.
set -euo pipefail
cd "$(dirname "$0")/.."

profile="${1:?usage: bash scripts/run_evaluation.sh <profile>}"
profile_file="scripts/profiles/${profile}.env"
[[ -f "$profile_file" ]] || { echo "unknown profile: $profile ($profile_file)" >&2; exit 2; }
# shellcheck disable=SC1090
source "$profile_file"
for name in CONFIG SERVE_SCRIPT MODEL_ID EVAL_DIR; do
  [[ -n "${!name:-}" ]] || { echo "profile $profile does not set $name" >&2; exit 2; }
done
[[ -f "$CONFIG" && -f "$SERVE_SCRIPT" ]] || { echo "missing $CONFIG or $SERVE_SCRIPT" >&2; exit 2; }
if [[ "${EVAL_CHECK_ONLY:-0}" == "1" ]]; then
  echo "profile $profile ok: config=$CONFIG model=$MODEL_ID output=$EVAL_DIR"
  exit 0
fi

PYTHON="${PYTHON:-.venv/bin/python}"
export PYTHONUNBUFFERED=1
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-local}"
export CONFIG MODEL_ID EVAL_DIR

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
  "$PYTHON" - <<'PY'
import json
import os
import urllib.request
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5) as response:
        models = json.load(response)['data']
    assert any(m['id'] == os.environ['MODEL_ID'] for m in models)
except Exception:
    raise SystemExit(1)
PY
}

if ! server_ready; then
  echo "$(date -u --iso-8601=seconds): starting $MODEL_ID evaluation server"
  bash "$SERVE_SCRIPT" serve &
  server_pid=$!
  deadline=$((SECONDS + 1200))
  until server_ready; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "$MODEL_ID server exited before becoming ready" >&2
      wait "$server_pid"
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "$MODEL_ID server did not become ready within 20 minutes" >&2
      exit 1
    fi
    sleep 5
  done
fi

# Check real RAG prompts with every source before committing to the full sweep.
echo "$(date -u --iso-8601=seconds): checking two answers with all sources"
"$PYTHON" - <<'PY'
import os
from evlab.config import load_config
from evlab.experiments.sweep import expand_sweep
from evlab.experiments.runner import run_experiment
config = load_config(os.environ['CONFIG'])
_, _, smoke = expand_sweep(config)[-1]
smoke['experiment'] += '__smoke'
smoke['query_limit'] = 2
smoke['output']['dir'] = os.environ['EVAL_DIR'] + '/smoke'
summary = run_experiment(smoke)
if summary['retrievers']['bm25']['answer_invalid_rate']:
    raise SystemExit(f"{os.environ['MODEL_ID']} reader smoke check produced invalid answers; inspect smoke answers.jsonl")
print(f"{os.environ['MODEL_ID']} reader smoke check passed", flush=True)
PY

echo "$(date -u --iso-8601=seconds): running BM25 + $MODEL_ID evaluation"
"$PYTHON" -m evlab.cli sweep "$CONFIG"
touch "$EVAL_DIR/COMPLETE"
echo "$(date -u --iso-8601=seconds): evaluation complete"
