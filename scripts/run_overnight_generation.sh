#!/usr/bin/env bash
# Run both nqopen1520 stages with the chosen model, starting its server if needed.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export GENERATION_MODEL="${GENERATION_MODEL:-deepseek-ai/DeepSeek-R1-0528-Qwen3-8B}"
GENERATION_LAUNCHER="${GENERATION_LAUNCHER:-scripts/run_deepseek_generation.sh}"

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
import os
import urllib.request
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5) as response:
        models = json.load(response)["data"]
    assert any(m["id"] == os.environ["GENERATION_MODEL"] for m in models)
except Exception:
    raise SystemExit(1)
PY
}

if ! server_ready; then
  echo "$(date -u --iso-8601=seconds): starting $GENERATION_MODEL server"
  bash "$GENERATION_LAUNCHER" serve &
  server_pid=$!
  deadline=$((SECONDS + 1200))
  until server_ready; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Model server exited before becoming ready" >&2
      wait "$server_pid"
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Model server did not become ready within 20 minutes" >&2
      exit 1
    fi
    sleep 5
  done
fi

echo "$(date -u --iso-8601=seconds): resuming correct and false nqopen1520 generation"
bash "$GENERATION_LAUNCHER" all
echo "$(date -u --iso-8601=seconds): generation finished"
