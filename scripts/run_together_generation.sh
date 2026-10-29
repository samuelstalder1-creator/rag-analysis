#!/usr/bin/env bash
# Source-grounded GPT-OSS correct/false generation through Together; no GPU server.
# Usage: bash scripts/run_together_generation.sh {dry-run|smoke|all|correct|false} [--retry-seed N]
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
export PYTHONUNBUFFERED=1
correct_config=configs/generation/nqopen1520-together-gpt-oss-120b-correct.yaml
false_config=configs/generation/nqopen1520-together-gpt-oss-120b-false.yaml
mode="${1:-dry-run}"
if (( $# > 0 )); then shift; fi
case "$mode" in
  dry-run|smoke|all|correct|false) ;;
  *) echo "Usage: bash $0 {dry-run|smoke|all|correct|false} [--retry-seed N]" >&2; exit 2 ;;
esac
retry_seed=0
if (( $# > 0 )); then
  if [[ $# != 2 || "$1" != --retry-seed || ! "$2" =~ ^[0-9]+$ ]]; then
    echo 'Expected --retry-seed followed by a non-negative integer.' >&2
    exit 2
  fi
  retry_seed="$2"
fi
if [[ "$mode" != dry-run && -z "${TOGETHER_API_KEY:-}" ]]; then
  echo 'TOGETHER_API_KEY is not set; export it in this terminal before generation.' >&2
  exit 1
fi
if [[ "$mode" == all ]]; then
  # Separate output directories and SQLite caches let both stages run together.
  correct_pid=""
  false_pid=""
  cleanup() {
    for child_pid in "$correct_pid" "$false_pid"; do
      if [[ -n "$child_pid" ]]; then
        kill -TERM "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
      fi
    done
  }
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  "$PYTHON" -m evlab.cli generate-correct-documents --config "$correct_config" --retry-seed "$retry_seed" &
  correct_pid=$!
  "$PYTHON" -m evlab.cli generate-false-documents --config "$false_config" --retry-seed "$retry_seed" &
  false_pid=$!
  echo "Together generation: correct PID=$correct_pid, false PID=$false_pid (parallel; retry seed=$retry_seed)"
  correct_status=0
  false_status=0
  wait "$correct_pid" || correct_status=$?
  correct_pid=""
  wait "$false_pid" || false_status=$?
  false_pid=""
  echo "Together generation finished: correct exit=$correct_status, false exit=$false_status"
  if (( correct_status != 0 )); then exit "$correct_status"; fi
  exit "$false_status"
fi
options=(--retry-seed "$retry_seed")
if [[ "$mode" == dry-run ]]; then options+=(--dry-run); fi
if [[ "$mode" == smoke ]]; then options+=(--limit 5); fi
if [[ "$mode" != false ]]; then
  # Bash 3 (macOS) treats an empty array as unset under nounset.
  "$PYTHON" -m evlab.cli generate-correct-documents --config "$correct_config" ${options[@]+"${options[@]}"}
fi
if [[ "$mode" != correct ]]; then
  "$PYTHON" -m evlab.cli generate-false-documents --config "$false_config" ${options[@]+"${options[@]}"}
fi
