#!/usr/bin/env bash
# Source-grounded GPT-OSS correct/false generation through Together; no GPU server.
# Usage: bash scripts/run_together_generation.sh {dry-run|smoke|all|correct|false}
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
export PYTHONUNBUFFERED=1
correct_config=configs/generation/nqopen1520-together-gpt-oss-120b-correct.yaml
false_config=configs/generation/nqopen1520-together-gpt-oss-120b-false.yaml
mode="${1:-dry-run}"
case "$mode" in
  dry-run|smoke|all|correct|false) ;;
  *) echo "Usage: bash $0 {dry-run|smoke|all|correct|false}" >&2; exit 2 ;;
esac
if [[ "$mode" != dry-run && -z "${TOGETHER_API_KEY:-}" ]]; then
  echo 'TOGETHER_API_KEY is not set; export it in this terminal before generation.' >&2
  exit 1
fi
options=()
if [[ "$mode" == dry-run ]]; then options=(--dry-run); fi
if [[ "$mode" == smoke ]]; then options=(--limit 5); fi
if [[ "$mode" != false ]]; then
  "$PYTHON" -m evlab.cli generate-correct-documents --config "$correct_config" "${options[@]}"
fi
if [[ "$mode" != correct ]]; then
  "$PYTHON" -m evlab.cli generate-false-documents --config "$false_config" "${options[@]}"
fi
