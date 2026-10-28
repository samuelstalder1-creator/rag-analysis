#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
TOKENIZER="models/deepseek-r1-bytelevel-tokenizer"
CORRECT="configs/archive/generation/nqopen1520-deepseek-r1-correct.yaml"
FALSE="configs/archive/generation/nqopen1520-deepseek-r1-false.yaml"
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-local}"
export PYTHONUNBUFFERED=1
# Avoid the CUDA compiler/FlashInfer sampler incompatibility on this server.
export VLLM_USE_FLASHINFER_SAMPLER=0

case "${1:-smoke}" in
  prepare)
    exec "$PYTHON" scripts/prepare_deepseek_tokenizer.py "${@:2}"
    ;;
  serve)
    nvidia-smi >/dev/null
    "$PYTHON" scripts/prepare_deepseek_tokenizer.py
    exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --host 127.0.0.1 --port 8000 --dtype bfloat16 \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
      --max-model-len "${MAX_MODEL_LEN:-12288}" \
      --max-num-seqs 1
    ;;
  smoke|all)
    # A direct API call bypasses the cache and catches wrong models, decoding
    # failures, and truncation before processing any dataset documents.
    "$PYTHON" - <<'PY'
from evlab.llm.local import LocalOpenAIClient
client = LocalOpenAIClient(
    base_url="http://localhost:8000/v1",
    model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    max_tokens=8192, timeout=1800,
)
text = client.generate('Return exactly this sentence, without commentary: Don Henley sang in 1994.', temperature=0.3)
if "Don Henley" not in text:
    raise SystemExit(f"DeepSeek smoke test failed: {text!r}")
print("DeepSeek API smoke test passed:", text)
PY
    limit=()
    if [[ "${1:-smoke}" == smoke ]]; then limit=(--limit 5); fi
    "$PYTHON" -m evlab.cli generate-correct-documents --config "$CORRECT" "${limit[@]}"
    "$PYTHON" -m evlab.cli generate-false-documents --config "$FALSE" "${limit[@]}"
    ;;
  *) echo "Usage: bash $0 {prepare|serve|smoke|all}" >&2; exit 2 ;;
esac
