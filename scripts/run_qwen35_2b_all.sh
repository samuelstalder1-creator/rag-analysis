#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/generation/qwen35-2b-all.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
GEN_LOG="${GEN_LOG:-$LOG_DIR/qwen35_2b_generation_$STAMP.log}"
GEN_PID_FILE="${GEN_PID_FILE:-$LOG_DIR/qwen35_2b_generation.pid}"
VLLM_LOG="${VLLM_LOG:-$LOG_DIR/qwen35_2b_vllm_$STAMP.log}"
VLLM_PID_FILE="${VLLM_PID_FILE:-$LOG_DIR/qwen35_2b_vllm.pid}"

export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-local}"

if [[ "${START_VLLM:-0}" == "1" ]]; then
  nohup uv run vllm serve Qwen/Qwen3.5-2B \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    > "$VLLM_LOG" 2>&1 &
  echo "$!" > "$VLLM_PID_FILE"
  echo "Started vLLM: pid=$(cat "$VLLM_PID_FILE") log=$VLLM_LOG"
  echo "Waiting ${VLLM_WAIT_SECONDS:-60}s for the OpenAI-compatible server..."
  sleep "${VLLM_WAIT_SECONDS:-60}"
fi

nohup uv run evlab generate --config "$CONFIG" > "$GEN_LOG" 2>&1 &
echo "$!" > "$GEN_PID_FILE"

echo "Started generation: pid=$(cat "$GEN_PID_FILE") log=$GEN_LOG"
echo "Follow progress with: tail -f $GEN_LOG"
