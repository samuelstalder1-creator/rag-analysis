#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
MODEL="Qwen/Qwen3-4B-Instruct-2507"
MODEL_DIR="models/qwen3-4b-instruct-2507"
export PYTHONUNBUFFERED=1
export LOCAL_LLM_API_KEY="${LOCAL_LLM_API_KEY:-local}"
export VLLM_USE_FLASHINFER_SAMPLER=0

case "${1:-smoke}" in
  prepare)
    "$PYTHON" - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer
path = Path("models/qwen3-4b-instruct-2507")
snapshot_download(
    "Qwen/Qwen3-4B-Instruct-2507", local_dir=str(path),
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "merges.txt", "vocab.json"],
)
raw = Tokenizer.from_file(str(path / "tokenizer.json"))
loaded = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
for text in ["Don Henley sang in 1994.\nNext line.", "Café — 北京"]:
    ids = raw.encode(text, add_special_tokens=False).ids
    assert loaded.encode(text, add_special_tokens=False) == ids, "Tokenizer changed input IDs"
    assert loaded.decode(ids, skip_special_tokens=True) == text, "Tokenizer corrupted output"
print("Qwen model downloaded; tokenizer verification passed")
PY
    ;;
  serve)
    nvidia-smi >/dev/null
    exec "$PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_DIR" \
      --served-model-name "$MODEL" --host 127.0.0.1 --port 8000 \
      --dtype bfloat16 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
      --max-model-len 8192 --max-num-seqs 1
    ;;
  smoke|all)
    "$PYTHON" - <<'PY'
import time
from evlab.llm.local import LocalOpenAIClient
client = LocalOpenAIClient(base_url="http://localhost:8000/v1", model="Qwen/Qwen3-4B-Instruct-2507", max_tokens=2048, timeout=300)
start = time.monotonic()
text = client.generate("Return exactly this sentence, without commentary: Don Henley sang in 1994.", temperature=0.3)
if "Don Henley" not in text or "<think>" in text:
    raise SystemExit(f"Qwen API smoke test failed: {text!r}")
print(f"Qwen API smoke test passed in {time.monotonic() - start:.2f}s: {text}")
PY
    limit=()
    if [[ "${1:-smoke}" == smoke ]]; then limit=(--limit 5); fi
    "$PYTHON" -m evlab.cli generate-correct-documents \
      --config configs/generation/nqopen1520-qwen3-4b-correct.yaml "${limit[@]}"
    "$PYTHON" -m evlab.cli generate-false-documents \
      --config configs/generation/nqopen1520-qwen3-4b-false.yaml "${limit[@]}"
    ;;
  *) echo "Usage: bash $0 {prepare|serve|smoke|all}" >&2; exit 2 ;;
esac
