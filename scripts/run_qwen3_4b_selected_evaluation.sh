#!/usr/bin/env bash
# Kept for the systemd units; the logic lives in scripts/run_evaluation.sh.
exec bash "$(dirname "$0")/run_evaluation.sh" qwen3_4b_selected
