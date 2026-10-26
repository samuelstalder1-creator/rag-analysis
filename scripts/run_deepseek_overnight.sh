#!/usr/bin/env bash
# Compatibility entry point for the original DeepSeek schedule.
set -euo pipefail
exec bash "$(dirname "$0")/run_overnight_generation.sh"
