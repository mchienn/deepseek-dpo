#!/bin/bash
# Run the SFT adapter only on dpo_mine; never point this script at final_test.
set -euo pipefail
WORKDIR="${WORKDIR:-/workspace/deepseek-dpo}"
RUN_ID="${1:?usage: $0 <run-id> [adapter-dir]}"
ADAPTER="${2:-$WORKDIR/final_adapter}"
BATCH_SIZE="${BATCH_SIZE:-256}"
cd "$WORKDIR"
source /venv/main/bin/activate
force_args=()
if [ "${FORCE:-0}" = "1" ]; then
  force_args+=(--force)
fi
python3 eval_adapter_v5.py --adapter "$ADAPTER" --test-file data/v5/dpo_mine.xlsx --run-id "$RUN_ID" --batch-size "$BATCH_SIZE" "${force_args[@]}"