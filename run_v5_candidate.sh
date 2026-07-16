#!/bin/bash
# Train one V5 DPO candidate from a completed, mine-only SFT baseline.
set -euo pipefail
WORKDIR="${WORKDIR:-/workspace/deepseek-dpo}"
RUN_ID="${1:?usage: $0 <run-id> [beta] [learning-rate] [seed]}"
BETA="${2:-0.10}"
LEARNING_RATE="${3:-5e-6}"
SEED="${4:-42}"
DPO_BATCH_SIZE="${DPO_BATCH_SIZE:-4}"
DPO_GRADIENT_ACCUMULATION="${DPO_GRADIENT_ACCUMULATION:-4}"
DPO_EPOCHS="${DPO_EPOCHS:-1}"
SFT_ADAPTER="${SFT_ADAPTER:-$WORKDIR/final_adapter}"
BASELINE_RUN_ID="${BASELINE_RUN_ID:-v5-sft-mine}"
cd "$WORKDIR"
source /venv/main/bin/activate
python3 validate_v5_data.py --split-dir data/v5
BASELINE_RESULT="runs/${BASELINE_RUN_ID}/evaluation.xlsx"
if [ ! -f "$BASELINE_RESULT" ]; then
  echo "Missing mine-only baseline: $BASELINE_RESULT" >&2
  exit 2
fi
python3 build_dpo_pairs_v5.py --evaluation "$BASELINE_RESULT" --mine-file data/v5/dpo_mine.xlsx --run-id "$RUN_ID" --seed "$SEED"
python3 validate_v5_data.py --split-dir data/v5 --pairs "runs/${RUN_ID}/pairs.jsonl"
python3 train_dpo_v5.py --run-id "$RUN_ID" --pairs "runs/${RUN_ID}/pairs.jsonl" --sft-adapter "$SFT_ADAPTER" --beta "$BETA" --learning-rate "$LEARNING_RATE" --seed "$SEED" --batch-size "$DPO_BATCH_SIZE" --gradient-accumulation "$DPO_GRADIENT_ACCUMULATION" --epochs "$DPO_EPOCHS"
python3 eval_adapter_v5.py --adapter "runs/${RUN_ID}/adapter" --test-file data/v5/dpo_validation.xlsx --run-id "${RUN_ID}-validation" --batch-size 16 --force
printf 'Candidate trained and evaluated on dpo_validation. Do not run final_test until candidate selection is complete.\n'
