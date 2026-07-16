#!/bin/bash
# Execute one V5 candidate only after setup_vast_v5.sh and a mined SFT result exist.
set -euo pipefail
WORKDIR="${WORKDIR:-/workspace/deepseek-dpo}"
RUN_ID="${1:?usage: $0 <run-id>}"
SFT_ADAPTER="${SFT_ADAPTER:-$WORKDIR/final_adapter}"
cd "$WORKDIR"
python3 validate_v5_data.py --split-dir data/v5
python3 eval_adapter_v5.py --adapter "$SFT_ADAPTER" --test-file data/v5/dpo_mine.xlsx --run-id "${RUN_ID}-mine-sft" --force
python3 build_dpo_pairs_v5.py --evaluation "runs/${RUN_ID}-mine-sft/evaluation.xlsx" --mine-file data/v5/dpo_mine.xlsx --run-id "$RUN_ID"
python3 validate_v5_data.py --split-dir data/v5 --pairs "runs/${RUN_ID}/pairs.jsonl"
python3 train_dpo_v5.py --run-id "$RUN_ID" --pairs "runs/${RUN_ID}/pairs.jsonl" --sft-adapter "$SFT_ADAPTER" --beta 0.10 --learning-rate 5e-6 --seed 42
python3 eval_adapter_v5.py --adapter "runs/${RUN_ID}/adapter" --test-file data/v5/dpo_validation.xlsx --run-id "${RUN_ID}-validation" --force
printf 'Candidate trained and evaluated on dpo_validation. Do not run final_test until candidate selection is complete.\n'
