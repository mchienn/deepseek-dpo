#!/bin/bash
# Prepare a Vast instance for v5. This script does not start training.
set -euo pipefail
WORKDIR="${WORKDIR:-/workspace/deepseek-dpo}"
cd "$WORKDIR"
git pull --ff-only
source /venv/main/bin/activate
uv pip install -q transformers==4.51.3 accelerate==1.5.2 peft==0.19.1 trl==0.16.0 datasets==3.5.0 sentence-transformers==3.4.2 pyarrow==19.0.1 openpyxl pandas tensorboard webcolors
python3 -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
python3 prepare_v5_splits.py --source test_cleaned.xlsx --output-dir data/v5
python3 validate_v5_data.py --split-dir data/v5
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
printf 'V5 preflight complete. Run baseline on data/v5/dpo_mine.xlsx before building pairs.\n'

