#!/bin/bash
LOGFILE="/workspace/deepseek-dpo/train_$(date +%Y%m%d_%H%M%S).log"
echo "Training log: $LOGFILE"
echo "Monitor:      tail -f $LOGFILE"
echo "TensorBoard:  http://localhost:8081"
cd /workspace/deepseek-dpo
/venv/main/bin/python3 train_dpo.py 2>&1 | tee "$LOGFILE"