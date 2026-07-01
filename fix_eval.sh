#!/bin/bash
cd /workspace/deepseek-dpo
git checkout eval_deepseek.py
sed -i 's|^ADAPTER_DIR *=.*|ADAPTER_DIR    = "./dpo_adapter"|' eval_deepseek.py
grep ADAPTER_DIR eval_deepseek.py
