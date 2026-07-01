#!/bin/bash
set -e

echo "=== 1. Clone repo ==="
cd /workspace
git clone https://github.com/mchienn/deepseek-dpo.git 2>/dev/null || echo "Already cloned"

echo "=== 2. Download adapter ==="
mkdir -p /workspace/deepseek-dpo/final_adapter
BASE="https://hf-mirror.com/nmc27705/deepseek-r1-distill-qwen-1.5b-lora-adapter/resolve/main"
cd /workspace/deepseek-dpo/final_adapter
wget -q "$BASE/adapter_config.json" -O adapter_config.json
wget "$BASE/adapter_model.safetensors" -O adapter_model.safetensors
ls -lh adapter_model.safetensors

echo "=== 3. Install deps ==="
cd /workspace/deepseek-dpo
pip install sentence-transformers==3.4.1 torch transformers accelerate peft trl datasets openpyxl pandas tensorboard webcolors -q

echo "=== 4. Verify CUDA ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('VRAM:', round(torch.cuda.get_device_properties(0).total_mem/1e9,1), 'GB')"

echo "=== 5. Start TensorBoard on 8081 ==="
pkill -f tensorboard 2>/dev/null || true
nohup tensorboard --logdir /workspace/deepseek-dpo/dpo_checkpoints --host 0.0.0.0 --port 8081 > /workspace/deepseek-dpo/tb.log 2>&1 &
sleep 1
echo "TensorBoard PID: $(pgrep -f 'tensorboard.*8081')"
echo "TensorBoard at http://localhost:8081"

echo "=== 6. Training log wrapper ==="
cat > /workspace/deepseek-dpo/train_and_log.sh << 'WRAPPER'
#!/bin/bash
LOGFILE="/workspace/deepseek-dpo/train_$(date +%Y%m%d_%H%M%S).log"
echo "Training log: $LOGFILE"
echo "TensorBoard:  http://localhost:8081"
echo "Run: tail -f $LOGFILE to watch progress"
cd /workspace/deepseek-dpo
python3 train_dpo.py 2>&1 | tee "$LOGFILE"
WRAPPER
chmod +x /workspace/deepseek-dpo/train_and_log.sh

echo ""
echo "============================================"
echo "  SETUP COMPLETE"
echo "============================================"
echo "  Connect:   ssh -p 59241 root@220.135.0.171 -L 8081:localhost:8081"
echo "  Train:     cd /workspace/deepseek-dpo && ./train_and_log.sh"
echo "  Monitor:   tail -f /workspace/deepseek-dpo/train_*.log"
echo "  Web:       http://localhost:8081 (TensorBoard)"
echo "============================================"
