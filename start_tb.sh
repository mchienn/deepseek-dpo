nohup tensorboard --logdir /workspace/deepseek-dpo/dpo_checkpoints --host 0.0.0.0 --port 8081 > /workspace/deepseek-dpo/tb.log 2>&1 &
echo "TensorBoard started"
