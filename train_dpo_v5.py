"""
train_dpo_v4.py
===============
DPO training with real model errors (v4 data).

Key differences from v1-v3:
  - Uses dpo_pairs_v4.jsonl (86% real model errors, 14% synthetic)
  - Lower beta (0.15) â€” real pairs need more aggressive signal
  - Slightly higher LR (1e-5) â€” more informative data supports it
  - 3 epochs with eval every 10 steps
  - Save best model by eval_loss

Chay: python train_dpo_v4.py
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from v5_run import create_manifest

# â”€â”€ Auto tee â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_log_path = f"train_{time.strftime('%Y%m%d_%H%M%S')}_v4.log"
_log_fh = open(_log_path, "w", encoding="utf-8")
_old_out = sys.stdout.write
_old_err = sys.stderr.write
def _tee(msg, old):
    _log_fh.write(msg); _log_fh.flush(); return old(msg)
sys.stdout.write = lambda m: _tee(m, _old_out)
sys.stderr.write = lambda m: _tee(m, _old_err)
print(f"Log: {os.path.abspath(_log_path)}", flush=True)

parser = argparse.ArgumentParser(description="Train a reproducible DPO v5 candidate.")
parser.add_argument("--run-id", required=True)
parser.add_argument("--pairs", required=True)
parser.add_argument("--sft-adapter", default="./final_adapter")
parser.add_argument("--runs-dir", default="runs")
parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
parser.add_argument("--beta", type=float, default=0.10)
parser.add_argument("--learning-rate", type=float, default=5e-6)
parser.add_argument("--epochs", type=float, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--gradient-accumulation", type=int, default=16)
ARGS = parser.parse_args()
MODEL_ID = ARGS.model
SFT_ADAPTER = ARGS.sft_adapter
PAIRS_PATH = ARGS.pairs
RUN_DIR = os.path.join(ARGS.runs_dir, ARGS.run_id)
OUTPUT_DIR = os.path.join(RUN_DIR, "checkpoints")
FINAL_ADAPTER = os.path.join(RUN_DIR, "adapter")
MAX_LEN = 1024
BETA = ARGS.beta
LEARNING_RATE = ARGS.learning_rate


def format_prompt(subtask: str) -> str:
    return (
        f"<|im_start|>user\n{subtask}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_pairs(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            rows.append({
                "prompt": format_prompt(p["prompt"]),
                "chosen": p["chosen"],
                "rejected": p["rejected"],
            })
    if not rows:
        sys.exit(f"No pairs loaded from {path}")
    return Dataset.from_list(rows)


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    create_manifest(
        RUN_DIR,
        kind="v5_dpo_train",
        config=vars(ARGS),
        inputs={"pairs": PAIRS_PATH, "sft_adapter_config": os.path.join(SFT_ADAPTER, "adapter_config.json")},
    )
    print("=" * 60)
    print("  DPO v4 â€” Real Error Preference Optimization")
    print("=" * 60)

    # â”€â”€ Load dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[1] Loading pairs from {PAIRS_PATH} ...")
    full_ds = load_pairs(PAIRS_PATH)
    split = full_ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"    Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # â”€â”€ Tokenizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[2] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # â”€â”€ Base model (bf16) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[3] Loading base model (bf16) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda:0", torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    )
    base_model.enable_input_require_grads()

    # â”€â”€ Policy model (SFT adapter, trainable) â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[4] Loading SFT adapter (trainable) ...")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER, is_trainable=True)
    model.print_trainable_parameters()

    # â”€â”€ Reference model (frozen) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[5] Loading reference model (frozen) ...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda:0", torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_base, SFT_ADAPTER, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model = ref_model.to("cuda:0")
    torch.cuda.empty_cache()

    # â”€â”€ DPO training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print("\n[6] Starting DPO v4 training ...")

    # Strategy: lower beta for harder real pairs, longer training
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,                    # Lower than v3 (0.2) â€” real pairs need stronger signal
        max_length=MAX_LEN,
        per_device_train_batch_size=ARGS.batch_size,
        gradient_accumulation_steps=ARGS.gradient_accumulation,  # Effective batch = 16
        dataloader_num_workers=2,
        optim="adamw_torch",
        learning_rate=LEARNING_RATE,           # Higher than v3 (5e-6) â€” real data supports it
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
        max_grad_norm=1.0,
        num_train_epochs=ARGS.epochs,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=10,                # More frequent eval
        save_strategy="steps",
        save_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=5,
        report_to="tensorboard",
        logging_dir=f"{OUTPUT_DIR}/logs",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # â”€â”€ TensorBoard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tb_thread = threading.Thread(
        target=lambda: subprocess.run(
            ["tensorboard", "--logdir", OUTPUT_DIR,
             "--host", "0.0.0.0", "--port", "8080"],
            capture_output=True,
        ),
        daemon=True,
    )
    tb_thread.start()
    print("    TensorBoard â†’ http://localhost:8080")

    trainer.train()

    # â”€â”€ Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"\n[7] Saving adapter to {FINAL_ADAPTER} ...")
    trainer.save_model(FINAL_ADAPTER)
    tokenizer.save_pretrained(FINAL_ADAPTER)

    print("\n" + "=" * 60)
    print("  DPO v4 training complete.")
    print(f"  Best checkpoint : {trainer.state.best_model_checkpoint}")
    print(f"  Best eval loss  : {trainer.state.best_metric:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
