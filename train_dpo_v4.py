"""
train_dpo_v4.py
===============
DPO training with real model errors (v4 data).

Key differences from v1-v3:
  - Uses dpo_pairs_v4.jsonl (86% real model errors, 14% synthetic)
  - Lower beta (0.15) — real pairs need more aggressive signal
  - Slightly higher LR (1e-5) — more informative data supports it
  - 3 epochs with eval every 10 steps
  - Save best model by eval_loss

Chay: python train_dpo_v4.py
"""

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

# ── Auto tee ──────────────────────────────────────────────
_log_path = f"train_{time.strftime('%Y%m%d_%H%M%S')}_v4.log"
_log_fh = open(_log_path, "w", encoding="utf-8")
_old_out = sys.stdout.write
_old_err = sys.stderr.write
def _tee(msg, old):
    _log_fh.write(msg); _log_fh.flush(); return old(msg)
sys.stdout.write = lambda m: _tee(m, _old_out)
sys.stderr.write = lambda m: _tee(m, _old_err)
print(f"Log: {os.path.abspath(_log_path)}", flush=True)

MODEL_ID       = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SFT_ADAPTER    = "./final_adapter"
PAIRS_PATH     = "dpo_pairs_v4.jsonl"
OUTPUT_DIR     = "./dpo_checkpoints_v4"
FINAL_ADAPTER  = "./dpo_adapter_v4"

MAX_LEN = 1024


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
    print("=" * 60)
    print("  DPO v4 — Real Error Preference Optimization")
    print("=" * 60)

    # ── Load dataset ─────────────────────────────────
    print(f"\n[1] Loading pairs from {PAIRS_PATH} ...")
    full_ds = load_pairs(PAIRS_PATH)
    split = full_ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"    Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # ── Tokenizer ────────────────────────────────────
    print("\n[2] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Base model (bf16) ────────────────────────────
    print("\n[3] Loading base model (bf16) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="cuda:0", torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    )
    base_model.enable_input_require_grads()

    # ── Policy model (SFT adapter, trainable) ────────
    print("\n[4] Loading SFT adapter (trainable) ...")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER, is_trainable=True)
    model.print_trainable_parameters()

    # ── Reference model (frozen) ─────────────────────
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

    # ── DPO training ─────────────────────────────────
    print("\n[6] Starting DPO v4 training ...")

    # Strategy: lower beta for harder real pairs, longer training
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.15,                    # Lower than v3 (0.2) — real pairs need stronger signal
        max_length=MAX_LEN,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch = 16
        dataloader_num_workers=2,
        optim="adamw_torch",
        learning_rate=1e-5,           # Higher than v3 (5e-6) — real data supports it
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
        max_grad_norm=1.0,
        num_train_epochs=3,
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

    # ── TensorBoard ──────────────────────────────────
    tb_thread = threading.Thread(
        target=lambda: subprocess.run(
            ["tensorboard", "--logdir", OUTPUT_DIR,
             "--host", "0.0.0.0", "--port", "8080"],
            capture_output=True,
        ),
        daemon=True,
    )
    tb_thread.start()
    print("    TensorBoard → http://localhost:8080")

    trainer.train()

    # ── Save ─────────────────────────────────────────
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
