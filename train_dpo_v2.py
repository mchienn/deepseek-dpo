"""
train_dpo_v2.py
===============
Version 2 of DPO training script.
Uses corrected dpo_pairs_v2.jsonl and avoids truncation by setting MAX_LEN to 1024.

Input:
  - dpo_pairs_v2.jsonl
  - ./final_adapter/

Output:
  - ./dpo_adapter_v2/
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

# Tee stdout/stderr to file
_log_path = f"train_v2_{time.strftime('%Y%m%d_%H%M%S')}.log"
_log_fh = open(_log_path, "w", encoding="utf-8")
_old_stdout_write = sys.stdout.write
_old_stderr_write = sys.stderr.write
def _tee_write(msg, old_write):
    _log_fh.write(msg)
    _log_fh.flush()
    return old_write(msg)
sys.stdout.write = lambda msg: _tee_write(msg, _old_stdout_write)
sys.stderr.write = lambda msg: _tee_write(msg, _old_stderr_write)
print(f"Log file: {os.path.abspath(_log_path)}", flush=True)

MODEL_ID       = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
SFT_ADAPTER    = "./final_adapter"
PAIRS_PATH     = "dpo_pairs_v2.jsonl"
OUTPUT_DIR     = "./dpo_checkpoints_v2"
FINAL_ADAPTER  = "./dpo_adapter_v2"

MAX_LEN = 1024


def format_prompt(subtask: str) -> str:
    return (
        f"<|im_start|>user\n{subtask}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def validate_pairs(rows):
    required = {"prompt", "chosen", "rejected"}
    for i, r in enumerate(rows):
        missing = required - set(r.keys())
        if missing:
            sys.exit(f"Row {i}: missing keys {missing}")
        for k in ("chosen", "rejected"):
            if not isinstance(r[k], str) or len(r[k]) == 0:
                sys.exit(f"Row {i}: '{k}' is empty or not a string")


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
    validate_pairs(rows)
    return Dataset.from_list(rows)


def main():
    print("=" * 60)
    print("  DPO Fine-Tuning v2 — DeepSeek-R1-Distill-Qwen-1.5B")
    print("=" * 60)

    # Load dataset
    print(f"\n[1] Loading pairs from {PAIRS_PATH} ...")
    full_ds = load_pairs(PAIRS_PATH)
    split = full_ds.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"    Train pairs: {len(train_ds)} | Eval pairs: {len(eval_ds)}")

    # Tokenizer
    print("\n[2] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"    Vocab size: {tokenizer.vocab_size:,}")
    print(f"    padding_side: {tokenizer.padding_side}")

    # Model (bf16)
    print("\n[3] Loading base model (bf16) ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    base_model.enable_input_require_grads()

    # Load SFT adapter
    print("\n[4] Loading SFT adapter (trainable) ...")
    model = PeftModel.from_pretrained(base_model, SFT_ADAPTER, is_trainable=True)
    model.print_trainable_parameters()

    # Reference model
    print("\n[5] Loading reference model (bf16, frozen) ...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    ref_model = PeftModel.from_pretrained(ref_base, SFT_ADAPTER, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print("    Reference model frozen.")

    # DPO training config
    print("\n[6] Starting DPO training ...")
    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=0.1,
        max_length=MAX_LEN,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        dataloader_num_workers=2,
        optim="adamw_torch",
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.0,
        max_grad_norm=1.0,
        num_train_epochs=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
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

    # TensorBoard server
    tb_thread = threading.Thread(
        target=lambda: subprocess.run(
            ["tensorboard", "--logdir", OUTPUT_DIR, "--host", "0.0.0.0", "--port", "8081"],
            capture_output=True,
        ),
        daemon=True,
    )
    tb_thread.start()
    print("    TensorBoard → http://localhost:8081")

    trainer.train()

    # Save
    print(f"\n[7] Saving adapter to {FINAL_ADAPTER} ...")
    trainer.save_model(FINAL_ADAPTER)
    tokenizer.save_pretrained(FINAL_ADAPTER)

    print("\n" + "=" * 60)
    print("  DPO training complete.")
    print(f"  Best checkpoint : {trainer.state.best_model_checkpoint}")
    print(f"  Best eval loss  : {trainer.state.best_metric:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
