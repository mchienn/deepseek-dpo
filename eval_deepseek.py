"""
eval_deepseek.py — Evaluate DeepSeek-R1-Distill-Qwen-1.5B LoRA adapter
──────────────────────────────────────────────────────────────────────────
Pipeline:
  1. Load base model + LoRA adapter (merged)
  2. Inference on test.xlsx  (Input → Reasoning → JSON output)
  3. Score JSON output vs ground truth using ScoreEval logic
  4. Save results to eval_deepseek_result.xlsx

Run on GPU (Vast.ai):
  pip install sentence-transformers webcolors openpyxl
  python eval_deepseek.py
──────────────────────────────────────────────────────────────────────────
"""

import math
import json
import re
import time
import torch
import numpy as np
import pandas as pd
import webcolors
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

BASE_MODEL     = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ADAPTER_DIR    = "./dpo_adapter"
TEST_FILE      = "test_cleaned.xlsx"
OUTPUT_FILE    = "eval_deepseek_result.xlsx"
THRESHOLD      = 0.7
WEIGHT_VALUE   = 1
MAX_NEW_TOKENS = 512
HF_BATCH       = 256      # RTX 4090: 28% at batch=8, ~90% at batch=32

# Special tokens to strip from model output
SPECIAL_MARKERS = [
    "<|im_end|>", "<|im_start|>", "<|endoftext|>",
    "<|end_of_sentence|>",
]

# ─────────────────────────────────────────────────────────────────
# SCORING FUNCTIONS (preserved from ScoreEval.py)
# ─────────────────────────────────────────────────────────────────

_embed_model = None
def load_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def flatten_json(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            out.update(flatten_json(v, new_prefix))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            out.update(flatten_json(v, new_prefix))
    else:
        out[prefix] = obj
    return out

def greedy_max_matching(sim_matrix, keys1, keys2, threshold=0.7):
    sim_matrix = sim_matrix.copy()
    matched = []
    used_i, used_j = set(), set()
    while True:
        i, j = divmod(sim_matrix.argmax(), sim_matrix.shape[1])
        score = sim_matrix[i, j]
        if score < threshold:
            break
        if i in used_i or j in used_j:
            sim_matrix[i, j] = -1
            continue
        matched.append((keys1[i], keys2[j], score))
        used_i.add(i)
        used_j.add(j)
        sim_matrix[i, :] = -1
        sim_matrix[:, j] = -1
    for i in range(len(keys1)):
        if i not in used_i:
            matched.append((keys1[i], None, 0))
    for j in range(len(keys2)):
        if j not in used_j:
            matched.append((None, keys2[j], 0))
    return matched

def parse_number(value: str):
    if isinstance(value, (int, float)):
        return float(value), ""
    if not isinstance(value, str):
        return None, None
    v = value.strip().lower()
    m = re.match(r"([-+]?\d*\.?\d+)([a-z%]*)", v)
    if m:
        return float(m.group(1)), m.group(2) or ""
    return None, None

def number_similarity(v1, v2, epsilon=1e-9):
    n1, u1 = parse_number(v1)
    n2, u2 = parse_number(v2)
    if n1 is None or n2 is None:
        return 0.0
    if u1 != u2 and u1 != "" and u2 != "":
        return 0.0
    diff = abs(n1 - n2)
    max_val = max(abs(n1), abs(n2), epsilon)
    return max(0.0, 1.0 - (diff / max_val))

def to_rgb(value: str):
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if value.startswith("rgb"):
        nums = re.findall(r"\d+", value)
        if len(nums) == 3:
            return tuple(int(n) for n in nums)
    if value.startswith("#"):
        try:
            return webcolors.hex_to_rgb(value)
        except ValueError:
            return None
    try:
        return webcolors.name_to_rgb(value)
    except ValueError:
        return None

MAX_RGB_DISTANCE = math.sqrt(3 * (255**2))

def color_similarity(c1, c2):
    rgb1 = to_rgb(c1)
    rgb2 = to_rgb(c2)
    if rgb1 is None or rgb2 is None:
        return 0.0
    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))
    if dist >= MAX_RGB_DISTANCE:
        return 0.0
    return max(0.0, min(1.0, 1.0 - dist / MAX_RGB_DISTANCE))

def bool_similarity(v1, v2):
    true_vals = {"true", "yes", "1", "enabled", "on"}
    false_vals = {"false", "no", "0", "disabled", "off"}
    v1, v2 = str(v1).lower(), str(v2).lower()
    if v1 in true_vals and v2 in true_vals: return 1.0
    if v1 in false_vals and v2 in false_vals: return 1.0
    return 0.0

def jaccard_similarity(s1, s2):
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

def text_similarity(v1, v2):
    v1, v2 = str(v1).strip(), str(v2).strip()
    if not v1 and not v2: return 1.0
    if not v1 or not v2: return 0.0
    if v1 == v2: return 1.0
    s_sem = 0.0
    try:
        embeddings = load_embed_model().encode([v1, v2], convert_to_tensor=True)
        s_sem = util.cos_sim(embeddings[0], embeddings[1]).item()
        s_sem = max(0.0, min(1.0, s_sem))
    except Exception:
        pass
    s_str = jaccard_similarity(v1, v2)
    return s_sem * 0.5 + s_str * 0.5

def semantic_similarity(v1, v2):
    v1, v2 = str(v1).strip(), str(v2).strip()
    v1_low, v2_low = v1.lower(), v2.lower()
    if to_rgb(v1) and to_rgb(v2):
        return color_similarity(v1, v2)
    n1, _ = parse_number(v1)
    n2, _ = parse_number(v2)
    if n1 is not None and n2 is not None:
        return number_similarity(v1, v2)
    bool_set = {"true","false","yes","no","enabled","disabled","on","off","0","1"}
    if v1_low in bool_set and v2_low in bool_set:
        return bool_similarity(v1, v2)
    return text_similarity(v1, v2)

def compare_values_with_keys(flat1, flat2, pairs, weightvalue=1):
    key_scores, value_scores, details = [], [], []
    for k1, k2, score_key in pairs:
        if k1 and k2:
            v1, v2 = flat1.get(k1), flat2.get(k2)
            score_val = max(0.0, min(1.0, semantic_similarity(v1, v2)))
            value_scores.append(score_val)
            key_scores.append(score_key)
            details.append((f"{k1}:{v1}", f"{k2}:{v2}", score_key, score_val))
        elif k1 and not k2:
            key_scores.append(0.0)
            value_scores.append(0.0)
            details.append((f"{k1}:{flat1.get(k1)}", "None", 0.0, 0.0))
        elif k2 and not k1:
            details.append(("None", f"{k2}:{flat2.get(k2)}", None, None))
    details_sorted = sorted(details, key=lambda x: (x[2] if x[2] is not None else -1), reverse=True)
    ks = max(0.0, min(1.0, sum(key_scores) / len(key_scores))) if key_scores else 0.0
    vs = max(0.0, min(1.0, sum(value_scores) / len(value_scores))) if value_scores else 0.0
    avg = (ks + vs * weightvalue) / (1 + weightvalue)
    return details_sorted, ks, vs, avg

def safe_json_parse(text: str):
    try:
        return json.loads(text), True
    except json.JSONDecodeError:
        fixed = text
        while fixed.count("{") > fixed.count("}"):
            fixed += "}"
        while fixed.count("{") < fixed.count("}"):
            fixed = "{" + fixed
        try:
            return json.loads(fixed), True
        except:
            return text, False

def parse_border_shorthand(border_str):
    if not isinstance(border_str, str):
        return {}
    parts = border_str.strip().split()
    expanded = {}
    border_styles = {"none","hidden","dotted","dashed","solid","double","groove","ridge","inset","outset"}
    for part in parts:
        n, u = parse_number(part)
        if n is not None and 'border-width' not in expanded:
            expanded['border-width'] = part
            continue
        if part.lower() in border_styles and 'border-style' not in expanded:
            expanded['border-style'] = part
            continue
        if to_rgb(part) and 'border-color' not in expanded:
            expanded['border-color'] = part
            continue
    return expanded

def expand_shorthand_css(obj):
    if not isinstance(obj, dict):
        return obj
    if 'border' in obj and isinstance(obj['border'], str):
        for key, val in parse_border_shorthand(obj['border']).items():
            if key not in obj:
                obj[key] = val
    for key, value in obj.items():
        if isinstance(value, dict):
            obj[key] = expand_shorthand_css(value)
        elif isinstance(value, list):
            obj[key] = [expand_shorthand_css(item) for item in value if isinstance(item, (dict, list))]
    return obj


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────

def strip_special(text: str) -> str:
    """Remove all known special tokens from model output."""
    for m in SPECIAL_MARKERS:
        text = text.replace(m, "")
    # Also strip the unicode variant
    text = text.replace("\uff5c", "|")       # fullwidth |
    text = text.replace("\u2581", "_")       # lower one eighth block
    text = re.sub(r"<\|end_of_sentence\|>", "", text)
    text = re.sub(r"<\|.*?\|>", "", text)    # catch-all for any remaining special tokens
    return text.strip()


def extract_json(full_output: str):
    """Extract JSON part from model output (after </think>)."""
    full_output = strip_special(full_output)
    if "</think>" in full_output:
        json_part = full_output.split("</think>", 1)[1].strip()
    else:
        json_part = full_output.strip()
    return json_part, full_output


# ─────────────────────────────────────────────────────────────────
# INFERENCE — Batched HF with checkpoint/resume
# ─────────────────────────────────────────────────────────────────

def run_hf_inference(subtasks: list) -> list:
    """
    Batched HuggingFace inference (batch=8) with JSONL checkpoint/resume.
    ~4x faster than single-sample. Safe to interrupt and resume.
    """
    CKPT = Path("eval_checkpoint.jsonl")

    # ── Resume from checkpoint ────────────────────────────────
    done = {}
    if CKPT.exists():
        with open(CKPT, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                done[obj["idx"]] = (obj["json_part"], obj["full_output"])
        print(f"  Resumed {len(done):,} from checkpoint")

    remaining = [(i, s) for i, s in enumerate(subtasks) if i not in done]
    if not remaining:
        print("  All inferred, skipping model load ...")
        return [done[i] for i in range(len(subtasks))]

    print(f"  {len(remaining):,} remaining (batch={HF_BATCH}) ...")

    # ── Load model ────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for batched generation

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Get <|im_end|> token ID so model stops early
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id]
    if isinstance(im_end_id, int) and im_end_id != tokenizer.eos_token_id:
        eos_ids.append(im_end_id)
    print(f"  Model loaded on {model.device}, eos_ids={eos_ids}")

    # ── Batched inference ─────────────────────────────────────
    ckpt_f = open(CKPT, "a", encoding="utf-8")
    num_batches = (len(remaining) + HF_BATCH - 1) // HF_BATCH

    for b in tqdm(range(num_batches), desc=f"Inference (batch={HF_BATCH})"):
        batch = remaining[b * HF_BATCH : (b + 1) * HF_BATCH]
        batch_idxs = [x[0] for x in batch]
        batch_subtasks = [x[1] for x in batch]

        prompts = [
            f"<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n"
            for s in batch_subtasks
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=768,
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,  # stop at <|im_end|>, not always 512 tokens
            )

        # Decode each sample in batch
        for i, idx in enumerate(batch_idxs):
            gen_ids = out_ids[i][input_len:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
            json_part, full_output = extract_json(raw)

            done[idx] = (json_part, full_output)
            ckpt_f.write(
                json.dumps({"idx": idx, "json_part": json_part,
                            "full_output": full_output}, ensure_ascii=False) + "\n"
            )

        ckpt_f.flush()  # flush every batch

    ckpt_f.close()
    del model
    torch.cuda.empty_cache()
    return [done[i] for i in range(len(subtasks))]


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DeepSeek-R1-Distill Evaluation")
    print("=" * 60)

    # ── Load embed model for scoring ─────────────────────────────
    print("\n[1] Loading embedding model ...")
    load_embed_model()
    print("    all-MiniLM-L6-v2 loaded.")

    # ── Load test data ───────────────────────────────────────────
    print(f"\n[2] Loading {TEST_FILE} ...")
    df = pd.read_excel(TEST_FILE)
    print(f"    Rows: {len(df):,}")
    print(f"    Columns: {list(df.columns)}")

    # ── Inference ────────────────────────────────────────────
    print(f"\n[3] Running inference ...")
    all_subtasks  = df["Sub Task"].tolist()
    all_step_strs = df["Step Object"].tolist()

    inference_results = run_hf_inference(all_subtasks)

    all_json_outputs = [r[0] for r in inference_results]
    all_full_outputs = [r[1] for r in inference_results]

    # ── Scoring ───────────────────────────────────────────────
    print(f"\n[4] Scoring {len(all_json_outputs):,} outputs ...")
    results = []
    pass_count = 0
    fail_count = 0
    invalid_json_count = 0

    for i in tqdm(range(len(all_subtasks)), desc="Scoring"):
        subtask       = all_subtasks[i]
        step_json_str = str(all_step_strs[i])
        json_output   = all_json_outputs[i]
        full_output   = all_full_outputs[i]

        # Parse ground truth
        try:
            step_json = json.loads(step_json_str)
        except:
            step_json = {}

        # Parse model output
        parsed, is_valid = safe_json_parse(json_output)
        if json_output in ({}, "{}"):
            is_valid = False

        if not is_valid:
            invalid_json_count += 1
            fail_count += 1
            results.append({
                "Sub Task": subtask,
                "Step Object": step_json_str,
                "Model": json_output,
                "Full Output": full_output,
                "Valid_JSON": False,
                "Result": "Fail",
                "AVG Score": 0.0,
                "Details": "",
                "Key Scores": 0.0,
                "Value Scores": 0.0,
            })
            continue

        # Score
        parsed = expand_shorthand_css(parsed)
        flat1 = flatten_json(step_json)
        flat2 = flatten_json(parsed)
        keys1, keys2 = list(flat1.keys()), list(flat2.keys())

        if not keys1 or not keys2:
            avg, key_score, value_score = 0.0, 0.0, 0.0
            detail_str = "empty keys"
        else:
            emb1 = load_embed_model().encode(keys1, convert_to_tensor=True)
            emb2 = load_embed_model().encode(keys2, convert_to_tensor=True)
            sim_matrix = util.cos_sim(emb1, emb2).cpu().numpy()
            pairs = greedy_max_matching(sim_matrix, keys1, keys2, threshold=THRESHOLD)
            details, key_score, value_score, avg = compare_values_with_keys(
                flat1, flat2, pairs, weightvalue=WEIGHT_VALUE
            )
            avg = max(0.0, min(1.0, avg))
            detail_str = "\n".join([
                f"{l} <-> {r} (key_score={ks:.3f}) (value_score={vs:.3f})"
                if ks is not None else f"{l} <-> {r} (skip)"
                for l, r, ks, vs in details
            ])

        result = "Pass" if avg >= THRESHOLD else "Fail"
        if result == "Pass":
            pass_count += 1
        else:
            fail_count += 1

        results.append({
            "Sub Task": subtask,
            "Step Object": step_json_str,
            "Model": json_output,
            "Full Output": full_output,
            "Valid_JSON": True,
            "Result": result,
            "AVG Score": avg,
            "Details": detail_str,
            "Key Scores": key_score,
            "Value Scores": value_score,
        })

    # ── Save results ─────────────────────────────────────────
    out_df = pd.DataFrame(results)
    out_df.to_excel(OUTPUT_FILE, index=False)

    # ── Summary ─────────────────────────────────────────────
    total = len(results)
    avg_score = out_df["AVG Score"].mean()
    valid_count = total - invalid_json_count
    pass_rate = pass_count / total * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total samples      : {total:,}")
    print(f"  Valid JSON         : {valid_count:,} ({valid_count/total*100:.1f}%)")
    print(f"  Invalid JSON       : {invalid_json_count:,} ({invalid_json_count/total*100:.1f}%)")
    print(f"  Pass (>={THRESHOLD})  : {pass_count:,} ({pass_rate:.1f}%)")
    print(f"  Fail               : {fail_count:,} ({(fail_count/total*100):.1f}%)")
    print(f"  Avg Score          : {avg_score:.4f}")
    print(f"  Output             : {OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
