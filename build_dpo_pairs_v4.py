"""
build_dpo_pairs_v4.py
=====================
Build DPO preference pairs from REAL model errors (eval results),
with minimal synthetic corruption.

Key innovation vs v1:
  - Source 1 (primary): real model outputs scored < threshold as rejected
  - Source 2: structural validity pairs (valid json chosen, invalid rejected)
  - Source 3: synthetic corruption (minimal, for diversity only)

Format (same as v1):
  prompt   = <|im_start|>user\n{subtask}<|im_end|>\n<|im_start|>assistant\n
  chosen   = {json_str}  (plain JSON — consistent with train_dpo.py)
  rejected = {json_str}  (plain JSON)

Chay: python build_dpo_pairs_v4.py
"""

import json
import os
import random
import sys
import time
from collections import Counter
import pandas as pd

# ── Auto tee ──────────────────────────────────────────────
_log_path = f"build_v4_{time.strftime('%Y%m%d_%H%M%S')}.log"
_log_fh = open(_log_path, "w", encoding="utf-8")
_old_out = sys.stdout.write
_old_err = sys.stderr.write
def _tee(msg, old):
    _log_fh.write(msg); _log_fh.flush(); return old(msg)
sys.stdout.write = lambda m: _tee(m, _old_out)
sys.stderr.write = lambda m: _tee(m, _old_err)
print(f"Log: {os.path.abspath(_log_path)}", flush=True)

random.seed(42)

EVAL_RESULT_PATH = "eval_deepseek_result.xlsx"
EVAL_CHECKPOINT_PATH = "eval_checkpoint.jsonl"
TRAIN_PATH = "train_reasoning_fixed.xlsx"
OUTPUT_PATH = "dpo_pairs_v4.jsonl"

TARGET_PAIRS = 3000

def safe_json_loads(s):
    if not isinstance(s, str): return None
    try: return json.loads(s)
    except json.JSONDecodeError: return None

def deep_copy(d):
    return json.loads(json.dumps(d))


# ── Source 1: Real model errors (hard negatives) ──────────

def extract_real_error_pairs(eval_df: pd.DataFrame) -> list[dict]:
    """
    For each sample where 0 < AVG Score < 0.98:
      - chosen = gold JSON (Step Object)
      - rejected = model JSON output (Model column)
    
    Strategy: prefer near-misses (score 0.70-0.95) where model
    almost got it right but made a subtle mistake. These are the
    most informative for DPO.
    """
    pairs = []
    
    # Tier 1: Near-misses (most informative)
    mask_near = (eval_df["AVG Score"] >= 0.70) & (eval_df["AVG Score"] < 0.95)
    # Tier 2: Partial errors (still useful)
    mask_partial = (eval_df["AVG Score"] >= 0.40) & (eval_df["AVG Score"] < 0.70)
    # Tier 3: Complete failures (less informative but add diversity)
    mask_fail = (eval_df["AVG Score"] > 0) & (eval_df["AVG Score"] < 0.40)
    
    tiers = [
        (mask_near, 0.65),    # 65% sampling from near-misses
        (mask_partial, 0.40), # 40% from partial
        (mask_fail, 0.20),    # 20% from failures (adds structural diversity)
    ]
    
    for mask, sample_rate in tiers:
        subset = eval_df[mask]
        sampled = subset.sample(frac=sample_rate, random_state=42)
        for _, row in sampled.iterrows():
            gold_json = safe_json_loads(str(row["Step Object"]))
            if gold_json is None:
                continue
            model_output = str(row["Model"])
            # Skip if model output is empty or same as gold
            if not model_output or model_output == str(row["Step Object"]):
                continue
            # Skip if model output is valid JSON matching gold
            model_obj = safe_json_loads(model_output)
            if model_obj is not None and model_obj == gold_json:
                continue
            
            pairs.append({
                "source": "real_error",
                "prompt": str(row["Sub Task"]),
                "chosen": json.dumps(gold_json, ensure_ascii=False),
                "rejected": model_output,
                "score": float(row["AVG Score"]),
            })
    
    print(f"  real_error (near-miss + partial + fail): {len(pairs)}")
    return pairs


# ── Source 2: Structural validity (invalid JSON) ─────────

def extract_validity_pairs(eval_df: pd.DataFrame) -> list[dict]:
    """
    For samples where model produced invalid JSON:
      - chosen = gold JSON (valid)
      - rejected = model output (invalid)
    
    This teaches the model structural correctness.
    """
    pairs = []
    invalid = eval_df[eval_df["Valid_JSON"] == False]
    for _, row in invalid.iterrows():
        gold_json = safe_json_loads(str(row["Step Object"]))
        if gold_json is None:
            continue
        model_output = str(row["Model"])
        if not model_output:
            continue
        pairs.append({
            "source": "validity",
            "prompt": str(row["Sub Task"]),
            "chosen": json.dumps(gold_json, ensure_ascii=False),
            "rejected": model_output,
            "score": 0.0,
        })
    
    print(f"  validity (invalid JSON): {len(pairs)}")
    return pairs


# ── Source 3: Synthetic corruption (reduced) ──────────────

SHORTHAND_CSS = {"padding", "margin", "border", "font", "background"}
HEX_PAT = __import__("re").compile(r"#[0-9a-fA-F]{6}\b")
CANONICAL_WRONG = {
    "navigation": "href",
    "text-decoration-line": "text-decoration",
    "background-color": "bg-color",
    "visibility": "display",
}
BOOL_MAP = {"visible": "hidden", "hidden": "visible",
            "enabled": "disabled", "disabled": "enabled",
            "true": "false", "false": "true"}
COLOR_MAP = {"green": "red", "red": "green", "blue": "gray",
             "white": "black", "black": "white"}

def corrupt_merge_shorthand(d):
    d = deep_copy(d)
    exp = d.get("expected", {})
    if not isinstance(exp, dict): return d
    for prop in SHORTHAND_CSS:
        sides = [f"{prop}-top", f"{prop}-right", f"{prop}-bottom", f"{prop}-left"]
        if all(s in exp for s in sides):
            val = exp[sides[0]]
            for s in sides: exp.pop(s, None)
            exp[prop] = val; break
    d["expected"] = exp; return d

def corrupt_inject_rgb(d):
    d = deep_copy(d)
    exp = d.get("expected", {})
    if not isinstance(exp, dict): return d
    for k, v in list(exp.items()):
        if isinstance(v, str) and HEX_PAT.fullmatch(v.strip()):
            h = v.strip().lstrip("#")
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            exp[k] = f"rgb({r}, {g}, {b})"; break
    d["expected"] = exp; return d

def corrupt_wrong_key(d):
    d = deep_copy(d)
    exp = d.get("expected", {})
    if not isinstance(exp, dict): return d
    for canon, wrong in CANONICAL_WRONG.items():
        if canon in exp:
            exp[wrong] = exp.pop(canon); break
    d["expected"] = exp; return d

def corrupt_swap_bool(d):
    d = deep_copy(d)
    exp = d.get("expected", {})
    if not isinstance(exp, dict): return d
    for k, v in list(exp.items()):
        if isinstance(v, str) and v.lower() in BOOL_MAP:
            exp[k] = BOOL_MAP[v.lower()]; break
    d["expected"] = exp; return d

def corrupt_drop_key(d):
    d = deep_copy(d)
    exp = d.get("expected", {})
    if not isinstance(exp, dict) or len(exp) <= 1: return d
    drop = random.choice(list(exp.keys()))
    exp.pop(drop); d["expected"] = exp; return d

CORRUPTIONS = [
    corrupt_merge_shorthand, corrupt_inject_rgb, corrupt_wrong_key,
    corrupt_swap_bool, corrupt_drop_key,
]

def build_synthetic_pairs(train_df: pd.DataFrame, n_needed: int) -> list[dict]:
    pairs = []
    rows = train_df.sample(frac=1, random_state=42).to_dict("records")
    for row in rows:
        if len(pairs) >= n_needed: break
        gold_obj = safe_json_loads(str(row["step"]))
        if not isinstance(gold_obj, dict) or not gold_obj.get("expected"):
            continue
        corrupt_fn = random.choice(CORRUPTIONS)
        corrupted = corrupt_fn(gold_obj)
        if corrupted == gold_obj: continue
        pairs.append({
            "source": f"synthetic_{corrupt_fn.__name__}",
            "prompt": str(row["subtask"]),
            "chosen": json.dumps(gold_obj, ensure_ascii=False),
            "rejected": json.dumps(corrupted, ensure_ascii=False),
        })
    return pairs


# ── Main ────────────────────────────────────────────────

def main():
    eval_df = pd.read_excel(EVAL_RESULT_PATH)
    train_df = pd.read_excel(TRAIN_PATH)
    print(f"Eval: {len(eval_df)} rows | Train: {len(train_df)} rows")
    
    pairs = []
    
    # Source 1: Real model errors
    pairs += extract_real_error_pairs(eval_df)
    
    # Source 2: Validity pairs
    pairs += extract_validity_pairs(eval_df)
    
    print(f"\nReal pairs total: {len(pairs)}")
    
    # Source 3: Synthetic (only if needed)
    if len(pairs) < TARGET_PAIRS:
        n_needed = TARGET_PAIRS - len(pairs)
        synthetic = build_synthetic_pairs(train_df, n_needed)
        print(f"  + synthetic (diversity): {len(synthetic)}")
        pairs += synthetic
    
    # Dedup by (prompt, chosen, rejected)
    seen = set()
    deduped = []
    for p in pairs:
        key = (p["prompt"], p["chosen"], p["rejected"])
        if key in seen: continue
        seen.add(key)
        # Drop score key before saving
        d = {"source": p["source"], "prompt": p["prompt"],
             "chosen": p["chosen"], "rejected": p["rejected"]}
        deduped.append(d)
    
    print(f"\nTotal after dedup: {len(deduped)}")
    print("\nSource distribution:")
    for src, cnt in Counter(p["source"] for p in deduped).most_common():
        print(f"  {src}: {cnt}")
    
    # Score distribution of real pairs
    real_pairs = [p for p in pairs if p.get("score") is not None]
    if real_pairs:
        scores = [p["score"] for p in real_pairs]
        print(f"\nReal pair score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"Real pair mean score: {sum(scores)/len(scores):.4f}")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for p in deduped:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\nSaved: {OUTPUT_PATH}")
    if len(deduped) < 500:
        print("WARNING: under 500 pairs — DPO may be unstable.")


if __name__ == "__main__":
    main()
