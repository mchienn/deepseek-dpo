"""
build_dpo_pairs.py
===================
Bước 0 của lộ trình DPO — build preference pairs từ 4 nguồn:

  1. invalid_json       — model tự sinh JSON lỗi
  2. shorthand_revert   — model trả về shorthand thay vì expanded keys
  3. rgb_to_hex_fix     — gold tự nó có lỗi rgb() thay vì hex
  4. hallucination_fix  — gold có quoted entity không tồn tại trong input
  5. synthetic_corruption — corrupt có chủ đích từ train_reasoning_fixed.xlsx

Input:
  - eval_deepseek_result.xlsx  (cột: Sub Task, Step Object, Model, Full Output, Valid_JSON)
  - train_reasoning_fixed.xlsx (cột: subtask, step, reasoning)

Output:
  - dpo_pairs.jsonl   (mỗi dòng: {"prompt", "chosen", "rejected", "source"})

Format:
  prompt   = <|im_start|>user\n{subtask}<|im_end|>\n<|im_start|>assistant\n
  chosen   = {json_str}  (plain JSON — no <think>, no <|im_end|>)
  rejected = {json_str}  (plain JSON)

Chạy: python build_dpo_pairs.py
"""

import json
import re
import random
from collections import Counter
import pandas as pd

random.seed(42)

EVAL_PATH  = "eval_deepseek_result.xlsx"
TRAIN_PATH = "train_reasoning_fixed.xlsx"
OUTPUT_PATH = "dpo_pairs.jsonl"

MIN_PAIRS_TARGET = 2000

SHORTHAND_CSS = {"padding", "margin", "border", "font", "background"}

RGB_PAT = re.compile(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
HEX_PAT = re.compile(r"#[0-9a-fA-F]{6}\b")
QUOTE_PAT = re.compile(r"'([^']+)'")

CANONICAL_TO_WRONG_KEY = {
    "navigation": "href",
    "text-decoration-line": "text-decoration",
    "background-color": "bg-color",
    "visibility": "display",
}


# ── Helpers ────────────────────────────────────────────────────────

def safe_json_loads(s):
    if not isinstance(s, str):
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def get_action(step_str):
    j = safe_json_loads(step_str)
    return j.get("action") if isinstance(j, dict) else None


# ── Source 1 — invalid JSON từ model ───────────────────────────────

def extract_invalid_json_pairs(eval_df: pd.DataFrame) -> list[dict]:
    pairs = []
    invalid = eval_df[eval_df["Valid_JSON"] == False]
    for _, row in invalid.iterrows():
        gold_json = safe_json_loads(str(row["Step Object"]))
        if gold_json is None:
            continue
        model_json = str(row["Model"])
        pairs.append({
            "source": "invalid_json",
            "prompt": str(row["Sub Task"]),
            "chosen": json.dumps(gold_json, ensure_ascii=False),
            "rejected": model_json,
        })
    return pairs


# ── Source 2 — shorthand revert ────────────────────────────────────

def is_shorthand_in_model(model_str: str) -> str | None:
    j = safe_json_loads(model_str)
    if not isinstance(j, dict):
        return None
    expected = j.get("expected", {})
    if not isinstance(expected, dict):
        return None
    for prop in SHORTHAND_CSS:
        if prop in expected:
            return prop
    return None


def extract_shorthand_revert_pairs(eval_df: pd.DataFrame) -> list[dict]:
    pairs = []
    for _, row in eval_df.iterrows():
        model_str = str(row["Model"])
        prop = is_shorthand_in_model(model_str)
        if prop is None:
            continue
        gold_act = get_action(str(row["Step Object"]))
        pred_act = get_action(model_str)
        if gold_act is None or gold_act != pred_act:
            continue
        pairs.append({
            "source": f"shorthand_revert_{prop}",
            "prompt": str(row["Sub Task"]),
            "chosen": str(row["Step Object"]),
            "rejected": model_str,
        })
    return pairs


# ── Source 3 — rgb() → hex ─────────────────────────────────────────

def rgb_to_hex(r, g, b) -> str:
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def fix_rgb_in_value(v):
    if isinstance(v, str) and RGB_PAT.search(v):
        return RGB_PAT.sub(lambda m: rgb_to_hex(m.group(1), m.group(2), m.group(3)), v)
    return v


def fix_rgb_recursive(obj):
    if isinstance(obj, dict):
        return {k: fix_rgb_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fix_rgb_recursive(v) for v in obj]
    if isinstance(obj, str):
        return fix_rgb_in_value(obj)
    return obj


def extract_rgb_fix_pairs(eval_df: pd.DataFrame) -> list[dict]:
    pairs = []
    seen = set()
    for _, row in eval_df.iterrows():
        gold_str = str(row["Step Object"])
        subtask = str(row["Sub Task"])
        if gold_str in seen:
            continue
        if not RGB_PAT.search(gold_str):
            continue
        if RGB_PAT.search(subtask):
            continue
        gold_obj = safe_json_loads(gold_str)
        if not isinstance(gold_obj, dict):
            continue
        fixed_obj = fix_rgb_recursive(gold_obj)
        chosen = json.dumps(fixed_obj, ensure_ascii=False)
        if chosen == gold_str:
            continue
        pairs.append({
            "source": "rgb_to_hex_fix",
            "prompt": subtask,
            "chosen": chosen,
            "rejected": gold_str,
        })
        seen.add(gold_str)
    return pairs


# ── Source 4 — hallucinated entity ─────────────────────────────────

def find_hallucinated_entity(selector: str, subtask: str) -> str | None:
    for entity in QUOTE_PAT.findall(selector):
        pattern = r"\b" + re.escape(entity.lower()) + r"\b"
        if not re.search(pattern, subtask.lower()):
            return entity
    return None


def strip_entity_from_selector(selector: str, entity: str) -> str:
    cleaned = re.sub(
        rf"\s*(in|for|of|next to|bên cạnh|trong)?\s*'{re.escape(entity)}'\s*\w*",
        "", selector
    )
    return cleaned.strip()


def extract_hallucination_fix_pairs(eval_df: pd.DataFrame) -> list[dict]:
    pairs = []
    seen = set()
    for _, row in eval_df.iterrows():
        gold_str = str(row["Step Object"])
        subtask = str(row["Sub Task"])
        if gold_str in seen:
            continue
        gold_json = safe_json_loads(gold_str)
        if not isinstance(gold_json, dict):
            continue
        sel = gold_json.get("selector", "")
        if not isinstance(sel, str):
            continue
        entity = find_hallucinated_entity(sel, subtask)
        if entity is None:
            continue
        new_sel = strip_entity_from_selector(sel, entity)
        if not new_sel or new_sel == sel:
            continue
        fixed = dict(gold_json)
        fixed["selector"] = new_sel
        pairs.append({
            "source": "hallucination_fix",
            "prompt": subtask,
            "chosen": json.dumps(fixed, ensure_ascii=False),
            "rejected": gold_str,
        })
        seen.add(gold_str)
    return pairs


# ── Source 5 — synthetic corruption ────────────────────────────────

def deep_copy(d):
    return json.loads(json.dumps(d))


def corrupt_merge_shorthand(d: dict) -> dict:
    d = deep_copy(d)
    expected = d.get("expected", {})
    if not isinstance(expected, dict):
        return d
    for prop in SHORTHAND_CSS:
        sides = [f"{prop}-top", f"{prop}-right", f"{prop}-bottom", f"{prop}-left"]
        if all(s in expected for s in sides):
            val = expected[sides[0]]
            for s in sides:
                expected.pop(s, None)
            expected[prop] = val
            break
    d["expected"] = expected
    return d


def corrupt_inject_rgb(d: dict) -> dict:
    d = deep_copy(d)
    expected = d.get("expected", {})
    if not isinstance(expected, dict):
        return d
    for k, v in list(expected.items()):
        if isinstance(v, str) and HEX_PAT.fullmatch(v.strip()):
            hexv = v.strip().lstrip("#")
            r, g, b = int(hexv[0:2], 16), int(hexv[2:4], 16), int(hexv[4:6], 16)
            expected[k] = f"rgb({r}, {g}, {b})"
            break
    d["expected"] = expected
    return d


def corrupt_wrong_key(d: dict) -> dict:
    d = deep_copy(d)
    expected = d.get("expected", {})
    if not isinstance(expected, dict):
        return d
    for canon, wrong in CANONICAL_TO_WRONG_KEY.items():
        if canon in expected:
            expected[wrong] = expected.pop(canon)
            break
    d["expected"] = expected
    return d


def corrupt_swap_bool(d: dict) -> dict:
    d = deep_copy(d)
    expected = d.get("expected", {})
    if not isinstance(expected, dict):
        return d
    bool_map = {"visible": "hidden", "hidden": "visible",
                "enabled": "disabled", "disabled": "enabled",
                "true": "false", "false": "true"}
    for k, v in list(expected.items()):
        if isinstance(v, str) and v.lower() in bool_map:
            expected[k] = bool_map[v.lower()]
            break
    d["expected"] = expected
    return d


def corrupt_drop_key(d: dict) -> dict:
    d = deep_copy(d)
    expected = d.get("expected", {})
    if not isinstance(expected, dict) or len(expected) <= 1:
        return d
    drop = random.choice(list(expected.keys()))
    expected.pop(drop)
    d["expected"] = expected
    return d


def corrupt_ternary(d: dict) -> dict:
    """Inject ternary expression into a color/bool value — model must learn to avoid."""
    d = deep_copy(d)
    expected = d.get("expected", {})
    if not isinstance(expected, dict):
        return d
    color_map = {
        "green": "red", "red": "green", "blue": "gray",
        "white": "black", "black": "white", "lightgreen": "lightcoral",
    }
    for k, v in list(expected.items()):
        if isinstance(v, str) and v.lower() in color_map:
            opposite = color_map[v.lower()]
            expected[k] = f'"{v}" if entry is active else "{opposite}"'
            break
    d["expected"] = expected
    return d


CORRUPTIONS = [
    corrupt_merge_shorthand,
    corrupt_inject_rgb,
    corrupt_wrong_key,
    corrupt_swap_bool,
    corrupt_drop_key,
    corrupt_ternary,
]


def build_synthetic_pairs(train_df: pd.DataFrame, n_needed: int) -> list[dict]:
    pairs = []
    rows = train_df.sample(frac=1, random_state=42).to_dict("records")
    for row in rows:
        if len(pairs) >= n_needed:
            break
        gold_json = safe_json_loads(str(row["step"]))
        if not isinstance(gold_json, dict) or not gold_json.get("expected"):
            continue
        corrupt_fn = random.choice(CORRUPTIONS)
        corrupted = corrupt_fn(gold_json)
        if corrupted == gold_json:
            continue
        pairs.append({
            "source": f"synthetic_{corrupt_fn.__name__}",
            "prompt": str(row["subtask"]),
            "chosen": json.dumps(gold_json, ensure_ascii=False),
            "rejected": json.dumps(corrupted, ensure_ascii=False),
        })
    return pairs


# ── Main ──────────────────────────────────────────────────────────

def main():
    eval_df = pd.read_excel(EVAL_PATH)
    train_df = pd.read_excel(TRAIN_PATH)

    print(f"Loaded eval: {len(eval_df)} rows | train: {len(train_df)} rows")

    pairs = []
    pairs += extract_invalid_json_pairs(eval_df)
    print(f"  + invalid_json:        {len(pairs)}")

    before = len(pairs)
    pairs += extract_shorthand_revert_pairs(eval_df)
    print(f"  + shorthand_revert:    {len(pairs) - before}")

    before = len(pairs)
    pairs += extract_rgb_fix_pairs(eval_df)
    print(f"  + rgb_to_hex_fix:      {len(pairs) - before}")

    before = len(pairs)
    pairs += extract_hallucination_fix_pairs(eval_df)
    print(f"  + hallucination_fix:   {len(pairs) - before}")

    print(f"\nReal pairs: {len(pairs)}")

    if len(pairs) < MIN_PAIRS_TARGET:
        n_needed = MIN_PAIRS_TARGET - len(pairs)
        synthetic = build_synthetic_pairs(train_df, n_needed)
        print(f"  + synthetic (bổ sung): {len(synthetic)}")
        pairs += synthetic

    # Dedup by (prompt, chosen, rejected)
    seen = set()
    deduped = []
    for p in pairs:
        key = (p["prompt"], p["chosen"], p["rejected"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    print(f"\nTotal after dedup: {len(deduped)}")

    print("\nPhân phối theo source:")
    for src, cnt in Counter(p["source"] for p in deduped).most_common():
        print(f"  {src}: {cnt}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for p in deduped:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nSaved: {OUTPUT_PATH} ({len(deduped)} pairs)")
    if len(deduped) < 500:
        print("CẢNH BÁO: dưới 500 pairs — DPO có thể unstable.")


if __name__ == "__main__":
    main()
