"""Build leakage-safe DPO v5 pairs from SFT errors on dpo_mine only."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from v5_run import create_manifest, write_json


def parse_json(value: object):
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return None


def canonical(value: object) -> str:
    parsed = parse_json(value)
    return json.dumps(parsed, ensure_ascii=False, sort_keys=True) if parsed is not None else str(value).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True, help="SFT result workbook produced only on dpo_mine")
    parser.add_argument("--mine-file", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--near-rate", type=float, default=0.70)
    parser.add_argument("--partial-rate", type=float, default=0.20)
    parser.add_argument("--fail-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if abs(args.near_rate + args.partial_rate + args.fail_rate - 1.0) > 1e-9:
        raise SystemExit("sampling rates must sum to 1")
    evaluation, mine_file = Path(args.evaluation), Path(args.mine_file)
    eval_df, mine_df = pd.read_excel(evaluation), pd.read_excel(mine_file)
    required = {"Sub Task", "Step Object", "Model", "AVG Score", "Valid_JSON"}
    if not required.issubset(eval_df.columns):
        raise SystemExit(f"evaluation missing columns: {sorted(required - set(eval_df.columns))}")
    mine_keys = {(str(row["Sub Task"]), canonical(row["Step Object"])) for _, row in mine_df.iterrows()}
    eval_keys = {(str(row["Sub Task"]), canonical(row["Step Object"])) for _, row in eval_df.iterrows()}
    if eval_keys != mine_keys:
        raise SystemExit("evaluation rows do not exactly match dpo_mine; refusing to build pairs")
    groups = [
        ("near_miss", eval_df[(eval_df["AVG Score"] >= .70) & (eval_df["AVG Score"] < .95)], args.near_rate),
        ("partial", eval_df[(eval_df["AVG Score"] >= .40) & (eval_df["AVG Score"] < .70)], args.partial_rate),
        ("failure", eval_df[(eval_df["AVG Score"] > 0) & (eval_df["AVG Score"] < .40)], args.fail_rate),
        ("invalid_json", eval_df[eval_df["Valid_JSON"] == False], args.fail_rate),
    ]
    pairs, seen = [], set()
    for source, frame, rate in groups:
        if frame.empty:
            continue
        count = max(1, round(len(frame) * rate))
        for _, row in frame.sample(n=min(count, len(frame)), random_state=args.seed).iterrows():
            chosen_obj = parse_json(row["Step Object"])
            rejected = str(row["Model"]).strip()
            if chosen_obj is None or not rejected:
                continue
            chosen = json.dumps(chosen_obj, ensure_ascii=False, sort_keys=True)
            if canonical(rejected) == chosen:
                continue
            triple = (str(row["Sub Task"]), chosen, rejected)
            if triple in seen:
                continue
            seen.add(triple)
            pairs.append({"source": source, "prompt": triple[0], "chosen": chosen, "rejected": rejected})
    if len(pairs) < 100:
        raise SystemExit(f"only {len(pairs)} pairs built; inspect SFT errors before training")
    run_dir = Path(args.runs_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "pairs.jsonl"
    output.write_text("".join(json.dumps(pair, ensure_ascii=False) + "\n" for pair in pairs), encoding="utf-8")
    summary = {"pairs": len(pairs), "sources": dict(Counter(pair["source"] for pair in pairs)), "evaluation_rows": len(eval_df), "mine_rows": len(mine_df)}
    write_json(run_dir / "pair_summary.json", summary)
    create_manifest(run_dir, kind="v5_pairs", config={**vars(args), **summary}, inputs={"evaluation": evaluation, "mine_file": mine_file})
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
