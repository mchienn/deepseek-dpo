"""Create deterministic, leakage-safe v5 splits from a labelled evaluation pool."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from v5_run import create_manifest, write_json


def row_key(row: pd.Series) -> str:
    return hashlib.sha256(
        (str(row["Sub Task"]).strip() + "\n" + str(row["Step Object"]).strip()).encode("utf-8")
    ).hexdigest()


def bucket(key: str) -> int:
    return int(key[:8], 16) % 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="test_cleaned.xlsx")
    parser.add_argument("--output-dir", default="data/v5")
    parser.add_argument("--mine-percent", type=int, default=60)
    parser.add_argument("--validation-percent", type=int, default=20)
    args = parser.parse_args()
    if not 0 < args.mine_percent < 100 or not 0 < args.validation_percent < 100 or args.mine_percent + args.validation_percent >= 100:
        raise SystemExit("split percentages must be positive and leave a final-test partition")

    source = Path(args.source)
    out = Path(args.output_dir)
    df = pd.read_excel(source)
    required = {"Sub Task", "Step Object"}
    if not required.issubset(df.columns):
        raise SystemExit(f"source must contain {sorted(required)}")
    df = df.copy()
    df["_v5_key"] = df.apply(row_key, axis=1)
    duplicate_rows = int(df["_v5_key"].duplicated(keep=False).sum())
    # Exact duplicates are allowed, but their identical hash keeps the whole group in one split.
    buckets = df["_v5_key"].map(bucket)
    mine = df[buckets < args.mine_percent]
    validation = df[(buckets >= args.mine_percent) & (buckets < args.mine_percent + args.validation_percent)]
    final_test = df[buckets >= args.mine_percent + args.validation_percent]
    out.mkdir(parents=True, exist_ok=True)
    splits = {"dpo_mine": mine, "dpo_validation": validation, "final_test": final_test}
    for name, split in splits.items():
        split.drop(columns="_v5_key").to_excel(out / f"{name}.xlsx", index=False)
        (out / f"{name}.keys.txt").write_text("\n".join(split["_v5_key"]) + "\n", encoding="utf-8")
    overlap = any(set(a["_v5_key"]) & set(b["_v5_key"]) for a, b in ((mine, validation), (mine, final_test), (validation, final_test)))
    if overlap or min(len(x) for x in splits.values()) == 0:
        raise SystemExit("invalid split: overlap or empty partition")
    report = {"source": str(source), "rows": len(df), "counts": {name: len(split) for name, split in splits.items()}, "duplicate_rows_grouped": duplicate_rows, "rule": "sha256(prompt + newline + gold) modulo 100"}
    write_json(out / "split_manifest.json", report)
    create_manifest(out, kind="v5_split", config=report, inputs={"source": source})
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
