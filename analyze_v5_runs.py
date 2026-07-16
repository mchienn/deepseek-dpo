"""Compare two run-isolated evaluation workbooks and write a machine-readable delta."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from v5_run import write_json


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    valid = frame["Valid_JSON"].fillna(False).astype(bool)
    return {
        "rows": int(len(frame)),
        "pass_rate": float((frame["Result"] == "Pass").mean() * 100),
        "avg_score": float(frame["AVG Score"].mean()),
        "invalid_json": int((~valid).sum()),
        "key_score": float(frame["Key Scores"].mean()),
        "value_score": float(frame["Value Scores"].mean()),
        "perfect": int((frame["AVG Score"] == 1).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline, candidate = pd.read_excel(args.baseline), pd.read_excel(args.candidate)
    keys = ["Sub Task", "Step Object"]
    if len(baseline) != len(candidate) or not baseline[keys].equals(candidate[keys]):
        raise SystemExit("workbooks are not aligned; comparison refused")
    base, cand = metrics(baseline), metrics(candidate)
    delta = {key: cand[key] - base[key] for key in base if key != "rows"}
    result = {"baseline": base, "candidate": cand, "delta": delta}
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
