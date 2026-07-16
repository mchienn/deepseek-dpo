"""Validate v5 split isolation and DPO pair JSON integrity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_keys(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", default="data/v5")
    parser.add_argument("--pairs")
    args = parser.parse_args()
    root = Path(args.split_dir)
    names = ["dpo_mine", "dpo_validation", "final_test"]
    keys = {name: load_keys(root / f"{name}.keys.txt") for name in names}
    if min(map(len, keys.values())) == 0:
        raise SystemExit("one or more split key files are empty")
    for left, right in (("dpo_mine", "dpo_validation"), ("dpo_mine", "final_test"), ("dpo_validation", "final_test")):
        if keys[left] & keys[right]:
            raise SystemExit(f"split overlap detected: {left} / {right}")
    result = {"splits": {name: len(value) for name, value in keys.items()}}
    if args.pairs:
        count = 0
        seen: set[tuple[str, str, str]] = set()
        for number, raw in enumerate(Path(args.pairs).read_text(encoding="utf-8").splitlines(), 1):
            item = json.loads(raw)
            triple = tuple(item.get(field, "") for field in ("prompt", "chosen", "rejected"))
            if not all(triple) or triple in seen:
                raise SystemExit(f"invalid or duplicate pair at line {number}")
            chosen_json = item["chosen"].rsplit("</think>", 1)[-1].strip()
            json.loads(chosen_json)
            if item["chosen"] == item["rejected"]:
                raise SystemExit(f"identical chosen/rejected at line {number}")
            seen.add(triple)
            count += 1
        result["pairs"] = count
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
