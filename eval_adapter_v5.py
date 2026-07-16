"""CLI wrapper around the legacy scorer with run-isolated checkpoints and a manifest."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from v5_run import create_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="discard only this run's inference checkpoint")
    args = parser.parse_args()
    adapter, test_file = Path(args.adapter).resolve(), Path(args.test_file).resolve()
    if not adapter.is_dir() or not test_file.is_file():
        raise SystemExit("adapter directory and test workbook must exist")
    run_dir = Path(args.runs_dir).resolve() / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "inference.jsonl"
    if args.force and checkpoint.exists():
        checkpoint.unlink()
    output = run_dir / "evaluation.xlsx"
    create_manifest(run_dir, kind="v5_evaluation", config={"run_id": args.run_id, "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size, "checkpoint": str(checkpoint)}, inputs={"adapter_config": adapter / "adapter_config.json", "test_file": test_file})
    import eval_deepseek as scorer
    scorer.ADAPTER_DIR = str(adapter)
    scorer.TEST_FILE = str(test_file)
    scorer.OUTPUT_FILE = str(output)
    scorer.MAX_NEW_TOKENS = args.max_new_tokens
    scorer.HF_BATCH = args.batch_size
    original_cwd = Path.cwd()
    os.chdir(run_dir)
    try:
        scorer.main()
    finally:
        os.chdir(original_cwd)
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
