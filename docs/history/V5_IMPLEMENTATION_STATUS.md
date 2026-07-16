# Version history — V5 implementation status

Date: 2026-07-16

## Delivered locally

- Added deterministic leakage-safe split tooling: `prepare_v5_splits.py` and `validate_v5_data.py`.
- Created `data/v5/` from `test_cleaned.xlsx`: 5,738 mining rows, 1,839 validation rows, 1,850 final-test rows. Six exact duplicate rows are grouped by identical hash, so they cannot cross a split boundary.
- Added `v5_run.py` for SHA-256 input manifests and run metadata.
- Added `eval_adapter_v5.py`, which isolates output and inference checkpoint inside `runs/<run-id>/`.
- Added `build_dpo_pairs_v5.py`, which refuses an evaluation workbook unless it exactly corresponds to the mining split.
- Added `train_dpo_v5.py`, `analyze_v5_runs.py`, `setup_vast_v5.sh`, and `run_v5_candidate.sh`.
- Pinned `pyarrow==19.0.1` in the V5 Vast setup because local PyArrow 18.1.0 lacks `pyarrow.json_`, required by the installed `datasets` package.

## Not yet executed

No GPU inference, DPO training, validation evaluation, final-test evaluation, or candidate-selection analysis has run for V5. These require a prepared Vast instance and a verified SFT adapter. Consequently, no V5 quality claim is made in this history entry.

## Required run order

1. Push this implementation to the Vast repository and run `setup_vast_v5.sh`.
2. Run one candidate with `run_v5_candidate.sh <run-id>`; it mines only `dpo_mine` and evaluates the resulting adapter only on `dpo_validation`.
3. Compare baseline/candidate validation outputs with `analyze_v5_runs.py`.
4. Select configuration across planned seeds before evaluating the winning candidates on `final_test` once.
5. Append final metrics, manifest paths and the deployment decision to this file.

## Verification evidence

The local non-GPU gates were run on 2026-07-16:

- `prepare_v5_splits.py` created 5,738 mine rows, 1,839 validation rows and 1,850 final-test rows from 9,427 source rows; six duplicate rows remained grouped by hash.
- `validate_v5_data.py` confirmed no key overlap across partitions.
- A synthetic mine-only evaluation workbook exercised `build_dpo_pairs_v5.py`; it produced 4,016 non-duplicate, JSON-valid chosen/rejected pairs, then passed pair validation.
- `analyze_v5_runs.py` reproduced the current SFT vs DPO v4 delta from aligned workbooks.
- `python -m py_compile` passed for all V5 Python entry points, and `bash -n` passed for both Vast V5 scripts.

The local runtime smoke test for `train_dpo_v5.py --help` is intentionally not a GPU readiness result: this workstation has Python 3.13 with PyArrow 18.1.0, while its installed `datasets` calls `pyarrow.json_`. The V5 Vast setup pins PyArrow 19.0.1, whose official API includes `pyarrow.json_`.

## Vast execution status

Attempted on 2026-07-16 after pushing branch `codex/dpo-v5-reproducible`. The configured Vast SSH endpoint closed the connection on port 3831 before authentication or any remote command ran. No checkout, dependency installation, data upload, GPU inference, training, or evaluation was performed on the remote instance. Resume from the required run order only after the instance is active and its SSH endpoint is updated/verified.
