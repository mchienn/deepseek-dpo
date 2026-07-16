# V5 Vast execution — candidate 01

Date: 2026-07-16
Branch: `codex/dpo-v5-reproducible`
Instance: Vast RTX 5090 (32,607 MiB VRAM)

## Completed artifacts

- Mining-only SFT baseline: `runs/v5-sft-mine/evaluation.xlsx` on `dpo_mine` (5,738 rows).
  - Pass 91.3%, average score 0.9021, valid JSON 99.7%.
- Preference-pair dataset: `runs/v5-dpo-b050-lr5e6-s42/pairs.jsonl`.
  - 1,569 valid, unique pairs: 1,472 near-miss, 94 partial, 1 failure, 2 invalid JSON.
  - Split validator confirmed zero overlap among mine (5,735 unique keys), validation (1,839) and final test (1,850).
- Candidate 01: beta 0.05, learning rate 5e-6, seed 42, 3 epochs, per-device batch 4 and gradient accumulation 4.
  - Best pair-held-out eval loss: 0.0012 at checkpoint 260.
- Controlled SFT baseline evaluation on `dpo_validation`: `runs/v5-sft-validation/evaluation.xlsx`.
- Candidate evaluation on the same `dpo_validation` rows: `runs/v5-dpo-b050-lr5e6-s42-validation/evaluation.xlsx`.

## Validation comparison and decision

| Metric | SFT baseline | DPO candidate 01 | Delta |
| --- | ---: | ---: | ---: |
| Pass rate | 91.46% | 56.06% | -35.40 pp |
| Average score | 0.9014 | 0.6748 | -0.2266 |
| Invalid JSON | 11 | 27 | +16 |
| Key score | 0.9287 | 0.7012 | -0.2275 |
| Value score | 0.8741 | 0.6483 | -0.2258 |
| Perfect outputs | 706 | 94 | -612 |

Decision: **reject candidate 01**. Do not evaluate it on `final_test`; that split remains untouched.

Machine-readable comparison: `runs/v5-dpo-b050-lr5e6-s42-validation/comparison_vs_sft.json` on the Vast workspace.

## Diagnosis and corrective action

The training objective was saturated on the pair-held-out split (best loss 0.0012), while task validation regressed severely. Two direct code/data observations explain why a new run must not reuse this configuration unchanged:

1. `build_dpo_pairs_v5.py` originally used evaluator column `Model` (the JSON-only scoring field) and writes `chosen` as the gold JSON alone, while the deployed task contract expects an assistant completion with `<think>...</think>` followed by JSON. This output-format mismatch is a training-data defect.
2. Candidate 01 applies 3 epochs at 5e-6 to only 1,569 pairs, 93.8% of which are near-misses. The near-zero pair loss plus the task regression is evidence of over-optimization for this narrow preference set.

Next candidate requirements:

- use evaluator `Full Output` to preserve the SFT completion structure in chosen/rejected pairs;
- reduce optimization strength (one epoch and lower LR/beta); and
- gate on the untouched `dpo_validation` comparison before any `final_test` run.
