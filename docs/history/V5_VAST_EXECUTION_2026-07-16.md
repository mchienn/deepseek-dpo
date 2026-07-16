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

## Candidate 02 — corrected completion format

Candidate 02 used `Full Output` for rejected completions, preserved the `<think>...</think>` prefix in all 1,569 chosen completions, and reduced optimization to one epoch with beta 0.01 and learning rate 1e-6. The pair and split validators passed before training.

| Metric | SFT baseline | DPO candidate 02 | Delta |
| --- | ---: | ---: | ---: |
| Pass rate | 91.46% | 90.21% | -1.25 pp |
| Average score | 0.9014 | 0.8940 | -0.0074 |
| Invalid JSON | 11 | 10 | -1 |
| Key score | 0.9287 | 0.9220 | -0.0067 |
| Value score | 0.8741 | 0.8659 | -0.0082 |
| Perfect outputs | 706 | 674 | -32 |

Decision: **reject candidate 02 for final test**. The format fix eliminated the catastrophic regression from candidate 01 and slightly improved JSON validity, but it did not exceed the SFT baseline on the pre-registered primary metrics.

## Remaining work

The experimental gate is complete for candidates 01 and 02, but the overall DPO-improvement goal is not complete: neither candidate qualifies for final-test evaluation or deployment. Before another GPU run, implement and validate a conservative preference objective that anchors the SFT behavior (for example, DPO plus a supervised/NLL anchor), then run one controlled validation candidate. Keep `final_test` untouched until a candidate exceeds the SFT validation baseline.