# DPO Fine-Tuning Report: Analysis & Improvement Plan

## 1. Executive Summary

| Metric | SFT Baseline | DPO v1 | DPO v2 | DPO v3 (latest) | Target |
|--------|:-----------:|:------:|:------:|:---------------:|:------:|
| Pass Rate | 91.04% | 90.47% | 90.47% | 91.04% | **>91.5%** |
| Avg Score | 0.8998 | 0.8945 | 0.8945 | 0.8998 | **>0.905** |
| Score = 1.0 | 3,676 | 3,398 | 3,398 | 3,676 | **>3,800** |
| Invalid JSON | 38 | 69 | 69 | 38 | **<20** |
| Test samples | 9,427 | 9,427 | 9,427 | 9,427 | **10,050** |

**DPO v1/v2 = regression.** Pass rate dropped −0.57%, invalid JSON doubled, Score=1.0 cases dropped by 278.

**DPO v3 = identical to SFT.** Training ran (loss decreased 0.70→0.11, weights changed) but downstream eval shows no difference. Suggests **preference pairs lack informative signal** for the full test distribution.

---

## 2. What We Did

### 2.1. Training Pipeline (SFT)

- **Model:** DeepSeek-R1-Distill-Qwen-1.5B (1.77B params)
- **SFT:** LoRA r=32, alpha=64, bf16, Qwen chat template
- **Data:** ~6,000 rows Web Automation JSON
- **Eval:** 91.04% Pass, 0.8998 Avg Score (9,427 test samples)
- **Adapter:** `final_adapter/` (on HF Hub + Vast local)

### 2.2. DPO Pipeline

#### Version History

| Version | Files | Key Features | Result |
|---------|-------|-------------|--------|
| v1 | `build_dpo_pairs.py` `train_dpo.py` | 4 sources + synthetic, wrap_pair() with reasoning, MAX_LEN=512, epochs=3, beta=0.1 | 90.47% (regression) |
| v2 | train_dpo_v2.py | Same approach, separate copy | 90.47% |
| v3 | Updated train_dpo.py | MAX_LEN=1024, epochs=2, beta=0.2, removed wrap_pair() | 91.04% (identical to SFT) |

#### Bugs Fixed (v3)

| # | Bug | Fix |
|---|-----|-----|
| 1 | Reasoning leak | Removed `wrap_pair()` — chosen/rejected now plain JSON |
| 2 | Missing ternary corruption | Added `corrupt_ternary()` to CORRUPTIONS |
| 3 | Completion truncation | `MAX_LEN` 512→1024, `max_prompt_length=256` |
| 4 | Overfitting | epochs 3→2, beta 0.1→0.2 |
| 5 | Eval loading wrong adapter | Removed `merge_and_unload()`, forced `device_map="cuda:0"` |
| 6 | Eval wrong dtype | Fixed `dtype` → `torch_dtype` |

#### Infrastructure Setup

- Git repo: `github.com/mchienn/deepseek-dpo`
- Vast instances: RTX 4090 (16GB→OOM), RTX 5090 (32GB→success)
- HF Hub: `nmc27705/deepseek-r1-distill-qwen-1.5b-lora-adapter`
- Logging: auto-tee to `train_*.log` + TensorBoard auto-start
- Adapter transfer: base64 SSH pipe → hf-mirror wget

---

## 3. Root Cause Analysis

### 3.1. DPO v1/v2 Regression

**Primary cause:** Preference pairs taught model to avoid specific corrupted patterns, but at the cost of forgetting correct outputs on many held-out cases (278 samples dropped from Score=1.0 → lower bins, 213 SFT Pass → DPO Fail).

Key finding from research: DPO inherently biases toward **lowering probability of rejected responses** without necessarily **raising probability of chosen responses** ("Rethinking DPO: The Role of Rejected Responses in Preference Misalignment", EMNLP 2025). Our corrupted rejections were easy to distinguish, so DPO just suppressed them — but this also suppressed correct outputs sharing similar surface forms.

### 3.2. DPO v3 = SFT Identity

Despite loss decreasing and weights changing, eval shows identical scores. Possible explanations:

1. **Preference pairs lack test-distribution coverage.** The 2000 pairs (mostly synthetic corruptions of common patterns) don't overlap with the 9427-sample eval distribution. Model learns to distinguish synthetic corruptions but never encounters these patterns at inference time.

2. **LoRA capacity insufficient.** With only 2% trainable params (42M/1.77B), the model cannot meaningfully shift behavior on unseen samples. The weight changes affect the DPO training distribution but don't generalize.

3. **Passive collapse.** The loss decreases because the model learns to increase chosen/rejected reward gap on training pairs, but this gap is dataset-specific and doesn't transfer. The implicit reward model overfits to training distribution.

### 3.3. Value Score Gap

Across all versions, Value Scores (0.8609) lag Key Scores (0.9280) by ~0.067. This gap **widened** under DPO. The model identifies correct CSS properties but struggles with specific values (colors, sizes, booleans). This is a fundamental limitation of token-level probability optimization — precise numeric/color values require exact logit placement which is harder to modulate via preference pairs than structural choices.

---

## 4. Improvement Plan

Based on systematic analysis + literature review (15+ papers), ordered by expected impact:

### 4.1. Phase 1: Fix Data Quality (Highest Impact)

**Problem:** Current DPO pairs are synthetic corruptions with weak signal. Research shows **chosen response quality dominates DPO performance** ("What Matters in Data for DPO", 2025).

**Action Items:**

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 1.1 | **Mine hard negatives from eval checkpoint.** Use `eval_checkpoint.jsonl` (9,427 samples of model outputs). For each sample where model scored <1.0, use SFT gold (1.0) as chosen and model output as rejected. | High | 1d |
| 1.2 | **Replace synthetic corruptions with real model errors.** Instead of programmatic corruption (shorthand→expanded, rgb→hex), use actual inference failures. Focus on near-miss pairs (score 0.85 vs 0.95). | High | 1d |
| 1.3 | **Apply D-optimal negative selection.** Research shows selecting negatives that span complementary directions in parameter space is optimal (MASS-DPO, 2025). For each prompt, select 2-3 negatives covering different failure modes. | Medium-High | 2d |
| 1.4 | **Add structural validation penalty.** Include pairs where chosen = valid JSON, rejected = invalid JSON. This directly addresses the doubled invalid JSON rate in DPO v1/v2. | Medium | 1d |

### 4.2. Phase 2: Training Hyperparameters

**Problem:** Static beta (0.2) treats all pairs equally. Research shows optimal beta varies with pair informativeness.

**Action Items:**

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 2.1 | **Dynamic beta scheduling.** Implement batch-level beta calibration (β-DPO framework): high beta for easy pairs (large reward gap), low beta for hard pairs (small gap). | Medium-High | 2d |
| 2.2 | **Iterative DPO (2 rounds).** Train DPO → sample from new policy → collect preferences → retrain. This closes the gap with on-policy RLHF (Llama-3 post-training). | Medium-High | 2d |
| 2.3 | **Tune beta via grid search.** Try beta = [0.05, 0.1, 0.2, 0.3, 0.5]. Current beta=0.2 was chosen a priori. Lower beta allows more aggressive learning. | Medium | 1d |
| 2.4 | **Increase effective batch size.** From 16 (current) to 32-64. Research shows effective batch 32-128 stabilizes DPO gradient estimates. | Medium | 1d |

### 4.3. Phase 3: Architecture & Loss Modifications

**Problem:** Vanilla DPO has known limitations — doesn't guarantee chosen probability increase, sensitive to distribution shift, poor OOD generalization.

**Action Items:**

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 3.1 | **Add NLL loss term (DPO+NLL).** `L = L_DPO + λ * NLL(chosen)`. This prevents the common failure mode where DPO decreases chosen probability while increasing the reward gap. | Medium | 1d |
| 3.2 | **Bounded-DPO (BDPO).** Replace standard DPO loss with BDPO formulation that bounds rejected-response influence via mixture distribution. This directly addresses the regression where correct outputs share surface forms with rejected outputs. | Medium-High | 2d |
| 3.3 | **Consider SimPO.** Reference-free preference optimization with length-normalized reward. Simplified training, fewer hyperparameters, competitive results in literature. | Medium | 1d |

### 4.4. Phase 4: Evaluation Infrastructure

**Problem:** Current eval misses 623 samples (6.2% of test set), and eval script was loading wrong adapter.

**Action Items:**

| # | Action | Expected Impact | Effort |
|---|--------|----------------|--------|
| 4.1 | **Fix eval to cover 100% of test set.** Set `max_new_tokens=512` (vs 256), add per-sample timeout. | High (measurement) | 1d |
| 4.2 | **Version eval results explicitly.** Include adapter hash, date, config in output filename. | Medium (reproducibility) | 0.5d |
| 4.3 | **CI script for controlled comparison.** Single command to run eval on SFT vs DPO and output delta table. | Medium | 1d |

---

## 5. Research References

| Paper | Key Insight | Relevance |
|-------|-------------|-----------|
| **What Matters in Data for DPO** (2025) | Chosen quality >> rejected quality | Phase 1 design |
| **MASS-DPO** (2025) | D-optimal negative selection | Phase 1.3 |
| **Rethinking DPO: Role of Rejected** (2025) | DPO fails to increase chosen prob | Phase 3.1, 3.2 |
| **β-DPO** (2025) | Dynamic beta calibration | Phase 2.1 |
| **Bounded-DPO (BDPO)** (2025) | Mixture distribution bound on rejected | Phase 3.2 |
| **SimPO** (2024) | Reference-free, length-normalized | Phase 3.3 |
| **Unpacking DPO vs PPO** (2025) | PPO > DPO; data quality > algorithm | Context |
| **Difficulty-Based Data Selection** (2025) | Small reward gap = informative pairs | Phase 1.1 |
| **Pre-DPO** (2026, AAAI) | Guiding reference model | Phase 3 |
| **Dynamic Noise PO** (2025) | Self-generated data can beat human | Phase 2.2 |
| **Margin-Adaptive DPO** (2025) | Per-sample weight from reward margin | Phase 2.1 |
| **ξ-DPO** (2026) | Ratio reward margin, robust beta | Phase 2.1 |

---

## 6. Recommended Next Step

**Immediate (highest ROI):** Fix preference data quality using real model errors from `eval_checkpoint.jsonl` instead of synthetic corruptions.

Goal: Go from 0 pairs using real model errors (current) to **>500 pairs with model-output-as-rejected**.

This requires:
1. Parse `eval_checkpoint.jsonl` for all 9,427 samples
2. For each score < 0.95, create pair (gold_chosen, model_rejected)
3. Filter to keep most informative pairs (score gap 0.05-0.30)
4. Run DPO with these real-error pairs + reduced synthetic corruption

Estimated timeline: **2-3 days** for full implementation + training + eval cycle.
