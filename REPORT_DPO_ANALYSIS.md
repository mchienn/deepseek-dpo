# DPO Fine-Tuning Report: Analysis & Improvement Plan

## 1. Executive Summary

| Metric | SFT Baseline | DPO v1/v2 | DPO v3 | DPO v4 (training) | Target |
|--------|:-----------:|:---------:|:------:|:-----------------:|:------:|
| Pass Rate | 91.04% | 90.47% | 90.47% | **in progress** | **>91.5%** |
| Avg Score | 0.8998 | 0.8945 | 0.8945 | **in progress** | **>0.905** |
| Score = 1.0 | 3,676 | 3,398 | 3,398 | **in progress** | **>3,800** |
| Invalid JSON | 38 | 69 | 69 | **in progress** | **<20** |
| Test samples | 9,427 | 9,427 | 9,427 | 9,427 | **10,050** |

**DPO v1/v2/v3 = regression.** All versions scored 90.47% (worse than SFT 91.04%). Invalid JSON doubled (38→69), Score=1.0 dropped by 278. DPO v3 "identical to SFT" was a measurement artifact — eval was loading wrong adapter.

**DPO v4 (in progress):** Replaced synthetic corruptions with **2,578 real model errors** as rejected pairs. Training on Vast RTX 5090 at step 352/507 with eval_loss=0.41, rewards/acc=79.3%.

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

Note: DPO v3 table row shows 91.04% but this was from eval with wrong adapter loading (SFT weights instead of DPO). Actual DPO v3 result is 90.47%, identical to v1/v2.

#### Version History

| Version | Files | Key Features | Result |
|---------|-------|-------------|--------|
| v1 | `build_dpo_pairs.py` `train_dpo.py` | 4 sources + synthetic, wrap_pair() with reasoning, MAX_LEN=512, epochs=3, beta=0.1 | 90.47% (regression) |
| v2 | train_dpo_v2.py | Same approach, separate copy | 90.47% |
| v3 | Updated train_dpo.py | MAX_LEN=1024, epochs=2, beta=0.2, removed wrap_pair() | 91.04% (identical to SFT) |
| v4 | `build_dpo_pairs_v4.py` `train_dpo_v4.py` | Real error mining (86% real pairs), beta=0.15, LR=1e-5, epochs=3 | **Training in progress** |

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

### 2.3. DPO v4 — Real Error Preference Optimization

**Key Innovation:** Instead of synthetic corruptions, mine actual model inference errors as rejected pairs.

| Aspect | v1-v3 | v4 |
|--------|-------|----|
| Source of rejected | 100% synthetic corruption | **86% real model errors** from eval |
| Real error pairs | 0 | 2,578 |
| Synthetic pairs | 2,000+ | 382 (14%, for diversity) |
| Validity pairs | 0 | 38 (invalid JSON → valid JSON) |
| Pair selection | Random | Stratified: 65% near-miss (0.70-0.95), 40% partial (0.40-0.70), 20% failures (<0.40) |
| Mean score of real pairs | N/A | 0.8040 |

#### Files Created

| File | Description |
|------|-------------|
| `build_dpo_pairs_v4.py` | Mines hard negatives from `eval_deepseek_result.xlsx`. 3 sources: real_error, validity, synthetic. |
| `train_dpo_v4.py` | Updated hyperparams: beta=0.15 (lower for harder pairs), LR=1e-5 (higher for informative data), epochs=3 |

#### Training Configuration (v4)

| Hyperparameter | v3 | v4 | Rationale |
|---------------|:--:|:--:|-----------|
| beta | 0.2 | 0.15 | Lower beta = more aggressive learning; real pairs need stronger signal |
| Learning rate | 5e-6 | 1e-5 | Real data is more informative, supports higher LR |
| Epochs | 2 | 3 | More diverse data needs more training |
| Effective batch | 16 | 16 | Same |
| Eval steps | 20 | 10 | More frequent monitoring |

#### Training Progress (Vast RTX 5090)

DPO v4 training started 2026-07-01 on `vast7` (RTX 5090, 32GB). Intermediate metrics from run:

| Step | Epoch | eval_loss | rewards/acc | rewards/margin | logps/chosen | logps/rejected | Note |
|:----:|:-----:|:---------:|:-----------:|:--------------:|:------------:|:--------------:|------|
| 5 | 0.03 | — | 33.8% | 0.006 | -63.63 | -55.09 | Model prefers rejected initially |
| 30 | 0.18 | 0.631 | 71.1% | 0.149 | -60.65 | -55.81 | Rapid improvement |
| 70 | 0.42 | 0.541 | 74.0% | 0.526 | -58.80 | -56.48 | Margins widening |
| 150 | 0.89 | 0.450 | 78.3% | 0.949 | -53.99 | -54.49 | **logps flip** — chosen > rejected |
| 230 | 1.36 | 0.435 | 77.6% | 1.281 | -54.14 | -56.85 | Gap = 2.71 nats |
| 310 | 1.84 | 0.413 | 79.3% | 1.466 | -57.82 | -61.77 | Gap = 3.95 nats |

Key observation: Model went from preferring rejected (logps/chosen -63.63 vs -55.09 at step 5) to strongly preferring chosen ( -57.82 vs -61.77 at step 310), a 5.8 nat swing.

#### Pair Quality Verification

Sample real_error pairs from `dpo_pairs_v4.jsonl`:

```
Prompt: Nút 'Tải lên' được tắt (disabled) khi không có tệp nào được chọn
Chosen:   {"action": "verify", "selector": "nút 'Tải lên'", "value": "", "expected": {"enabled": false}}
Rejected: {"action": "verify", "selector": "'Tải lên' button", "value": "", "expected": {"enabled": false}}
# Difference: selector translation nuance — model used English button name

Prompt: Verify the comparison table has title 'Product Comparison' with font size 32px
Chosen:   {"action": "verify", "selector": "comparison table title", "value": "", "expected": {"text-content": "Product Comparison", "font-size": "32px"}}
Rejected: {"action": "verify", "selector": "comparison table", "value": "", "expected": {"title": "Product Comparison", "font-size": "32px"}}
# Difference: model used wrong key "title" instead of "text-content"

Prompt: Kiểm tra tiêu đề modal 'Thông báo quan trọng' có độ đậm chữ 600 và màu #333333
Chosen:   {"action": "verify", ..., "expected": {"text-content": "Thông báo quan trọng", "font-weight": "600", "color": "#333333"}}
Rejected: {"action": "verify", ..., "expected": {"font-weight": "600", "color": "#333333"}}
# Difference: model dropped "text-content" key entirely
```

These are exactly the kind of subtle, real-world errors that DPO should learn from — wrong keys, missing properties, selector differences.


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

## 6. Next Steps

### ✅ Completed (v4)
- [x] `build_dpo_pairs_v4.py` — 2,578 real error pairs mined from eval (Phase 1.1, 1.2)
- [x] 38 validity pairs for invalid JSON (Phase 1.4)
- [x] `train_dpo_v4.py` — beta=0.15, LR=1e-5, epochs=3
- [x] Training running on Vast RTX 5090 (step 352/507 at last check)
- [x] Auto-eval queued to compare vs SFT baseline

### 🔜 Once Training Completes
1. Check eval result — if Pass rate > 91.04%, v4 is successful
2. Push adapter to HF Hub
3. Update report with final metrics

### 🔮 If v4 Doesn't Beat SFT
| Priority | Action | Expected Impact | Effort |
|:--------:|--------|----------------|--------|
| 1 | **Iterative DPO** (Phase 2.2) — train → sample → collect → retrain | High | 2d |
| 2 | **Add NLL term** (Phase 3.1) — prevent chosen probability collapse | Medium | 1d |
| 3 | **Grid search beta** (Phase 2.3) — try [0.05, 0.1, 0.3, 0.5] | Medium | 1d |
| 4 | **D-optimal selection** (Phase 1.3) — span complementary failure modes | Medium-High | 2d |
