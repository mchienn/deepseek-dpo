# Kế hoạch DPO v5: tái lập được, không leakage, ưu tiên generalization

## Mục tiêu

Xác định preference optimization có cải thiện generalization so với SFT hay không. V5 không tối ưu trên test hiện hữu; nó là một thí nghiệm có thể rerun và ra quyết định dựa trên nhiều seed.

## Pha 0 — Đóng băng thí nghiệm (P0)

1. Tạo manifest SFT: Git SHA, SHA-256 adapter/workbook, model ID, tokenizer, seed và generation config.
2. Đặt run ID duy nhất cho adapter, checkpoints, JSONL inference checkpoint và workbook output.
3. Sửa evaluator nhận `--adapter`, `--output`, `--checkpoint`, `--test-file`, `--run-id`; fail fast nếu không hợp lệ.
4. Chuẩn hoá `eval_config.json` (threshold, generation tokens, batch, scorer version) và ghi cạnh result.

**Gate:** rerun cùng adapter tạo đủ 100% mẫu; manifest xác định duy nhất weights/dataset/config.

## Pha 1 — Chia dữ liệu đúng (P0)

1. Tạo split cố định theo hash prompt: SFT-train, DPO-mine, DPO-validation, final-test (có thể 80/10/10).
2. Final-test không xuất hiện trong fitting, error mining, pair building, hyperparameter selection hoặc prompt dedup gần giống.
3. Mine lỗi SFT chỉ ở DPO-mine; chọn hyperparameter ở DPO-validation; final-test chỉ mở cho candidate đã chốt.
4. Lưu ID/prompt hash từng split và kiểm tra giao nhau bằng script.

**Gate:** overlap final-test với mọi dữ liệu train/mining bằng 0; report có số mẫu và action distribution theo split.

## Pha 2 — Baseline và metric (P0)

1. Chạy SFT baseline 3 seed trên final-test, cùng generation config.
2. Giữ score hiện tại, bổ sung valid JSON, action exact, selector/key exact/F1, expected-property exact/F1, value exact theo type, output length và rate thiếu `</think>`.
3. Báo cáo paired delta theo prompt, bootstrap 95% CI và breakdown theo action/error family.

**Gate:** scorer có unit tests parse JSON, CSS shorthand và matching; baseline có manifest.

## Pha 3 — Preference data v5 (P1)

1. Mine real-error từ DPO-mine, deduplicate prompt/chosen/rejected và loại output bằng gold sau canonicalization.
2. Mix khởi điểm: 70% near-miss, 20% partial, 10% invalid/structural; synthetic ≤10% và chỉ khi chứng minh thiếu coverage.
3. Gắn taxonomy: invalid JSON, action, selector, missing/wrong key, shorthand, numeric/color/bool, reasoning-format.
4. Manual audit ngẫu nhiên 100 pairs; lưu audit CSV và pair source.
5. Dùng completion nhất quán với inference, hoặc giữ JSON thuần nhưng thêm NLL trên completion SFT chuẩn.

**Gate:** chosen JSON valid 100%; rejected khác chosen; lỗi audit nghiêm trọng <2%; không có prompt final-test.

## Pha 4 — Ma trận DPO tối thiểu (P1)

Mỗi candidate chạy 3 seed, cùng split và budget:

| Candidate | Objective | Beta | LR | Mục đích |
|---|---|---:|---:|---|
| A | SFT baseline | — | — | control |
| B | DPO v4-style đã làm sạch | 0,10 | 5e-6 | đo dữ liệu/split |
| C | DPO v4-style đã làm sạch | 0,15 | 5e-6 | so beta |
| D | DPO + NLL chosen | 0,10/0,15 | 5e-6 | chống forgetting/collapse |

- Batch hiệu dụng 32 nếu VRAM cho phép, không thì giữ 16 và ghi manifest.
- Early stop theo DPO-validation, không theo final-test.
- Final-test chỉ đánh giá checkpoint tốt nhất từng seed.

**Decision rule:** chọn candidate khi mean pass tăng ≥0,5 điểm %, mean AVG tăng ≥0,003, paired CI không cho thấy regression lớn và invalid JSON không tăng. Nếu không candidate qua gate, dừng DPO để ưu tiên error-targeted SFT.

## Pha 5 — Sau khi v5 pass (P2)

1. Iterative DPO trên mining pool mới, không chạm final-test.
2. Thử diverse/D-optimal negative selection theo taxonomy.
3. Chỉ cân nhắc GRPO khi có reward verifier gần-exact cho JSON/CSS, reward-hacking checks và compute budget online sampling.

## Deliverables

- `runs/<run-id>/manifest.json`, config, adapter hash, metrics và result workbook.
- Script split/pair-validation/eval có CLI và tests.
- Bảng so sánh SFT/DPO v5 theo seed + paired delta.
- Quyết định rõ: deploy, iterate DPO hoặc quay về SFT.
