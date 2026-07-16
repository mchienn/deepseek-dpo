# Review toÃ n bá»™ quÃ¡ trÃ¬nh SFT â†’ DPO v4

NgÃ y review: 2026-07-16
Pháº¡m vi: code hiá»‡n cÃ³, checkpoint/log cá»¥c bá»™ vÃ  hai workbook Ä‘Ã¡nh giÃ¡ 9.427 máº«u.

## Káº¿t luáº­n Ä‘iá»u hÃ nh

DPO v4 Ä‘i Ä‘Ãºng hÆ°á»›ng á»Ÿ cháº¥t lÆ°á»£ng negative: 2.578/2.998 preference pairs (86,0%) lÃ  lá»—i sinh thá»±c táº¿, thay vÃ¬ pháº§n lá»›n lÃ  corruption nhÃ¢n táº¡o nhÆ° v1â€“v3. Tuy nhiÃªn, DPO v4 chÆ°a tá»‘t hÆ¡n SFT theo cÃ¡c chá»‰ sá»‘ tá»•ng há»£p Ä‘ang lÆ°u: pass rate giáº£m 0,276 Ä‘iá»ƒm pháº§n trÄƒm vÃ  AVG Score giáº£m 0,000355. NÃ³ giáº£m 17 output JSON invalid, nhÆ°ng Ä‘Ã¡nh Ä‘á»•i báº±ng Ä‘á»™ chÃ­nh xÃ¡c giÃ¡ trá»‹ CSS/thuá»™c tÃ­nh vÃ  sá»‘ output hoÃ n háº£o.

Æ¯u tiÃªn v5 khÃ´ng pháº£i chuyá»ƒn ngay sang GRPO/PPO. Cáº§n cÃ´ láº­p táº­p test, Ä‘Ã³ng gÃ³i provenance cá»§a tá»«ng run vÃ  lÃ m ablation cÃ³ kiá»ƒm soÃ¡t. Preference pairs hiá»‡n Ä‘Æ°á»£c khai thÃ¡c tá»« chÃ­nh táº­p dÃ¹ng Ä‘á»ƒ Ä‘Ã¡nh giÃ¡, nÃªn chÆ°a chá»©ng minh Ä‘Æ°á»£c generalization.

## Sá»‘ liá»‡u Ä‘á»‘i chiáº¿u trá»±c tiáº¿p

Nguá»“n: `eval_deepseek_result.xlsx` vÃ  `eval_dpo_v4_result.xlsx`, má»—i file 9.427 dÃ²ng.

| Metric | SFT | DPO v4 | Delta DPO v4 âˆ’ SFT |
|---|---:|---:|---:|
| Pass rate (score â‰¥ 0,70) | 91,0364% | 90,7606% | -0,2758 Ä‘iá»ƒm % |
| AVG Score | 0,899810 | 0,899455 | -0,000355 |
| Invalid JSON | 38 | 21 | **-17** |
| Mean Key Score | 0,930487 | 0,931796 | +0,001309 |
| Mean Value Score | 0,869133 | 0,867114 | -0,002019 |
| Score = 1,0 | 3.498 | 3.131 | -367 |

`eval_v6.log` lÃ  láº§n cháº¡y khÃ¡c trÃªn 10.050 máº«u (pass 91,0%, AVG 0,9005); khÃ´ng dÃ¹ng Ä‘á»ƒ suy ra delta DPO v4 vÃ¬ khÃ¡c táº­p dá»¯ liá»‡u vÃ  hiá»‡n output file mang tÃªn baseline.

## Review theo giai Ä‘oáº¡n

### 1. SFT / QLoRA

`train_qlora.py` cÃ³ ná»n táº£ng há»£p lÃ½: split 90/10, completion-only loss cho `<think> + JSON`, LoRA rank 32 trÃªn báº£y projection, bf16, cosine LR vÃ  early stopping. `train.log` cho tháº¥y 7 epoch hoÃ n táº¥t; best checkpoint step 450, eval loss 0,0672.

Äiá»ƒm yáº¿u lÃ  provenance: source hiá»‡n ghi output `qlora_checkpoints2`/`final_adapter2`, cÃ²n log vÃ  DPO dÃ¹ng `qlora_checkpoints`/`final_adapter`. Chá»‰ source hiá»‡n táº¡i khÃ´ng tÃ¡i táº¡o Ä‘Æ°á»£c chÃ­nh xÃ¡c baseline adapter.

### 2. DPO v1â€“v3

CÃ¡c báº£n cÅ© chá»§ yáº¿u dÃ¹ng synthetic corruption. Káº¿t quáº£ lá»‹ch sá»­ giáº£m tá»« SFT 91,04% xuá»‘ng khoáº£ng 90,47%; Ä‘iá»u nÃ y phÃ¹ há»£p vá»›i giáº£ thuyáº¿t negative dá»…/khÃ¡c phÃ¢n phá»‘i chá»‰ dáº¡y model nÃ© pattern bá» máº·t. CÃ¡c sá»­a lá»—i Ä‘á»™ dÃ i completion, loading adapter vÃ  dtype lÃ  cáº§n thiáº¿t, nhÆ°ng há»‡ thá»‘ng Ä‘áº·t tÃªn v1/v2/v3 khiáº¿n má»™t sá»‘ káº¿t quáº£ tá»«ng bá»‹ Ä‘á»c nháº§m lÃ  SFT. KhÃ´ng nÃªn dÃ¹ng chÃºng lÃ m baseline thá»­ nghiá»‡m má»›i.

### 3. DPO v4

`build_dpo_pairs_v4.py` mine lá»—i ba táº§ng: near-miss (0,70â€“0,95), partial (0,40â€“0,70), fail (<0,40); cÃ³ 38 validity pairs vÃ  chá»‰ 382 synthetic pairs. `train_dpo_v4.py` fine-tune tá»« SFT adapter vá»›i reference SFT, beta 0,15, LR 1e-5, batch hiá»‡u dá»¥ng 16 vÃ  3 epoch. ÄÃ¢y lÃ  thay Ä‘á»•i Ä‘Ãºng nháº¥t trong toÃ n bá»™ chuá»—i DPO.

## CÃ¡c Ä‘iá»ƒm cáº§n cáº£i thiá»‡n

1. **P0 â€” RÃ² rá»‰ evaluation.** v4 mine `eval_deepseek_result.xlsx`, rá»“i Ä‘Ã¡nh giÃ¡ `eval_dpo_v4_result.xlsx` trÃªn cÃ¹ng 9.427 máº«u. Gold `Step Object` cá»§a táº­p Ä‘Ã¡nh giÃ¡ Ä‘i vÃ o chosen DPO. Sá»‘ v4 chá»‰ lÃ  diagnostic, khÃ´ng pháº£i generalization.
2. **P0 â€” Provenance adapter thiáº¿u.** `dpo_adapter_v4`/checkpoint v4 khÃ´ng cÃ³ trong workspace; evaluator láº¡i cÃ³ adapter máº·c Ä‘á»‹nh khÃ¡c nhau (`dpo_adapter`, `dpo_adapter_v2`). KhÃ´ng thá»ƒ rerun cÃ¹ng weights.
3. **P0 â€” Auto-eval sai giao diá»‡n.** `_auto_eval_v4.sh` truyá»n `--adapter`/`--output`, nhÆ°ng `eval_deepseek_v2.py` khÃ´ng parse chÃºng; script cÃ³ thá»ƒ Ä‘Ã¡nh giÃ¡ nháº§m adapter vÃ  ghi nháº§m file.
4. **P0 â€” Resume checkpoint láº«n model.** JSONL inference checkpoint dÃ¹ng tÃªn chung. Náº¿u khÃ´ng version theo run ID, model má»›i cÃ³ thá»ƒ tÃ¡i sá»­ dá»¥ng generation cÅ©.
5. **P1 â€” Máº¥t khá»›p Ä‘á»‹nh dáº¡ng.** SFT há»c reasoning + JSON; chosen/rejected v4 lÃ  plain JSON, trong khi inference váº«n táº¡o reasoning trÆ°á»›c JSON. Cáº§n completion thá»‘ng nháº¥t hoáº·c NLL neo vÃ o completion SFT chuáº©n.
6. **P1 â€” ThÃ­ nghiá»‡m quÃ¡ Ã­t.** Chá»‰ má»™t cáº¥u hÃ¬nh, khÃ´ng láº·p seed, khÃ´ng ablation beta/LR/tá»· lá»‡ pair nguá»“n. Delta AVG hiá»‡n quÃ¡ nhá» Ä‘á»ƒ káº¿t luáº­n.
7. **P1 â€” Reward/scorer lÃ  proxy.** Semantic greedy matching (ngÆ°á»¡ng 0,7) há»¯u Ã­ch nhÆ°ng chÆ°a Ä‘á»§; cáº§n action/key/value exact metrics Ä‘á»ƒ biáº¿t lá»—i nÃ o thá»±c sá»± Ä‘Æ°á»£c sá»­a.

## Nháº­n Ä‘á»‹nh vá» RL

ChÆ°a nÃªn chuyá»ƒn ngay sang GRPO/PPO. Reward hiá»‡n lÃ  heuristic semantic vÃ  chÆ°a cÃ³ validation/test Ä‘á»™c láº­p; online RL sáº½ tá»‘i Æ°u proxy, tá»‘n compute vÃ  khÃ³ debug hÆ¡n. Vá»›i JSON cÃ³ gold rÃµ rÃ ng, hÆ°á»›ng phÃ¹ há»£p nháº¥t lÃ  **DPO cÃ³ neo SFT (DPO+NLL hoáº·c tÆ°Æ¡ng Ä‘Æ°Æ¡ng)** sau khi lÃ m sáº¡ch split/pairs. Chá»‰ cÃ¢n nháº¯c GRPO khi v5 Ä‘Ã£ cÃ³ evaluation sáº¡ch, verifier gáº§n-exact vÃ  DPO+NLL bÃ£o hoÃ .

## TiÃªu chÃ­ thÃ nh cÃ´ng v5

- Final test báº¥t biáº¿n, chÆ°a xuáº¥t hiá»‡n trong SFT/DPO mining.
- Trung bÃ¬nh â‰¥3 seed: pass rate â‰¥ SFT + 0,5 Ä‘iá»ƒm %, AVG â‰¥ SFT + 0,003.
- Invalid JSON khÃ´ng tÄƒng; action/key exact metrics khÃ´ng giáº£m.
- Má»—i result cÃ³ manifest: Git SHA, adapter SHA-256, dataset SHA-256, config, seed, checkpoint vÃ  inference checkpoint riÃªng.
