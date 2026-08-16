# `defenses/` 索引

`defenses/` 目錄本身被 `.gitignore` 排除（跟 `runs/` 同待遇），這份索引留在版控裡做導覽用。完整方法論、動機、發現見 `CLAUDE.md` 附錄 C。

## 高斯噪訊防禦基準曲線（固定攻擊者 = E01 clean baseline）

用 `scripts/05_apply_defense.py --defense gaussian` 對 `runs/E01_baseline_clean_20260816_1302` 這個攻擊者掃 `sigma_ratio`：

| sigma_ratio | dir 前綴 | PSR | N_TGE | GE@9000 |
|---|---|---|---|---|
| 0.1 | `gaussian_sigma0.1_20260816_225331` | 0.0089 | 514 | 0.00 |
| 0.25 | `gaussian_sigma0.25_20260816_225339` | 0.0223 | 775 | 0.00 |
| 0.5 | `gaussian_sigma0.5_20260816_225035`（攻擊者=E01） | 0.0446 | 2276 | 0.00 |
| 0.5 | `gaussian_sigma0.5_20260816_224957`（攻擊者=E02，對照組） | 0.0446 | 230 | 0.00 |
| 0.75 | `gaussian_sigma0.75_20260816_225349` | 0.0668 | 5080 | 0.02 |
| 1.0 | `gaussian_sigma1_20260816_225358` | 0.0891 | 7019 | 0.15 |
| **1.5** | `gaussian_sigma1.5_20260816_225407` | **0.1337** | None | **1.81**（首次無法在9000條內攻破） |
| 2.0 | `gaussian_sigma2_20260816_225416` | 0.1783 | None | 5.61 |
| 3.0 | `gaussian_sigma3_20260816_225426` | 0.2674 | None | 43.68 |

**用途**：這是 GAN 防禦之後要打敗的第一條基準線。比較方式：在相同 PSR 成本下，GAN 防禦能不能把 GE@9000 壓得比這條曲線同一個PSR點更低（或用更低的PSR達到同等 GE@9000）。`sigma_ratio≈1.5`（PSR≈0.134）是曲線上第一個讓攻擊者九千條軌跡都攻不破的門檻，是一個自然的比較錨點。

同一批 sigma_ratio=0.5 的資料也對 E02（訓練時就用同款噪訊增強過的攻擊者）跑過一次，N_TGE 只退化到230（vs E01的2276）——證實攻擊者選擇本身是實驗設計的關鍵變數，後續一律固定用 E01 當基準攻擊者，見 CLAUDE.md 附錄 C.2。

## 重跑方式

```bash
python3 scripts/05_apply_defense.py --run runs/E01_baseline_clean_20260816_1302 \
  --defense gaussian --sigma-ratio 1.5
# 印出的下兩行指令直接複製貼上執行即可
```
