# 結果圖總表

由 `scripts/09_make_figures.py` 一次產出（`python3 scripts/09_make_figures.py --with-snr`）。
每張圖的每一條曲線與每一個數字都是從 `runs*/metrics.json`、`runs*/train_history.csv`
或 `defenses*/cost_metrics.json` 讀回來的，不是從 CLAUDE.md 手抄，所以圖不會跟產生它的
那次執行脫節。兩個例外已在圖上與腳本 docstring 標明：F05 與 F12 左圖是從 `.h5` 重算 SNR
（SNR 是資料的性質，不屬於任何一次 run）；F11 右圖的四個 SNR 峰值是 附錄 B.66/B.67 的量測值
（byte 3 的資料庫在 GPU server 上，不在本機 `data/`）。

| 圖 | 內容 | 對應附錄 |
|---|---|---|
| F01 | E01 從無法收斂到 N_TGE=475 的四個方法論修正 | B.7–B.15 |
| F02 | one-cycle 三個維度各自單獨掃描（以 N_TGE 比較，非 GE） | B.13–B.15 |
| F03 | 洩漏模型決定難度：ID / HW / ID_MASKED | B.17、B.19 |
| F04 | 噪訊增強：收斂快 2.3 倍，且 PI 由負轉正 | B.23 |
| F05 | desync 把訊號打散而非消滅；正規化互相關把它找回來 | B.29、B.30、B.33 |
| F06 | desync50／desync100 各自的四維度掃描 | B.38–B.42、B.51–B.54 |
| F07 | 三個 desync 等級各自調參後的最佳曲線 | B.42、B.54 |
| F08 | cnn_best 六次嘗試全部落在同一個雜訊帶內 | B.50 |
| F09 | resnet 十個種子＋patience 對照＋1-epoch 對照 | B.57–B.59 |
| F10 | variable-key：管線正確，ID 任務本身學不起來 | B.64、B.65 |
| F11 | byte 2 vs byte 3：ID 攻擊是二階，視窗要同時裝下遮罩值與遮罩 | B.66、B.67 |
| F12 | E11 ChipWhisperer 基準：未防護實作單條軌跡即破 | B.68 |
| F13 | 硬體上的防禦劑量反應曲線，以及 PSR 作為成本指標的失效 | B.68、B.69 |
| F14 | 高斯噪訊防禦基準線（GAN 防禦要打敗的對象） | C.3 |

每一次 run 自己的 GE／SR 曲線由 `scripts/04_make_report.py` 寫進該 run 的 `figures/`，
跟著 `runs/` 一起被 `.gitignore` 排除，不在這裡。
