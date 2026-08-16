# dlsca-attack-v2

基於深度學習的旁通道分析（Deep Learning Side-Channel Analysis, DLSCA）攻擊端管線，針對 ASCAD 資料集實作 profiled attack。本專案是「基於生成對抗網路之主動式對抗旁通道防禦機制」專題的攻擊端子系統（防禦端另案 `dlsca-defense-v2`，本 repo 為其預留輸入介面）。

完整方法論、威脅模型定義、逐項實驗紀錄與調查過程見 [`CLAUDE.md`](./CLAUDE.md)；所有跑過的實驗索引見 [`docs/runs.md`](./docs/runs.md)。這份 README 只涵蓋「怎麼裝、怎麼跑」。

## 威脅模型

Profiled attack：攻擊者擁有一台可完全控制的參考裝置（已知金鑰、已知明文）用於訓練攻擊模型，對目標裝置只能採集少量軌跡（已知明文、不知金鑰）試圖恢復金鑰。細節見 CLAUDE.md §5.1。

## 專案結構

```
configs/          # YAML 組態（base → data → model → exp 逐層合併，CLI 可再覆寫）
src/
  config.py       # 組態載入/合併/驗證
  seeding.py       # 全域種子控制
  data/           # 讀檔、標籤（洩漏模型）、切分、前處理/增強
  models/         # cnn_light / cnn_best / resnet
  train/          # 訓練迴圈、GE-based model selection、One-Cycle LR
  attack/         # 預測、log-likelihood 分數、Key Rank 評估
  metrics/        # SNR / PI 等評估指標
  report/         # 圖表與表格產生器
scripts/          # 00_inspect_data → 01_train_attacker → 02_run_attack → 03_evaluate → 04_make_report
tests/            # pytest 護欄（含極端案例）
runs/             # 執行產物（git-ignored，見下方「執行結果去哪找」）
docs/runs.md      # runs/ 目錄索引（有進版控）
ASCAD/            # ANSSI/CEA 上游參考實作（vendored third-party，BSD 授權，唯讀）
```

## 環境需求與安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

若要用 GPU 訓練，`requirements.txt` 裡的 `tensorflow==2.21.0` 本身不含 CUDA/cuDNN，需改裝：

```bash
pip install 'tensorflow[and-cuda]==2.21.0'
```

pip 裝的 NVIDIA 函式庫不在系統預設的動態連結器搜尋路徑上，需要手動補 `LD_LIBRARY_PATH`（可以寫進 `.venv/bin/activate` 讓每次 `source` 自動生效）：

```bash
SITE=$(.venv/bin/python3 -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$(find "$SITE/nvidia" -maxdepth 2 -type d -name lib | paste -sd: -):$LD_LIBRARY_PATH"
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### 資料

`data/*.h5`（ASCAD.h5、ASCAD_desync50.h5、ASCAD_desync100.h5）不隨此 repo 附上（被 `.gitignore` 排除）。原始資料下載連結見 `ASCAD/ATMEGA_AES_v1/*/Readme.md`。若已有處理好的三個 `.h5` 檔，直接放進 `data/` 目錄即可（並可用 SHA-256 對照官方雜湊值確認完整性）。

## 快速開始

```bash
# 0. 資料健檢（必跑一次，確認 mask index、正確金鑰、SNR 峰值都對）
python3 scripts/00_inspect_data.py --data data/ASCAD.h5

# 1. 訓練攻擊模型
python3 scripts/01_train_attacker.py --config configs/exp/E01_baseline_clean.yaml

# 2. 對 Attack 集跑推論，輸出 probs.npy
python3 scripts/02_run_attack.py --run runs/E01_baseline_clean_<timestamp>

# 3. 金鑰恢復評估（100 次獨立重排攻擊，計算 GE / SR1 / N_TGE / PI）
python3 scripts/03_evaluate.py --run runs/E01_baseline_clean_<timestamp>

# 4. 掃描 runs/ 下所有 metrics.json，產生跨實驗比較表
python3 scripts/04_make_report.py
```

單一超參數想微調不必修改 YAML，直接：

```bash
python3 scripts/01_train_attacker.py --config configs/exp/E01_baseline_clean.yaml \
  --override train.epochs=20 augment.gaussian.enabled=true
```

`03_evaluate.py` 也支援 `--override`，可以不重跑訓練/推論、只用不同的 `attack.max_traces` / `n_runs` / `seed` 重新評估同一份 `probs.npy`。

## 測試

```bash
python3 -m pytest tests/ -q
```

兩個護欄極端案例（完美預測 GE 應在 1 條軌跡內歸零、均勻分佈 GE 應維持在 127.5 附近）是整套評估邏輯最基本的正確性防線。

## 目前結果摘要

完整過程與陷阱記錄在 CLAUDE.md 附錄 B，這裡只列目前最重要的結論：

| 實驗 | 洩漏模型 / 資料 | N_TGE | 備註 |
|---|---|---|---|
| **E02**（噪訊增強） | ID, desync0 | **206** | **目前全專案最佳真實攻擊結果**，PI 由負轉正 |
| E01（clean baseline） | ID, desync0 | 475 | 唯一另一組已驗證收斂的真實攻擊結果 |
| E05（HW 洩漏模型） | HW, desync0 | 1361 | 9 類，收斂但比 ID 慢 |
| E08（遮罩已知標籤） | ID_MASKED, desync0 | 3 | 評估者視角上界，非真實攻擊能力 |
| E03（desync50，含 resync 前處理） | ID | GE@10000=12.00（結案，未超越E02） | 訓練前先用互相關盲對齊校正時間抖動，投入四維度超參數調查（跟desync100同等規模）後大幅推進，但仍未達成N_TGE，見附錄 B.29-B.31、B.51-B.54 |
| E04（desync100，含 resync 前處理） | ID | GE@10000=7.00（已結案，接受為誠實最佳結果） | 需要專屬四維度超參數調查，Attack 集 10000 條軌跡的評估窗口上限可能是瓶頸，見附錄 B.33-B.42 |
| E06（cnn_best） | ID, desync0 | 未收斂 | 6次系統性嘗試（LR schedule/峰值/batch size）全數收斂到同一負面區間，判定為模型規模（66.6M參數）本身的優化困難，結案，見附錄 B.43-B.50 |
| E07（resnet） | ID, desync0 | 未收斂，GE@9000=51.32 | 異架構對照組，四維度超參數調查已結案（局部最優），架構本身收斂較慢，見附錄 B.45-B.49 |

**方法論重點**：Key Rank 一律用 100 次獨立重排攻擊平均（單次攻擊曲線震盪劇烈，不可直接下結論）；模型選擇用 GE-based（非 val_loss，SCA 分類準確率恆在隨機基準附近，跟攻擊效能關聯薄弱）；所有評估指標與訓練迴圈完全解耦，`probs.npy` 存檔後模型即可卸載。

## 授權

`ASCAD/` 為 ANSSI/CEA 上游原始碼，BSD 授權（見 `ASCAD/LICENSE`），唯讀、不修改。其餘程式碼為本專題自行撰寫。
