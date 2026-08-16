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

## 完整攻擊流程詳解

整條管線是 `00 → 01 → 02 → 03 → 04` 五個腳本依序執行，對應 CLAUDE.md §5.2 的六個攻擊階段（階段0-5）。每個腳本都是獨立可重跑的：只要上一階段的產物還在，可以單獨重跑任何一步。

### 階段 0：資料健檢（`00_inspect_data.py`）

在寫任何模型之前，先確認資料本身沒有誤解——這是全套流程裡最容易踩雷、也最容易被跳過的一步。

```bash
python3 scripts/00_inspect_data.py --h5 data/ASCAD.h5
```

會做五件事並印出結果：印出 A/V/D/E 四個切分集的 shape/dtype/數值範圍；從 `attack_meta['key'][0][2]` 讀出正確金鑰（絕不從論文抄）；掃描全部16個 mask 欄位算 SNR 峰值、自動選出遮罩已知標籤的最佳欄位；對照組計算未遮罩標籤的 SNR 峰值（應該接近0）；印出 desync 欄位分佈。最後有 PASS/FAIL 判定：**遮罩已知標籤的 SNR 峰值要顯著高於未遮罩對照組（預設門檻10倍），沒過這關後面全部白做**。

**desync50/desync100 資料庫要注意**：這兩個資料庫的自動 mask index 偵測會失敗（不是bug，是預期行為——時間抖動把單點SNR統計量稀釋到雜訊層級，見 CLAUDE.md 附錄 B.6），要顯式帶入從 `ASCAD.h5`（desync0）驗證過的 mask index：

```bash
python3 scripts/00_inspect_data.py --h5 data/ASCAD_desync50.h5 --mask-index 0
```

### 階段 1-3：訓練攻擊模型（`01_train_attacker.py`）

```bash
python3 scripts/01_train_attacker.py --config configs/exp/E01_baseline_clean.yaml
```

這一步做完 CLAUDE.md §5.2 的階段1-3：讀資料 → 四路切分（A/V/D/E，切分索引存進 `split_indices.npz` 確保可重現）→ 前處理（逐點標準化，只在 A 上 fit）→ 依洩漏模型算標籤 → 建模型 → 訓練（`GEModelSelection` callback 每 K 個 epoch 在 V 集上跑一次縮小規模的 GE 評估，取代不可靠的 val_loss 選 checkpoint）。

執行結果會建立 `runs/{exp_id}_{timestamp}_{pid}/`，裡面有：

| 檔案 | 內容 |
|---|---|
| `config_snapshot.yaml` | 這次執行完整展開後的組態（base+data+model+exp+override 全部合併） |
| `env.json` | Python/TF/numpy 版本、GPU型號、git commit hash |
| `model.keras` | GE-based selection 選出的最佳 checkpoint |
| `train_history.csv` | 每個 epoch 的 loss/accuracy/（有跑GE預覽時的）GE/N_TGE |
| `split_indices.npz` | A/V/D/E 的索引，重跑同一份 config 會得到逐位元一致的結果 |

**要注意的 config 欄位**（都在 `train:` 底下）：
- `optimizer` / `lr`：`cnn_light` 用 Adam，`cnn_best` **一定要用 RMSprop**（改 Adam 會發散，見陷阱清單#10）
- `lr_schedule: one_cycle`：搭配 `one_cycle.end_percentage` / `one_cycle.scale_percentage` 三個超參數，**注意實際峰值LR = `lr × 100 × scale_percentage²`**，只有 `scale_percentage=0.1` 時峰值才等於 `lr` 本身（這個公式踩過至少兩次坑，見附錄 B.15）
- `selection.eval_every` / `n_runs_val` / `patience`：GE-based 早停設定，`patience` 用「評估次數」不是「epoch數」

### 階段 4：對 Attack 集跑推論（`02_run_attack.py`）

```bash
python3 scripts/02_run_attack.py --run runs/E01_baseline_clean_20260816_1302
```

載入 `model.keras`，對 E 集（Attack 集，全程跟訓練隔離）跑推論，輸出 `probs.npy`（形狀 `(10000, 256)` 或 `(10000, 9)` 視洩漏模型而定）。這是攻擊階段跟評估階段唯一的介面——存檔後模型即可卸載，不用再碰 TensorFlow。**若 config 有開 `preprocess.resync`，這一步會用 A 集重新算一次對齊參考模板再拿去對齊 E**（resync 是確定性運算，不需要額外存檔，重算一次結果保證一致）。

若要測試「防禦後的波形」（Stage B 靜態攻擊者），只需要把不同的軌跡陣列餵進同一個已訓練模型重跑這一步，不用重新訓練。

### 階段 5：金鑰恢復與評估（`03_evaluate.py`）

```bash
python3 scripts/03_evaluate.py --run runs/E01_baseline_clean_20260816_1302
```

讀 `probs.npy`，對每條軌跡、每個金鑰假設算 log-likelihood 分數（**log 相加，不是機率連乘**——連乘在 N>50 時會下溢為0），跑 **100 次獨立重排攻擊**（每次隨機打亂 E 集取前 `attack.max_traces` 條累加分數算排名），輸出：

| 指標 | 意義 |
|---|---|
| `GE(N)` | 100次排名的平均，主指標 |
| `SR1(N)` | 排名=0（正確金鑰第一名）的比例 |
| `N_TGE` | 最小N使其後GE**全程**低於1（不是第一次觸底，是之後不再回頭） |
| `N_SR90` | 最小N使其後SR1≥0.9 |
| `PI` | 與N無關的資訊萃取量，`H[Z]+mean(log2 p(z\|t))` |
| 25/50/75百分位 | rank分佈，畫成陰影帶比只看平均誠實 |

**單次 attack run 的曲線必然劇烈震盪，那是抽樣雜訊不是結論**——這也是為什麼一定要跑滿100次獨立重排再下判斷；本專案的調查過程反覆證實，只看訓練期間少量run（例如GE-based selection內部用的20-run快速預覽）很容易被雜訊騙到，正式結論一律要用這裡的100-run正式評估確認（詳見 CLAUDE.md 附錄 B.24等多處假警報案例）。

**不用重跑訓練/推論就能重新評估**：

```bash
# 拉寬評估窗口看GE有沒有真的收斂
python3 scripts/03_evaluate.py --run runs/E01_baseline_clean_20260816_1302 --override attack.max_traces=9000
```

**注意：`metrics.json` 會被每次 `03_evaluate.py` 執行直接覆寫**，不會自動保留舊版本——如果只是想「順便看一下」不同窗口/次數的結果、又不想弄丟原本已經記錄在報告裡的正式數字，記得先手動備份 `metrics.json`，或是把 `--override` 的值跟原始 `config_snapshot.yaml` 的 `attack:` 區塊對一下，跑完記得用正確設定重跑一次才收工。

### 階段收尾：跨實驗比較報告（`04_make_report.py`）

```bash
python3 scripts/04_make_report.py
```

掃描 `runs/` 底下所有含 `metrics.json` 的目錄，幫每個 run 產生 `figures/ge_curve.png`（GE隨N變化，含25/75百分位陰影帶）跟 `figures/sr_curve.png`，並把所有 run 的關鍵指標彙整成 `reports/comparison.md` / `reports/comparison.tex`，可以直接貼進期末報告。

### 平行跑多組超參數（GPU 上很划算）

單張 GPU 上 `cnn_light`/`resnet` 的 epoch 成本很低，值得同時用兩張 GPU 跑不同超參數：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/01_train_attacker.py --config configs/exp/E03_desync50.yaml \
  --override preprocess.resync.enabled=true train.lr=1.0e-3 --runs-dir runs_sweep &
CUDA_VISIBLE_DEVICES=1 python3 scripts/01_train_attacker.py --config configs/exp/E03_desync50.yaml \
  --override preprocess.resync.enabled=true train.lr=1.0e-2 --runs-dir runs_sweep &
wait
```

**注意**：`run_dir` 命名格式是 `{exp_id}_{timestamp}_{pid}`（含 PID 保證唯一），如果兩個平行行程剛好在同一秒啟動，只靠時間戳仍會撞名——這個坑踩過兩次才修好（CLAUDE.md 附錄 B.35/B.37），現在已經用 PID 徹底解決，不會再發生。用不同的 `--runs-dir` 也可以進一步隔離不同批次的掃描結果，方便事後整理。

## 如何運用程式碼：常見擴充情境

專案設計原則是「組態驅動，程式碼零修改」——大部分新實驗只需要寫新的 YAML，不用碰 `.py`。

**跑一個全新的超參數組合**：不用建新檔案，直接 `--override`（見上面「單一超參數想微調」段落）。

**新增一個正式的實驗編號**：在 `configs/exp/` 底下新建 YAML，`data`/`model` 欄位指向 `configs/data/`、`configs/model/` 底下的檔案（或直接內嵌 dict），其餘欄位覆寫 `configs/base.yaml` 的預設值即可，範例可以直接抄 `configs/exp/E01_baseline_clean.yaml`。

**新增一個洩漏模型**：在 `src/data/labels.py::build()` 裡加一個分支（輸入 metadata + target_byte，輸出 `(N,) int` 標籤陣列），同時要在 `src/attack/scores.py::build()` 對應加上金鑰假設要映射到 probs 哪一欄的邏輯（`ID`/`ID_MASKED`/`HW` 三種現有映射方式可以參考）——**這兩處一定要同步改**，只改標籤不改評分邏輯是本專案抓到過的真實 bug（見附錄 B.17/B.19，訓練對了、評分錯了，GE 永遠不會收斂）。

**新增一個模型架構**：在 `src/models/` 底下新增檔案，用 `@register("your_model_name")` 裝飾一個 `build(input_dim, n_classes) -> keras.Model` 函式（參考 `src/models/cnn_light.py`），再建一個對應的 `configs/model/*.yaml`，`train.py` 不用改一行。

**新增一個評估指標**：在 `src/metrics/leakage.py`（軌跡層級，如 `snr`/`nicv`/`t_test`）或 `src/metrics/information.py`（機率分佈層級，如 `pi`/`mi`）裡新增一個純函式，輸入輸出都是 numpy array/float，不碰檔案系統。純函式的好處是不用改動任何既有程式碼就能單獨測試、單獨呼叫。

**處理新的時間抖動資料集**：`src/data/resync.py::resync_iterative()` 是通用的互相關盲對齊，`preprocess.resync.max_shift` 設成資料集的實際抖動範圍即可套用；**注意這個 max_shift 越大，對齊分數一定要用正規化互相關（目前實作已經是），未正規化的版本在抖動範圍較大時會系統性失效**（見附錄 B.33，desync100 曾經因為這個 bug 讓對齊近乎完全失敗）。

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
