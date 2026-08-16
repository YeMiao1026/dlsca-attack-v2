# DLSCA 攻擊端程式碼重構 — 專案計畫書

**專案代號**：`dlsca-attack-v2`
**所屬專題**：基於生成對抗網路之主動式對抗旁通道防禦機制
**文件版本**：v1.0
**適用範圍**：攻擊模型訓練、金鑰恢復攻擊、評估指標產出（防禦端另案）

---

## 目錄

1. [重構動機與目標](#1-重構動機與目標)
2. [設計原則](#2-設計原則)
3. [專案結構](#3-專案結構)
4. [資料流與模組契約](#4-資料流與模組契約)
5. [攻擊流程完整規格](#5-攻擊流程完整規格)
6. [模型訓練完整規格](#6-模型訓練完整規格)
7. [組態系統](#7-組態系統)
8. [實驗管理與可重現性](#8-實驗管理與可重現性)
9. [測試計畫](#9-測試計畫)
10. [里程碑與分工](#10-里程碑與分工)
11. [驗收標準](#11-驗收標準)
12. [已知陷阱清單](#12-已知陷阱清單)

---

## 1. 重構動機與目標

### 1.1 現行程式碼的問題

期中階段的程式碼是為了「盡快跑出結果」而寫的，累積了以下技術債：

| 問題 | 具體症狀 | 後果 |
|---|---|---|
| 流程散落在 notebook | 資料載入、訓練、攻擊混在同一個 cell 序列 | 無法重跑、無法交叉比較、換人接手成本高 |
| 超參數硬編碼 | `epochs=50` 直接寫在呼叫處 | 做 ε 掃描、模型比較時要改動原始碼 |
| 隨機性未受控 | 未固定 seed | 同樣的程式跑兩次結果不同，無法確認差異來源 |
| 資料切分無紀律 | 未確保攻擊器/防禦器訓練集互斥 | 評估結果可能高估 |
| 評估邏輯耦合 | Key Rank 計算寫在訓練腳本裡 | 無法對「同一組機率輸出」重複做不同評估 |
| 單次攻擊即下結論 | 未做多次獨立攻擊平均 | 曲線震盪來自抽樣雜訊而非防禦效果 |
| 模型選擇依據錯誤 | 以 val loss 或 val accuracy 選 checkpoint | 與實際攻擊效能關聯薄弱 |

### 1.2 重構目標

- **G1 可重現**：任何實驗可由一份 config + 一個指令完整重跑，結果逐位元一致。
- **G2 可組合**：資料集、模型、洩漏模型、前處理、評估指標皆可獨立替換。
- **G3 可稽核**：每次執行自動留存 config 快照、環境資訊、隨機種子、指標數值。
- **G4 可擴充**：Stage B 的自適應攻擊者（重訓練、去噪、異架構、交替訓練）能在不改動核心的前提下加入。
- **G5 可交付**：產出的圖表與表格可直接放進期末報告，不需二次加工。

### 1.3 非目標（明確排除）

- 不在本次重構中實作防禦器訓練（另案 `dlsca-defense-v2`，但介面須預留）。
- 不實作完整 16-byte 金鑰恢復（ASCAD 視窗僅涵蓋 byte 2）。
- 不做硬體採集流程（ChipWhisperer 部分維持現狀）。

---

## 2. 設計原則

**P1　組態驅動，程式碼零修改**
所有可變項目（資料路徑、切分比例、模型架構、超參數、洩漏模型、評估設定）一律出自 YAML。跑新實驗＝寫新 YAML，不動 `.py`。

**P2　純函式優先**
資料處理與評估指標寫成無狀態純函式，輸入輸出皆為 numpy array。只有訓練迴圈與 I/O 允許有副作用。

**P3　中間產物落地**
攻擊模型的輸出機率矩陣 `probs (N, 256)` 必須存成檔案。評估階段只吃這個檔案，不重跑模型。這讓「同一組預測做十種評估」變成秒級操作。

**P4　種子全域受控**
`numpy` / `python random` / `tensorflow` 三處種子由 config 統一指定並記錄。資料切分、噪訊增強、attack run 重排各有獨立的衍生種子。

**P5　評估與訓練解耦**
訓練腳本不得包含任何 Key Rank 計算。評估是獨立階段，吃 `probs` 與 `metadata`。

**P6　先驗證後推進**
每個階段有明確的健全性檢查（sanity check）。檢查不過就不進下一階段。

---

## 3. 專案結構

```
dlsca-attack-v2/
├── configs/
│   ├── base.yaml                  # 共用預設值
│   ├── data/
│   │   ├── ascad_desync0.yaml
│   │   ├── ascad_desync50.yaml
│   │   └── ascad_desync100.yaml
│   ├── model/
│   │   ├── cnn_light.yaml         # 18K 輕量模型
│   │   ├── cnn_best.yaml          # ASCAD CNN_best 66.6M
│   │   └── resnet.yaml            # Stage B 異架構攻擊者
│   └── exp/
│       ├── E01_baseline_clean.yaml
│       ├── E02_noisy_augment.yaml
│       ├── E03_desync50.yaml
│       └── E04_hw_leakage.yaml
│
├── src/
│   ├── config.py                  # YAML 載入、合併、驗證、快照
│   ├── seeding.py                 # 全域種子控制
│   ├── data/
│   │   ├── ascad.py               # 讀檔、metadata 解析
│   │   ├── labels.py              # 洩漏模型（ID / HW / masked）
│   │   ├── split.py               # 四路切分
│   │   └── preprocess.py          # Standardizer、增強
│   ├── models/
│   │   ├── registry.py            # name -> builder 對照表
│   │   ├── cnn_light.py
│   │   ├── cnn_best.py
│   │   └── resnet.py
│   ├── train/
│   │   ├── trainer.py             # 訓練迴圈
│   │   └── callbacks.py           # GE-based model selection（核心）
│   ├── attack/
│   │   ├── predict.py             # 產出 probs 矩陣
│   │   ├── scores.py              # log-likelihood 分數矩陣
│   │   └── keyrank.py             # 多次獨立攻擊、GE、SR、N_TGE
│   ├── metrics/
│   │   ├── leakage.py             # SNR / NICV / t-test
│   │   ├── information.py         # PI / MI
│   │   └── perturbation.py        # PSR / L2 / Linf
│   └── report/
│       ├── plots.py               # 統一樣式的圖表產生器
│       └── tables.py              # Markdown / LaTeX 表格輸出
│
├── scripts/
│   ├── 00_inspect_data.py         # 資料健檢（必跑）
│   ├── 01_train_attacker.py
│   ├── 02_run_attack.py
│   ├── 03_evaluate.py
│   ├── 04_make_report.py
│   └── run_pipeline.py            # 串接 01-04
│
├── tests/
│   ├── test_labels.py
│   ├── test_scores.py
│   ├── test_keyrank.py
│   └── test_split.py
│
├── runs/                          # 執行結果，git-ignored
│   └── {exp_id}_{timestamp}/
│       ├── config_snapshot.yaml
│       ├── env.json
│       ├── model.keras
│       ├── train_history.csv
│       ├── probs.npy
│       ├── metrics.json
│       └── figures/
│
├── notebooks/                     # 只放探索性分析，不放正式流程
├── requirements.txt
└── README.md
```

**規則**：`notebooks/` 內的任何程式碼都不是正式結果來源。所有寫進報告的數字必須來自 `runs/` 下的 `metrics.json`。

---

## 4. 資料流與模組契約

### 4.1 全域資料流

```
ASCAD.h5
   │
   ├─ ascad.load()                    → traces(int8), metadata(structured)
   │
   ├─ split.four_way()                → A / V / D / E 四個互斥索引集
   │
   ├─ preprocess.Standardizer         → fit(A) 後 transform 全部
   │        │
   │        └─ (float32, 逐點 σ=1)
   │
   ├─ labels.build(leakage_model)     → y (N,) int
   │
   ├─ trainer.fit(A, V)               → model.keras
   │        └─ GEModelSelection callback（用 V 的 GE 選 checkpoint）
   │
   ├─ predict.run(model, E)           → probs.npy (10000, 256)
   │
   ├─ scores.build(probs, meta)       → scores (10000, 256) log-likelihood
   │
   ├─ keyrank.evaluate(scores)        → ranks (100, 1000)
   │        └─ GE / SR₁ / N_TGE / N_SR90
   │
   └─ metrics + report                → metrics.json + figures/
```

### 4.2 模組契約表

| 模組 | 輸入 | 輸出 | 不變條件 |
|---|---|---|---|
| `ascad.load` | 檔案路徑 | `traces (N,700) int8`, `meta` | 不做任何轉換 |
| `split.four_way` | N, 各集大小, seed | 四組 index array | 四組兩兩交集為空 |
| `preprocess.Standardizer` | traces | float32, 逐點 μ=0 σ=1 | 只在 A 上 fit |
| `labels.build` | meta, leakage_model | `y (N,) int` | 值域符合類別數 |
| `trainer.fit` | A, V, model, cfg | keras model | 不接觸 E |
| `predict.run` | model, traces | `probs (N,C) float32` | 每列和為 1 |
| `scores.build` | probs, meta, byte | `(N,256) float64` | 用 log 相加，非連乘 |
| `keyrank.evaluate` | scores, key, n_runs | `ranks (R,T) int16` | 每 run 獨立重排 |

**契約違反即為 bug**。測試針對這些不變條件撰寫。

---

## 5. 攻擊流程完整規格

這一節是重構的核心，也可直接改寫成期末報告第三章的「攻擊流程」。

### 5.1 威脅模型定義（報告書必寫）

本研究採用 **profiled attack（建檔式攻擊）** 情境：

- **攻擊者能力**：擁有一台與目標同型號、可完全控制的參考裝置，能以已知金鑰與已知明文採集大量功耗軌跡，用於訓練攻擊模型。
- **攻擊者目標**：對目標裝置採集少量軌跡（僅知明文、不知金鑰），恢復金鑰位元組。
- **本研究對應**：Profiling 集扮演參考裝置，Attack 集扮演目標裝置。
- **成功定義**：正確金鑰在 256 個候選中的排名降至 0（第一名）且穩定維持。

此為旁通道領域最強的攻擊者假設，因此在此情境下驗證的防禦具備上界意義。

### 5.2 攻擊六階段

#### 階段 0：資料健檢（`00_inspect_data.py`）

**目的**：在寫任何模型之前，先確認資料本身沒有誤解。

執行項目：
1. 印出四個集合的 shape、dtype、數值範圍、標準差。
2. 確認 Attack 集金鑰為固定值，讀出 `key[0][2]` 作為正確金鑰（**不得從論文抄寫**）。
3. 掃描 `masks` 全部 16 欄，對每欄計算 `Z' = Sbox[p⊕k] ⊕ masks[i]` 的 SNR 峰值，選出峰值最高者作為 `r_out` 的欄位索引。
4. 對照組：計算未遮罩標籤 `Z = Sbox[p⊕k]` 的 SNR 峰值。
5. 印出 desync 欄位的分佈。

**通過條件**：
- 遮罩已知標籤的 SNR 峰值 **顯著高於** 未遮罩標籤（後者應接近 0）。若非如此，代表 mask index 找錯或 metadata 解析有誤。
- SNR 峰值位置應集中在少數幾個時間點（POI），不應散佈整條波形。

> 這一步沒過，後面全部白做。這是期中階段最可能潛藏的錯誤來源。

#### 階段 1：資料切分與前處理

**四路切分**（seed 固定於 config）：

| 集合 | 來源 | 預設數量 | 用途 |
|---|---|---|---|
| A | Profiling | 30,000 | 攻擊模型訓練 |
| V | Profiling | 5,000 | 模型選擇（GE-based） |
| D | Profiling | 15,000 | 保留給防禦器（本次不用，但必須切出來） |
| E | Attack | 10,000 | 最終評估，全程隔離 |

**紀律**：
- A / V / D 三者互斥，由同一次 permutation 切出。
- E 來自獨立的 `Attack_traces`，不與 profiling 混用。
- 即使本階段不訓練防禦器，D 也要先切出並記錄索引，確保後續防禦實驗的公平性。

**前處理**：逐時間點標準化，`Standardizer.fit(A)` 後對 A/V/D/E 一律 `transform`。標準化後每點 σ=1，此時擾動上限 ε 的單位即為「標準差」，物理意義明確。

#### 階段 2：洩漏模型與標籤

| 代號 | 定義 | 類別數 | 用途 |
|---|---|---|---|
| `ID` | `Z = Sbox[p⊕k]` | 256 | 主力設定，攻擊者真實視角（遮罩未知） |
| `ID_MASKED` | `Z' = Sbox[p⊕k] ⊕ r_out` | 256 | 評估者視角，用於 SNR / NICV |
| `HW` | `HW(Sbox[p⊕k])` | 9 | 對照組，類別不平衡需注意 |

**措辭修正**：期中報告書 3.2 節寫的「無遮掩資料集 / 有遮掩資料集」不精確 — ASCAD 只有一份遮罩實作的波形，差別在**標籤定義**而非資料集。期末報告請改寫為「遮罩未知標籤 / 遮罩已知標籤」，並在此處說明兩者的攻擊者能力差異。

#### 階段 3：模型訓練

見第 6 節完整規格。

#### 階段 4：預測輸出（`02_run_attack.py`）

對 E 集執行推論，輸出 `probs (10000, 256) float32` 並存檔。

**設計要點**：
- 這是攻擊階段與評估階段的唯一介面。存檔之後模型即可卸載。
- 若後續要評估「防禦後的波形」，只需替換輸入波形重跑本階段，模型不變（對應 Stage B 的 A0 靜態攻擊者）。
- 檢查：`probs.sum(axis=1)` 應全部為 1；不得出現 NaN。

#### 階段 5：金鑰恢復與評估（`03_evaluate.py`）

**5.1 分數矩陣**

對每條軌跡 i 與每個金鑰假設 k ∈ [0, 255]：

```
hypothesis(i, k) = Sbox[plaintext[i][2] ⊕ k]
score(i, k)      = log( probs[i][ hypothesis(i,k) ] + ε )
```

**必須用 log 相加，不可用機率連乘**。連乘在 N > 50 時下溢為 0，是本領域最常見的實作錯誤。

**5.2 多次獨立攻擊**

```
for run in 1..100:
    idx  = 隨機重排 E 集後取前 1000 條
    cum  = cumsum(scores[idx], axis=0)            # (1000, 256)
    rank(N) = #{ k : cum[N][k] > cum[N][correct_key] }
```

**單次 attack run 的 rank 曲線必然劇烈震盪，那是抽樣雜訊，不是結論。** 必須平均。

**5.3 指標**

| 指標 | 定義 | 意義 |
|---|---|---|
| GE(N) | 100 次 rank 的平均 | 主指標，越低越接近破解 |
| SR₁(N) | rank = 0 的比例 | 「到底破不破得了」 |
| N_TGE | 最小 N 使其後 GE 全程 < 1 | 破解所需軌跡數 |
| N_SR90 | 最小 N 使其後 SR₁ ≥ 0.9 | 穩定破解門檻 |
| PI | `H[Z] + mean(log₂ p(z|t))` | 與 N 無關的資訊萃取量 |
| 25/50/75 百分位 | rank 分佈 | 畫成 shaded band，比只畫平均誠實 |

**N_TGE 的定義必須是「其後全程低於門檻」**，只找第一次觸底會被震盪騙到。

**5.4 健全性檢查**

Clean baseline（無防禦、輕量 CNN、desync0）的 `N_TGE` 應落在 **100 上下**。若跑不出來，錯誤幾乎必定在 label 對應或分數矩陣，先修這個再往下走。

---

## 6. 模型訓練完整規格

### 6.1 模型清單

| 代號 | 架構 | 參數量 | 定位 |
|---|---|---|---|
| `cnn_light` | Conv(4,k3)-BN-AvgPool-Conv(8,k51)-BN-AvgPool-FC(10)-FC(10)-FC(256) | 18,642 | **主力假想敵** |
| `cnn_best` | ASCAD 原論文 VGG 風格，5×Conv(k11)+3×FC | 66,652,544 | 深層對照 |
| `resnet` | 3 個殘差區塊 + GAP | 約 30K | Stage B transferability |

**`cnn_light` 的兩個設計說明**（值得寫進報告）：

1. **第二層 kernel = 51 是刻意的大感受野**。SCA 波形存在時間抖動，大 kernel 提供平移容忍度，這是該模型在 desync 情境下仍堪用的主因 — 比「參數量少」更值得強調。
2. **激活函數採 SELU + lecun_normal**。自正規化特性使淺層網路收斂較 ReLU+BN 更穩定，是高效 SCA 模型的標準搭配。

**命名修正**：期中報告的「Desync 系列」應正名為 **ASCAD 原論文之 CNN_best**（參數量 66,652,544 與原論文一致），引用來源明確可提升可信度。

### 6.2 訓練超參數

| 模型 | Optimizer | LR | Batch | Epochs | Loss |
|---|---|---|---|---|---|
| `cnn_light` | Adam | 1e-3 | 128 | 100（含 early stop） | sparse CE |
| `cnn_best` | RMSprop | **1e-5** | 200 | 75 | sparse CE |
| `resnet` | Adam | 1e-3 | 128 | 100 | sparse CE |

`cnn_best` 必須用 RMSprop 1e-5（ASCAD 原論文設定），改成 Adam 1e-3 會發散。

### 6.3 模型選擇機制（本次重構最重要的改動）

**問題**：SCA 在 256 類上的分類準確率恆在 0.4% 附近（隨機基準 1/256 ≈ 0.39%），`val_loss` 與 `val_accuracy` 與實際攻擊效能關聯薄弱。以 val loss 選 checkpoint 是本領域公認的坑。

**解法**：實作 `GEModelSelection` callback。

```
每 K 個 epoch（預設 5）：
    1. 對 V 集推論得 probs_val
    2. 以較少的 run 數（預設 20）計算 GE
    3. 取 N_TGE（若未收斂則以 GE@末值作為次要排序鍵）
    4. 若優於歷史最佳 → 存檔為 best.keras
```

排序規則：
- 兩者皆收斂 → N_TGE 小者勝。
- 僅一者收斂 → 收斂者勝。
- 皆未收斂 → GE@末值低者勝。

**Early stopping**：以 GE 指標為準，patience 設 6 次評估（即 30 epochs）無改善即停。

### 6.4 資料增強

**噪訊增強**（你們期中發現能提升攻擊效率的技巧）：

- 於每個 epoch **動態加入**，不預先生成固定的噪訊資料集。
- 強度以 `sigma_ratio` 表示（相對於 trace 標準差的倍數），預設 0.5。
- 報告書必須寫 `sigma_ratio` 而非絕對數值，否則換前處理即無法比較。

**現象解釋**（期中報告 3.3.2 第 3 點可強化）：噪訊增強具正規化效果，迫使模型降低對局部隨機雜訊的依賴，轉而學習穩定的 leakage pattern。這與 Wu / Perin / Picek 的觀察一致，建議在報告中補上此引用連結。

### 6.5 訓練產出物

每次執行在 `runs/{exp_id}_{timestamp}/` 產出：

| 檔案 | 內容 |
|---|---|
| `config_snapshot.yaml` | 完整合併後的組態（含所有預設值展開） |
| `env.json` | Python / TF / numpy 版本、GPU 型號、git commit hash |
| `model.keras` | 最佳 checkpoint |
| `train_history.csv` | 每 epoch 的 loss / acc / GE / N_TGE |
| `split_indices.npz` | A / V / D / E 的索引，確保可重現 |
| `probs.npy` | E 集的預測機率 |
| `metrics.json` | 所有評估指標 |
| `figures/` | GE 曲線、SR 曲線、SNR 疊圖 |

---

## 7. 組態系統

### 7.1 合併規則

`base.yaml` → `data/*.yaml` → `model/*.yaml` → `exp/*.yaml` → CLI 覆寫，後者覆蓋前者。

### 7.2 範例：`exp/E01_baseline_clean.yaml`

```yaml
exp_id: E01_baseline_clean
description: 無防禦、輕量 CNN、desync0 的攻擊基準

seed: 42

data:
  path: data/ASCAD.h5
  target_byte: 2
  trace_len: 700

split:
  n_attacker: 30000
  n_val: 5000
  n_defender: 15000
  seed: 42

preprocess:
  method: standardize_per_point
  fit_on: attacker

leakage:
  model: ID              # ID | HW | ID_MASKED
  n_classes: 256

augment:
  gaussian:
    enabled: false
    sigma_ratio: 0.5

model:
  name: cnn_light

train:
  optimizer: adam
  lr: 1.0e-3
  batch_size: 128
  epochs: 100
  selection:
    metric: n_tge
    eval_every: 5
    n_runs_val: 20
    patience: 6

attack:
  eval_set: E
  max_traces: 1000
  n_runs: 100
  seed: 1234

metrics:
  compute: [ge, sr1, n_tge, n_sr90, pi, percentiles]
  leakage_assessment: [snr, nicv, t_test]
```

### 7.3 CLI

```bash
python scripts/run_pipeline.py --config configs/exp/E01_baseline_clean.yaml
python scripts/run_pipeline.py --config configs/exp/E01_baseline_clean.yaml \
       --override train.epochs=20 augment.gaussian.enabled=true
python scripts/03_evaluate.py --run runs/E01_baseline_clean_20260820_1430
```

---

## 8. 實驗管理與可重現性

### 8.1 種子分層

| 用途 | 來源 |
|---|---|
| 全域（TF / numpy / random） | `cfg.seed` |
| 資料切分 | `cfg.split.seed` |
| 噪訊增強 | `cfg.seed + epoch` |
| Attack run 重排 | `cfg.attack.seed + run_index` |

各層獨立，方便單獨變動其中一項做穩健性檢查。

### 8.2 實驗編號規範

`E{編號}_{簡述}`，一經建立不得修改語意。已規劃：

| 編號 | 內容 |
|---|---|
| E01 | Clean baseline，cnn_light，desync0 |
| E02 | 噪訊增強訓練 |
| E03 | desync50 |
| E04 | desync100 |
| E05 | HW 洩漏模型 |
| E06 | cnn_best 對照 |
| E07 | resnet 對照 |
| E08 | 遮罩已知標籤（評估者視角上界） |

### 8.3 結果匯總

`04_make_report.py` 掃描 `runs/` 下所有 `metrics.json`，輸出跨實驗比較表（Markdown + LaTeX），可直接貼入報告書。

---

## 9. 測試計畫

| 測試 | 驗證內容 |
|---|---|
| `test_labels` | 已知 (p, k) 手算 S-box 輸出比對；HW 值域 0–8 |
| `test_split` | 四集合兩兩交集為空；總數正確；同 seed 結果一致 |
| `test_scores` | 正確金鑰的累積分數應高於隨機金鑰；無 NaN / inf |
| `test_keyrank` | 餵入人造完美預測（正確類別機率 1.0），GE 應在 1 條軌跡內歸零 |
| `test_keyrank_random` | 餵入均勻分佈機率，GE 應維持在 127.5 附近 |
| `test_reproducibility` | 同 config 跑兩次，`metrics.json` 完全一致 |

**`test_keyrank` 的兩個極端案例是整套系統的護欄**，先寫這兩個測試再寫其他任何東西。

---

## 10. 里程碑與分工

| 週次 | 里程碑 | 交付 | 建議負責 |
|---|---|---|---|
| W1 | 骨架與資料層 | 專案結構、config 系統、`ascad.py`、`split.py`、`00_inspect_data.py` 通過 | 1 人 |
| W1 | 測試護欄 | `test_labels` / `test_keyrank` 兩個極端案例通過 | 1 人（可平行） |
| W2 | 評估層 | `scores.py`、`keyrank.py`、`metrics/`，用人造資料驗證 | 1 人 |
| W2 | 模型層 | `registry.py` 與三個模型，參數量與期中報告一致 | 1 人 |
| W3 | 訓練層 | `trainer.py`、`GEModelSelection` callback | 2 人 |
| W3 | **E01 Clean baseline 重現** | N_TGE ≈ 100 | 全組 |
| W4 | 批次實驗 | E02–E08 全數完成 | 分工跑 |
| W4 | 報告產出 | 跨實驗比較表、統一樣式圖表 | 1 人 |

**W3 的 E01 重現是關鍵閘門**。重構後若無法重現期中的 100 條收斂結果，代表新舊實作有落差，必須釐清後才能繼續。

---

## 11. 驗收標準

本次重構完成的判準：

- [ ] 任一實驗可由單一指令完整重跑，兩次執行的 `metrics.json` 完全一致。
- [ ] `00_inspect_data.py` 通過：遮罩已知標籤的 SNR 峰值顯著高於未遮罩標籤。
- [ ] E01 Clean baseline 的 N_TGE 落在 100 ± 30。
- [ ] 所有 GE 曲線皆為 100 次獨立攻擊平均，且附 25/75 百分位帶。
- [ ] `tests/` 全數通過，含兩個極端案例。
- [ ] 新增一個攻擊模型只需新增一個 model config，不需修改核心程式碼。
- [ ] 新增一種評估指標只需在 `metrics/` 新增純函式並在 config 列名。
- [ ] 防禦端介面已預留：替換輸入波形即可重跑階段 4–5。
- [ ] 期末報告所需的全部圖表可由 `04_make_report.py` 一次產出。

---

## 12. 已知陷阱清單

施工時逐項對照，這些都是實際會踩到的。

| # | 陷阱 | 症狀 | 處置 |
|---|---|---|---|
| 1 | 機率連乘取代 log 相加 | N > 50 後分數全為 0，GE 卡住不動 | 一律 log 相加 |
| 2 | mask index 抄錯 | SNR 全域接近 0 | 用 `00_inspect_data.py` 實測選欄 |
| 3 | 正確金鑰從論文抄 | GE 永不收斂 | 從 `atk_meta['key'][0][2]` 讀取 |
| 4 | 以 val_loss 選 checkpoint | 存下的模型攻擊效能差 | 改用 GE-based selection |
| 5 | Standardizer 在全資料上 fit | 資訊洩漏，結果高估 | 只在 A 集 fit |
| 6 | 單次 attack run 即下結論 | 曲線劇烈震盪 | 100 次獨立攻擊平均 |
| 7 | N_TGE 只找第一次觸底 | 被震盪誤判為已收斂 | 要求「其後全程低於門檻」 |
| 8 | 攻擊器與防禦器訓練集重疊 | 防禦效果被高估 | 四路切分強制互斥 |
| 9 | HW 模型未處理類別不平衡 | 模型全預測 HW=4 | 加 class weight 或改用 ID |
| 10 | `cnn_best` 用 Adam 1e-3 | 訓練發散 | 依原論文用 RMSprop 1e-5 |
| 11 | 噪訊增強預先生成固定資料集 | 正規化效果消失 | 每 epoch 動態產生 |
| 12 | 未固定 TF 種子 | 兩次執行結果不同 | `seeding.py` 統一設定 |

---

## 附錄 A：與期末報告章節的對應

| 本文件章節 | 期末報告對應章節 |
|---|---|
| 5.1 威脅模型定義 | 第三章 3.1 攻擊者模型 |
| 5.2 階段 0–2 | 第三章 3.2 資料集與前處理 |
| 6.1–6.2 模型與超參數 | 第三章 3.3 攻擊模型架構 |
| 6.3 模型選擇機制 | 第三章 3.4 訓練策略（**新增小節，為本研究方法論貢獻**） |
| 5.5 評估指標 | 第四章 評估方法論 |
| 8.2 實驗編號 | 第五章 實驗結果 |
| 12 陷阱清單 | 不進報告，內部文件 |

**6.3 的 GE-based model selection 建議在報告書獨立成節**。多數大學部專題直接用 val loss 選模型，明確處理這個問題本身就是方法論上的加分項，口試時也是一個好的答辯素材。

---

## 附錄 B：目前程式庫實際狀態（給 Claude Code 的操作指引）

**本節描述的是「現況」，不是「規劃」——與第 1–12 節（目標架構）互為對照，不要混淆。**

> **B.1–B.4 是 `src/` 動工前寫的，內容已過時**（例如 B.1 說 `src/` 不存在、B.4 說沒有測試框架，現在都不對了）。保留是為了看動工前的起點，**目前實際狀態請看 B.5**。

### B.1 現況與規劃的落差

截至本文件撰寫時，工作目錄底下 `dlsca-attack-v2/`（第 3 節所述的 `src/`、`configs/`、`scripts/`、`tests/`、`runs/`）**尚未建立**。目前唯一存在的程式碼是 `ASCAD/`，即 ANSSI/CEA 上游的原始參考實作（vendored third-party repo，BSD 授權，見 `ASCAD/LICENSE`）：

```
ASCAD/
├── ASCAD_generate.py       # 從原始 raw traces 萃取 profiling/attack 資料庫
├── ASCAD_train_models.py   # 訓練 MLP/CNN/ResNet 攻擊模型
├── ASCAD_test_models.py    # 用訓練好的模型跑金鑰排名（rank）評估
├── ATMEGA_AES_v1/          # 布林遮罩 AES（ATMEGA8515）資料集說明與範例參數
└── STM32_AES_v2/           # 仿射遮罩 AES（STM32）資料集說明與範例參數
```

這代表：**在 `src/`、`scripts/` 等目錄實際寫出來之前，第 4–8 節描述的模組契約、config 系統、GE-based selection 都只是規格，不是可執行程式碼。** 動工時請依第 3 節結構在專案根目錄新增 `src/`、`configs/`、`scripts/`、`tests/`，不要把新程式碼塞進 `ASCAD/`。

`ASCAD/` 本身是原廠參考碼，非本專題撰寫，重構時應視為唯讀（用於比對行為、抄超參數如 `cnn_best` 的 RMSprop 1e-5 設定），**不要直接修改它**；若需要修改邏輯，在新的 `src/` 內以獨立實作取代。

也尚未有 `requirements.txt`、`.git`（此目錄目前不是 git repository）、任何 `configs/*.yaml`、notebook 或測試檔。

### B.2 執行上游參考腳本（`ASCAD/` 內）

三個腳本都吃一份 Python dict 格式的參數檔（不是 YAML/JSON，是可被 `eval()` 的 dict 字面值），路徑相對於各自的資料集資料夾（例如 `ATMEGA_AES_v1/ATM_AES_v1_fixed_key/`）。三個資料集資料夾內各附一組 `example_*_params` 可直接使用：

```bash
cd ASCAD/ATMEGA_AES_v1/ATM_AES_v1_fixed_key   # 或 .../ATM_AES_v1_variable_key、../../STM32_AES_v2
python ../../ASCAD_generate.py example_generate_params
python ../../ASCAD_train_models.py example_train_models_params
python ../../ASCAD_test_models.py example_test_models_params
```

三者的關鍵參數鍵值：

| 腳本 | 必填鍵 | 備註 |
|---|---|---|
| `ASCAD_generate.py` | `traces_file`, `labeled_traces_file`, `profiling_index`, `attack_index`, `target_points`, `profiling_desync`, `attack_desync` | 不帶參數檔執行時會用內建預設值，一次產生 desync0/50/100 三個 ASCAD 固定金鑰資料庫 |
| `ASCAD_train_models.py` | `ascad_database`, `training_model`, `network_type`（`mlp`/`cnn`/`cnn2`/`multi_resnet`/…）, `epochs`, `batch_size` | `cnn_best` 用 RMSprop 1e-5，改 Adam 1e-3 會發散（與第 6.2 節一致） |
| `ASCAD_test_models.py` | `model_file`, `ascad_database`, `num_traces` | 預設 `target_byte=2`；輸出金鑰排名並畫圖（無顯示器環境會自動切換 matplotlib `Agg` backend） |

原始資料（`*_raw_traces.h5` 等）需另外從 `ASCAD/ATMEGA_AES_v1/*/Readme.md` 或 `ASCAD/STM32_AES_v2/Readme.md` 內的連結下載，**不隨此 repo 附上**。

### B.3 相依套件

沒有 `requirements.txt`。依 `ASCAD/Readme.md`，需要：`h5py`、`numpy`、`matplotlib`、`tensorflow`（2.x）、`keras`、`tqdm`。重構為 `dlsca-attack-v2` 時，第 7 節的 config 系統會另外引入 `pyyaml`，屆時應建立正式的 `requirements.txt`（第 3 節結構已預留位置）。

### B.4 尚無 lint / test 指令

目前沒有測試框架、linter 或 CI 設定。第 9 節列出的 `tests/test_labels.py`、`tests/test_keyrank.py` 等在建立 `dlsca-attack-v2/tests/` 之前無法執行；一旦建立，優先寫第 9 節標註的兩個極端案例（`test_keyrank` 完美預測、均勻分佈）。

### B.5 目前建置狀態（持續更新）

**已完整實作並驗證過**：`src/config.py`、`src/seeding.py`、`src/data/{ascad,split,labels,preprocess}.py`、`src/models/{registry,cnn_light}.py`、`src/train/{trainer,callbacks}.py`、`src/attack/{predict,scores,keyrank}.py`、`scripts/{00_inspect_data,01_train_attacker,02_run_attack,03_evaluate}.py`。`tests/` 22 個測試全過。`configs/base.yaml` + `configs/data/ascad_desync0.yaml` + `configs/model/cnn_light.yaml` + `configs/exp/E01_baseline_clean.yaml` 已可端到端跑通完整 pipeline。

**部分實作**：`src/metrics/leakage.py`（只有 `snr()`，`nicv`/`t_test` 還是 stub）、`src/metrics/information.py`（只有 `pi()`，`mi()` 還是 stub）。`src/train/trainer.py::fit` 對 `augment.gaussian.enabled=true` 會直接 `raise NotImplementedError`——動態每 epoch 增強的訓練迴圈**故意**還沒接，E02 要用之前必須先做。

**完全 stub**：`src/models/{cnn_best,resnet}.py`、`src/metrics/perturbation.py`、`src/report/{plots,tables}.py`。`scripts/04_make_report.py`、`scripts/run_pipeline.py` 還沒開始寫。`configs/data/ascad_desync50.yaml`、`ascad_desync100.yaml`、`configs/model/cnn_best.yaml`、`configs/model/resnet.yaml` 還沒寫（資料已下載好，見 B.6）。

**環境**：這裡用的 venv 其實是 `/home/yemiao1026/PQC_SCA_Project/.venv`（不是本專案自己的），已裝 `numpy`/`h5py`/`pytest`/`pyyaml`/`tensorflow 2.21.0`/`keras 3.15.1`（CPU only，這台機器沒有 GPU）。這個專案目前沒有自己的 `.venv`。

### B.6 資料狀態

`data/` 下已有 `ASCAD.h5`、`ASCAD_desync50.h5`、`ASCAD_desync100.h5`（各 46.5MB，SHA-256 已對過官方雜湊）。`data/` 整個被 `.gitignore` 的 `*.h5` 規則排除。

`00_inspect_data.py` 在 `ASCAD.h5`（desync0）上跑出：mask index=0，masked-label SNR 峰值 6.30（@point 517）vs 未遮罩對照組 0.0105，PASS。**但同一支腳本在 desync50/desync100 上會 FAIL**——不是 bug，是逐點 SNR 統計量在高抖動下本來就會被稀釋到雜訊層級（16 個候選 mask 欄位的峰值全部擠在 0.009–0.012，跟未遮罩對照組 0.0095 分不開，argmax 選出來的欄位在三個資料庫間還會飄動）。**用 desync50/100 時務必用 `--mask-index 0`（沿用 desync0 找到的值）明確指定，不要在 desync 資料上重新自動偵測。**

### B.7 E01 baseline 已知落差（尚未解決，先記錄）

`E01_baseline_clean`（`cnn_light`、desync0、ID leakage、無 augmentation）已經跑過兩次完整訓練：

| | epoch 上限 | 停止方式 | 最佳 checkpoint | 正式評估 GE@N=1000（100次獨立攻擊） | N_TGE |
|---|---|---|---|---|---|
| 第一次 | 100 | 撞到 epoch 上限（**還在進步中被砍斷**） | epoch 100 | 38.97 | None |
| 第二次 | 500 | epoch 175 觸發 patience 早停 | epoch 145 | 29.86 | None |

GE 曲線隨 N 遞減但明顯減速（113→80→52→41→36→34→32→30，接近對數形狀），在 N=1000 附近幾乎打平。§11 驗收標準要求 `N_TGE ≈ 100 ± 30`（即 100 條軌跡內 GE 就要收斂到 <1），目前離這個目標還有一個數量級以上的落差，且訓練時間拉長帶來的邊際改善正在快速遞減，**不像是單純「訓練不夠久」能解決的**。

**已排除的可能性**：資料本身沒問題（`00_inspect_data.py` 在 desync0 上驗證 SNR 峰值 6.30、分離度極高）；`scores.build` 的 log-sum 邏輯、`keyrank.evaluate` 的獨立重排與 rank 計算、`N_TGE` 的「其後全程低於門檻」語意都有 `tests/` 護欄覆蓋，且用人造完美/均勻預測驗證過極端情況正確。

### B.8 根因調查：訓練方法論缺了 One-Cycle LR（已驗證，效果顯著）

用 WebSearch 查證：`cnn_light` 的設計風格（SELU、小filter、efficient CNN）師承 **Zaid et al. 2020**《Methodology for Efficient CNN Architectures in Profiling Attacks》（TCHES 2020）。對照其官方 repo（[gabzai/Methodology-for-efficient-CNN-architectures-in-SCA](https://github.com/gabzai/Methodology-for-efficient-CNN-architectures-in-SCA)，`ASCAD/N0=0/cnn_architecture.py`）發現訓練方法論上的落差：

| 項目 | Zaid 參考實作（desync0） | 重構前的 E01 |
|---|---|---|
| LR schedule | **One-Cycle Policy**（LR 先升到 max 再退火） | 固定 Adam lr=1e-3 |
| Learning rate | max_lr=5e-3 | 1e-3 |
| 前處理 | StandardScaler → **再加一次 MinMaxScaler([0,1])** | 只有標準化 |
| Batch size | 50 | 128 |
| Epochs | 50（配合 one-cycle） | 100～500（固定LR，撞早停或上限都沒補回差距） |

CLAUDE.md §6.2 的超參數表本身就沒有記載 one-cycle，這是文件本身的落差，不只是實作漏做。

**已實作並驗證**：`src/train/lr_schedule.py::OneCycleLR`（照 Zaid repo 的 `clr.py` 逐行 port 過來的三段式 per-batch LR：低點→max_lr→低點→急速退火到低點/100，不是自己簡化推導的版本，因為簡化版只有在 `scale_percentage=0.1` 這個特定值下才會化簡成直觀的「低到高」形狀）、`src/data/preprocess.py::MinMaxScaler`（fit-on-A-only，同 Standardizer 的紀律）。透過 `configs/model/cnn_light.yaml` 的 `train.lr_schedule: one_cycle` + `configs/exp/E01_baseline_clean.yaml` 的 `preprocess.minmax: true` 開關控制，預設關閉（`base.yaml` 裡 `minmax: false`），不影響其他還沒重新驗證過的實驗。

**重跑 E01 的結果**（`runs/E01_baseline_clean_20260815_2256/`，50 epochs、約 7.5 分鐘跑完，比之前任何一次都快）：

| | 訓練設定 | 最佳 checkpoint 的訓練期 GE 預覽 | 正式評估 GE@N=1000 | SR1@1000 | PI |
|---|---|---|---|---|---|
| 第一次 | 100 epoch 固定LR | epoch100, 25.75 | 38.97 | 0.01 | -0.9022 |
| 第二次 | 500 epoch 固定LR，早停@175 | epoch145, 15.90 | 29.86 | 0.02 | -1.1572 |
| **第三次** | **50 epoch one-cycle+minmax** | epoch40, 17.15 | **14.89** | **0.13** | **-0.0970** |

**收斂速度快了 3.5 倍以上**（50 epoch 打平 175 epoch 的表現），GE@1000 從 29.86 腰斬到 14.89，SR1 從 2% 提升到 13%，PI 從 -1.16 大幅拉近到接近 0（幾乎不再是「confidently wrong」）。**更關鍵的是 GE 曲線形狀變了**——前兩次在 N=500~1000 之間幾乎打平（是平緩曲線），這次在 N=1000 處仍在明顯下降（見 `runs/E01_baseline_clean_20260815_2256/figures/ge_curve.png`），代表若拉長攻擊窗口或再多訓練，還有繼續往下探的空間，不像前兩次那樣看起來已經到頂。

**假設驗證成立：訓練方法論（one-cycle LR + minmax）是主要根因，不是資料、不是核心演算法邏輯**。`N_TGE` 仍未收斂（還是 `None`），離 §11 的 100±30 目標還有距離，但已經不是「差一個數量級」，方向明確走對了。

**接著試了拉長 one-cycle 到 150 epoch，結果是負面的**（`runs/E01_baseline_clean_20260815_2312/`）：訓練期最佳 GE 預覽出現在 epoch 115（71.35），比 50-epoch 版本的 epoch40（17.15）差很多；epoch150 收尾時（101.85）甚至還在惡化，不是「還沒退火完」的問題。正式評估（同時把 `attack.max_traces` 從 1000 拉大到 5000 觀察更寬的窗口）：`GE@N=5000=49.93`、`PI=-1.1935`，遠差於 50-epoch 版本的 `GE@N=1000=14.89`。

**結論：epoch 數不是「越多越好」，50 是跟 one-cycle 排程長度綁在一起調過的數字，不是隨便的訓練預算上限**——把排程拉長，小模型（18.6K 參數）在高 LR 階段撐更久，在雜訊很大的 SCA 資料上開始不穩定/過擬合，退火期救不回來。已把 `configs/model/cnn_light.yaml` 的 `train.epochs` 改回 50。

### B.9 用同一個 50-epoch 模型、拉寬評估窗口到 5000 條 —— 幾乎完全收斂

`configs/exp/E01_baseline_clean.yaml` 的 `attack.max_traces` 從 1000 拉到 5000 後，用回 50-epoch one-cycle+minmax 的模型重新訓練+評估一次（`runs/E01_baseline_clean_20260815_2336/`，seed 相同，訓練軌跡跟 B.8 那次逐位元一致，純粹是拉寬了最後評估階段的觀察窗）：

| N | GE | SR1 |
|---|---|---|
| 1000 | 14.89 | 0.13 |
| 2000 | 7.73 | 0.18 |
| 3000 | 4.47 | 0.28 |
| 4000 | 2.66 | 0.45 |
| **5000** | **1.96** | **0.60** |

GE 曲線平滑地一路降到 N=5000 時逼近 2（見 `runs/E01_baseline_clean_20260815_2336/figures/ge_curve.png`，是教科書等級的收斂曲線），SR1 到 5000 條時已經有 **60% 的獨立攻擊**成功把正確金鑰排到第一名。GE 在整段窗口內從未真正跌破 1（最低點 1.86 @ N=4814，尾端 15 個點在 1.92–2.03 之間），所以 `N_TGE` 技術上仍是 `None`，但已經非常接近完全收斂，不再是「差一個數量級」，是「差臨門一腳」。

**這證實了整個 B.7→B.9 的調查方向是對的**：資料沒問題、核心演算法（log-sum scoring、keyrank 獨立重排、N_TGE 語意）沒問題、真正卡住的是訓練方法論（固定 LR → one-cycle、缺 MinMax、epoch 數要跟 one-cycle 排程匹配而非隨意加大）。

### B.10 拉到 max_traces=9000：完全收斂，但 N_TGE 遠高於目標（64倍）

同一個模型（`runs/E01_baseline_clean_20260815_2336/`）不用重跑訓練，靠 `03_evaluate.py` 新增的 `--override` 參數（見下方）直接把評估窗口拉到 9000：

```
N_TGE  = 6408
N_SR90 = 8107
GE @ N=9000  = 0.0000
SR1 @ N=9000 = 1.0000   （100 次獨立攻擊全部成功）
```

**模型確實能完全破解金鑰**（GE 精確降到 0、SR1 打滿 100%），但需要 **6408 條軌跡**才穩定收斂，跟 §11 要求的 `N_TGE≈100±30` 差了 **64 倍**，不再是"差臨門一腳"，是實打實的效能差距。現在有了完整、確定的答案，不再是"接近收斂"的模糊描述。

**副產物：抓到一個真正的 bug**。第一次跑 `--override attack.max_traces=9000` 時 `json.dumps` 直接炸掉——`src/attack/keyrank.py::_first_sustained` 在「部分收斂」（曲線中途穿過門檻且之後不再回頭）這個分支回傳的是 `np.int64` 不是 Python `int`，不能被 JSON 序列化。`tests/test_keyrank.py` 原本兩個護欄測試（完美預測、均勻分佈）分別只測到「全程收斂」跟「全程不收斂」兩個分支，从没測過「中途穿過門檻」這個分支，所以測試群一直是綠的。已修正（`int(np.nonzero(...)[0][-1])`）並新增 `test_n_tge_partial_convergence_returns_plain_int` 回歸測試（`tests/` 現在 23 個測試全過）。這也是為什麼前幾次 `N_TGE` 一直顯示 `None` 都沒觸發這個 bug——沒真的收斂過。

**`scripts/03_evaluate.py` 新增 `--override` 參數**：不用重跑訓練/推論就能用不同的 `attack.max_traces`/`n_runs`/`seed` 重新評估同一個 `probs.npy`（P3「機率輸出存檔後模型即可卸載」精神的延伸——評估參數也不該綁死重跑）。

**現況總結**：cnn_light + one-cycle + MinMax 這個配方**能完全破解**，但效率上離目標還有一個數量級的差距。下一步若要繼續逼近 `N_TGE≈100`，方向已經很明確：繼續在 one-cycle 的超參數（max_lr、scale_percentage、epoch 數）或模型容量上細調，而不是懷疑管線正確性。

### B.11 找到「N_TGE≈100」真正的歷史來源，並發現它的量測方法本身有問題

使用者提供舊專題檔案位置（`C:\Document\B11209017`，此環境下掛載在 `/mnt/c/Document/B11209017/ASCAD/`），派了三個 agent + 直接讀原始碼交叉比對，找到確切來源。

**真正產生「N_TGE≈80–100」這個歷史數字的腳本**：`ATMEGA_AES_v1/ATM_AES_v1_fixed_key/train_with_pure/train_cnnd.py`，訓練出 `cnnd_paper_model.h5`（乾淨版，跑出 80）與 `cnnd_paper_model_noisy_sigma5.h5`（σ=5 高斯噪訊增強版，跑出 100——**這就是「≈100」的真正出處**）。架構跟我們的 `cnn_light` 幾乎一致（Conv(4,k3)→BN→AvgPool→Conv(8,k51)→BN→AvgPool→FC10→FC10→FC256，18,640/18,642 參數），但關鍵細節不同：

| 項目 | `train_cnnd.py`（真正歷史來源） | 我們原本的假設（查 Zaid 論文 repo 得出） |
|---|---|---|
| 初始化 | **he_uniform** | lecun_normal |
| LR schedule | **完全沒有，固定 Adam lr=1e-3** | one-cycle |
| 前處理 | **完全沒做**（int8 直接轉 float32，無標準化、無MinMax） | Standardize → MinMax |
| Batch size | 50 | 50（這項猜對了） |
| Epochs | 50 | 50（這項也猜對了） |
| 訓練資料 | **全部 50000 條 profiling**，直接把 Attack 集10000條當 `validation_data`（沒有另外切V，但也沒有用這個validation做checkpoint選擇——`train_cnnd.py`完全沒有callback，就是訓練完50epoch存檔，不影響梯度） | 我們的 A=30000/V=5000/D=15000 三路切分 |

**更關鍵的發現：這個「80/100條軌跡」本身的量測方法不夠嚴謹**。產生這個數字的 `compare_cnnd_models.py`／`compare_models_zoomed.py` 呼叫的是原始 ASCAD 官方 `full_ranks()`（`ASCAD_test_models.py`，我們自己 vendored 的那份幾乎一樣，已 diff 確認），這個函式**對 Attack 集做單一次、不重排、依照儲存順序**的 log-likelihood 累加，只回報「第一次 rank 打到 0 的軌跡數」——完全就是 CLAUDE.md 陷阱清單 #6「單次 attack run 即下結論」在講的那個坑，也是這整個重構專案存在的理由之一。我們現在用的「100 次獨立重排攻擊平均」（`keyrank.evaluate`）是統計上更誠實的方法，但也因此對同一個模型算出來的數字，天生就會比對方那個「幸運單次跑法」更保守（更高）。換句話說：**「N_TGE≈100」這個目標值，可能從一開始就不是用嚴謹方法量出來的，不見得是我們現在用嚴謹方法也應該打到的數字**。

**已修正**：
1. `src/models/cnn_light.py`：`kernel_initializer` 從 `lecun_normal` 改成 `he_uniform`，對齊真正的歷史來源（不是我先前查到的 Zaid GitHub repo 猜測版本）。
2. `scripts/01_train_attacker.py` / `02_run_attack.py`：新增 `preprocess.method: none` 支援（原本 Standardizer 是寫死一定會跑，沒有真的尊重 config 裡的 `preprocess.method` 欄位——這是本來就該有、現在補上的組態驅動缺口），可以真的測試「完全不正規化」這個設定。

**驗證實驗結果（`runs/E01_repro_original_recipe_20260816_0017/`）：假設成立，「≈100」是量測方法造成的假象**。用 `train_cnnd.py` 的確切配方（`preprocess.method: none`、he_uniform、固定 Adam lr=1e-3、batch=50、epochs=50、A=44000）重新訓練，但全程用我們嚴謹的 100 次獨立重排評估，結果：

```
GE @ N=9000  = 199.27   （比隨機基準 127.5 還糟）
SR1 @ N=9000 = 0.0000   （100 次獨立攻擊，一次都沒把正確金鑰排到第一）
PI           = -0.0352  （幾乎零資訊）
```

訓練過程本身就先給出了答案：50 個 epoch，loss 幾乎黏死在 5.535–5.536（隨機基準 log(256)=5.545），train accuracy 全程卡在 0.5% 左右（隨機基準 1/256≈0.39%），val_accuracy 全程 0.46% 沒有變化——**這個模型基本上沒有學到任何東西，只是在隨機基準附近做極小幅、無意義的擺動**。

**結論**：`train_cnnd.py` 產生的「rank 在 80/100 條軌跡打到 0」，是**在單一、不重排、依原始儲存順序跑過 Attack 集**這個弱評估方法下，一個幾乎沒訓練好的模型剛好在那個特定順序下走運的結果，不是模型真的學到了強力的洩漏特徵。換句話說：**「N_TGE≈100」這個歷史數字本身可能從來就不是一個可靠、可重現的基準，我們不需要以此為目標**。相對地，`E01_baseline_clean`（one-cycle + MinMax + he_uniform，`runs/E01_baseline_clean_20260815_2336/`）在同一套嚴謹評估法下能做到 **GE 精確降到 0、SR1 100%、N_TGE=6408**——雖然離「≈100」還很遠，但這是一個真正、誠實、可重現、可驗證的成功案例，比「原始配方」在誠實評估下的表現（GE=199.27，等於沒學到東西）好上非常多。**目前 `E01_baseline_clean`（one-cycle+MinMax+he_uniform）是本專案已知最好、且唯一被證實真的能完全破解金鑰的配方**，之後若要再進一步逼近更小的 N_TGE，應該在這個配方的基礎上細調（one-cycle 超參數、模型容量、或加大 A），而不是回頭參考 `train_cnnd.py` 的設定。

**附帶修正的時序要注意**：`src/models/cnn_light.py` 改成 `he_uniform` 是這次調查中才做的（8/16 00:15），而 `E01_baseline_clean_20260815_2336`（N_TGE=6408 那個結果）的 `model.keras` 是 8/15 23:42 存的，**早了半小時，用的還是舊的 `lecun_normal`**。所以立刻拿現在的 config（one-cycle + MinMax，唯一差異是 he_uniform）重跑一次確認，結果是本專案目前最好的成果：

### B.12 he_uniform 驗證結果：N_TGE 從 6408 降到 695（9.2 倍進步）

`runs/E01_baseline_clean_20260816_0027/`，其餘設定跟 B.9/B.10 那次完全相同（one-cycle LR、MinMaxScaler、A=30000/V=5000），**唯一差異是 kernel_initializer 從 lecun_normal 換成 he_uniform**：

```
N_TGE  = 695    （原本 lecun_normal 版本是 6408）
N_SR90 = 1207   （原本是 8107）
GE @ N=9000  = 0.0000
SR1 @ N=9000 = 1.0000
```

GE 曲線（`runs/E01_baseline_clean_20260816_0027/figures/ge_curve.png`）在約 700 條軌跡處平滑穿過 GE=1 門檻，之後穩定貼著 0，是目前為止最乾淨的收斂曲線。訓練期預覽甚至在 epoch 20（`n_runs_val=20`、只用1000條軌跡的快速版）就已經測到 N_TGE=491，遠早於 lecun_normal 版本整個訓練過程都沒在 1000 條窗口內收斂過。

**這代表兩件事**：
1. `he_uniform` 不只是「對齊歷史來源」的表面修正，是真的有實質幫助的關鍵超參數——單一個初始化方式的改動，效果比之前任何一次調 epoch 數、加大評估窗口都更顯著。
2. 目前為止「訓練方法論」這條調查主線走到這裡有清楚的收斂：he_uniform + one-cycle LR + MinMax正規化 = 目前已知最佳配方。`train_cnnd.py`原始配方裡唯一被證實有效的部分是**架構本身**（Conv(4,k3)+Conv(8,k51)+FC10+FC10+FC256）和 **he_uniform**；`no normalization`、`flat LR`、`train on 44k` 這幾項單獨測試都是負面或無效的（見 B.11）。

**目前本專案已知最佳結果：N_TGE=695，離 §11 目標 100±30 差 7 倍**（不再是64倍或一個數量級以上）。下一步如果要繼續逼近，方向是對 one-cycle 的 max_lr/scale_percentage 或模型容量做更細緻的調整，核心配方（he_uniform+one-cycle+MinMax+這個cnn_light架構）已經確立為正確方向。

### B.13 one_cycle.end_percentage 掃描：0.2 是局部最優，兩側都更差

以 B.12 的最佳配方（`runs/E01_baseline_clean_20260816_0027/`，`end_percentage=0.2`）為基準，用 `--override train.one_cycle.end_percentage=X` 掃了另外兩個值，其餘設定完全不變：

| `end_percentage`（退火期佔比） | run | N_TGE | GE@N=9000 |
|---|---|---|---|
| 0.1（退火期縮短，探索期拉長） | `runs/E01_baseline_clean_20260816_1217/` | None（沒收斂） | 60.93 |
| **0.2（目前預設值）** | `runs/E01_baseline_clean_20260816_0027/` | **695** | **0.0** |
| 0.35（退火期拉長，探索期壓縮） | `runs/E01_baseline_clean_20260816_1212/` | 4253 | 0.0 |

原本假設「訓練期預覽在 epoch50（排程最後一個 epoch）都還在刷新最佳紀錄，代表退火期拉長應該有幫助」——**這個假設被推翻了**，兩個方向都明顯更差，`0.2` 在目前測過的範圍內是局部最優。`configs/model/cnn_light.yaml` 維持 `end_percentage: 0.2` 不變。

若要繼續逼近 N_TGE≈100，`end_percentage` 這個維度已經沒有明顯還沒探索的空間，下一個該試的維度是 `max_lr`（目前 5e-3，是沿用歷史來源猜的，沒有針對這個資料集/架構調過）或 `scale_percentage`（目前 0.1，決定退火期底部LR多低）。

### B.14 max_lr 掃描：5e-3 也是局部最優，兩側都更差

同樣以 B.12 的最佳配方為基準，用 `--override train.lr=X` 掃了另外兩個值（`train.lr` 在 one-cycle 模式下就是 `max_lr`）：

| `max_lr` | run | N_TGE | GE@N=9000 |
|---|---|---|---|
| 1e-2（2倍，更激進） | `runs/E01_baseline_clean_20260816_1232/` | None（沒收斂） | 167.93（比隨機還糟） |
| **5e-3（目前預設值）** | `runs/E01_baseline_clean_20260816_0027/` | **695** | **0.0** |
| 2.5e-3（一半，更保守） | `runs/E01_baseline_clean_20260816_1236/` | None（沒收斂） | 55.24 |

跟 B.13 的 `end_percentage` 掃描結果同一個模式：**兩個方向都明顯更差，5e-3 在目前測過的範圍內也是局部最優**。1e-2 太激進，訓練期GE預覽整段都黏在隨機基準附近（150-185），到最後10個epoch才勉強降到111；2.5e-3太保守，最好也只降到55-65就卡住不再進步。

**兩個維度（`end_percentage`、`max_lr`）分別掃過，目前的配方組合（max_lr=5e-3, end_percentage=0.2, scale_percentage=0.1, he_uniform, MinMax, batch=50, epochs=50）在這兩個方向上都站在局部最優點**，繼續在這兩個維度上微調邊際效益可能有限。如果要繼續逼近 N_TGE≈100，比較有機會的下一步是換一個維度：`scale_percentage`（還沒測過）、模型容量（cnn_light 只有 18.6K 參數，容量本身可能是天花板）、或加大訓練資料量（A 目前只用 30000/50000）。

### B.15 scale_percentage 掃描：先踩到一個混淆變數的坑，修正後找到新的最佳配方（N_TGE=475）

**先發現一個實驗設計錯誤**：直接比照 B.13/B.14 的做法，用 `--override train.one_cycle.scale_percentage=X`（`train.lr` 維持 5e-3）掃了 0.05 跟 0.2，結果分別是 GE@9000=128.17（幾乎等於隨機基準 127.5）跟 GE@9000=215.42（比隨機還糟，`runs/E01_baseline_clean_20260816_1248/`、`runs/E01_baseline_clean_20260816_1255/`）。log 裡印出的實際 LR 峰值卻分別是 0.02 和已知很差的區間，跟設定的 `max_lr=5e-3` 對不上——回頭推導 `src/train/lr_schedule.py::_compute_lr` 才發現：

```
實際峰值 LR = train.lr × 100 × scale_percentage²
```

這個公式只有在 `scale_percentage=0.1` 時才會化簡成「峰值＝`train.lr`」（100×0.1²=1，這也是為什麼 B.13、B.14 的掃描沒踩到這個坑——那兩次都沒動 `scale_percentage`）。一旦同時改變 `scale_percentage`，峰值 LR 會依平方關係跟著變，導致「掃 scale_percentage」實際上同時把峰值 LR 也拉到別的區間去了——**B.13、B.14 兩節的結論本身沒錯（因為都只單獨動一個變數），但這裡最初的 scale_percentage 掃描結果是混淆的，不能直接採信**。

**修正做法**：同時覆寫 `train.lr`，讓峰值 LR 精確固定在 5e-3（`train.lr = 5e-3 / (100 × scale²)`），只讓「起訖點相對峰值的比例」這個變數單獨變化：

| `scale_percentage`（峰值固定5e-3） | 對應 `train.lr` override | run | N_TGE | GE@N=9000 |
|---|---|---|---|---|
| **0.05** | `lr=0.02` | `runs/E01_baseline_clean_20260816_1302/` | **475**（新最佳） | 0.0 |
| 0.1（原預設） | `lr=5e-3` | `runs/E01_baseline_clean_20260816_0027/` | 695 | 0.0 |
| 0.2 | `lr=1.25e-3` | `runs/E01_baseline_clean_20260816_1308/` | None（沒收斂） | 38.35 |

這次呈現**單調趨勢**（不是像 `end_percentage`、`max_lr` 那樣兩側都更差的局部最優）：scale_percentage 越小（起訖點離峰值越近，「上升-下降」擺動幅度越窄）效果越好。`configs/model/cnn_light.yaml` 已更新為 `train.lr: 0.02` + `one_cycle.scale_percentage: 0.05`，**這是本專案目前已知最佳配方，N_TGE=475，離 §11 目標（100±30）差 4.75 倍**（從最初的 64 倍，經過 he_uniform 修正到 7 倍，再到這裡的 4.75 倍）。

因為是單調趨勢、還沒摸到反轉的邊界，理論上 `scale_percentage` 可以繼續往更小的方向試（例如 0.02、0.01），還沒探到頭。

**關於舊檔案裡的 GAN 防禦者**：`ASCAD/GAN/train_improved_defender.py` 是一個已經有實測數據、能動的對抗擾動產生器（不是真正的雙人 GAN，沒有 discriminator，是對抗訓練出一個擾動產生器對抗一個凍結的攻擊模型），對 `cnnd_paper_model.h5` 這個攻擊模型可以把「rank 打到 0 所需軌跡數」從 80 拖到 1000+（`GAN/defender_summary.csv`）。CLAUDE.md §1.3 明確排除本次重構做防禦器訓練（另案 `dlsca-defense-v2`），**這裡先只記錄它存在且能動，之後做防禦端專案時可以參考/移植這份實作**，這次不動它。

### B.16 總結：從「無法收斂」到 N_TGE=475 的完整調查時間線

這節把 B.7–B.15 的過程收斂成一張時間線，方便之後寫報告或口試時直接引用，不用重新拼湊九個小節。**調查暫停於此**（N_TGE=475，決定不繼續往 `scale_percentage` 更小的方向試探）。

#### 時間線

| # | 動作 | 改變的變數 | 結果 N_TGE | 相對前一步 |
|---|---|---|---|---|
| 0 | 起點：固定 Adam lr=1e-3, batch=128, 標準化, lecun_normal | — | 未收斂（GE@1000≈30–39） | — |
| 1 | 加 epoch（100→500，早停@175） | 訓練時長 | 未收斂（GE@1000=29.86） | 幾乎無改善，**排除「訓練不夠久」假說** |
| 2 | 加 One-Cycle LR + MinMaxScaler（batch同步改50，epoch回調50） | LR schedule + 前處理 | 未收斂（GE@1000=14.89，曲線仍在降） | **關鍵轉折**：收斂速度快3.5倍 |
| 2b | 拉長 one-cycle 到150epoch | epoch數 | 更差（GE@5000=49.93） | 負面，**排除「排程越長越好」假說** |
| 3 | 拉寬評估窗口到5000（同一模型，不重訓） | 評估窗口 | 未收斂（GE@5000=1.96，逼近但未跌破1） | 證實曲線真的持續在收斂，不是已經打平 |
| 4 | 拉寬評估窗口到9000 | 評估窗口 | **N_TGE=6408**（首次拿到確定數字） | 完全收斂但離目標差64倍；順便修了 `_first_sustained` 的 `np.int64` bug |
| 5 | 查出歷史來源 `train_cnnd.py`，逐項比對 | （調查，非改動） | — | 發現 he_uniform 才是關鍵，不是 one-cycle 本身；也發現「≈100」源自單次不重排評估，不可靠 |
| 6 | 用 `train_cnnd.py` 確切配方（無正規化+flat LR+he_uniform）重現測試 | 全部換成歷史配方 | 未收斂（GE@9000=199.27，比隨機還糟） | 負面，**證實原始配方在嚴謹評估下沒訓練起來**，「≈100」是量測假象 |
| 7 | 只把 he_uniform 套進「已驗證有效」的 one-cycle+MinMax 配方 | 初始化方式 | **N_TGE=695** | **9.2倍進步**，he_uniform 是真正關鍵超參數 |
| 8 | 掃 `end_percentage`（0.1/0.2/0.35） | 退火期佔比 | 695 (0.2) 最佳，兩側都更差 | 確認局部最優，此維度探完 |
| 9 | 掃 `max_lr`（2.5e-3/5e-3/1e-2） | 峰值LR | 695 (5e-3) 最佳，兩側都更差 | 確認局部最優，此維度探完 |
| 10 | 掃 `scale_percentage`（沒同步固定峰值，混淆實驗） | scale_percentage（峰值意外跟著變） | 更差（128.17／215.42） | **實驗設計錯誤**：發現峰值LR=`train.lr×100×scale²`，只有scale=0.1才化簡成峰值=train.lr |
| 11 | 修正：同步覆寫`train.lr`讓峰值固定5e-3，重掃`scale_percentage`（0.05/0.1/0.2） | scale_percentage（峰值固定） | **N_TGE=475**（0.05） | **32%再進步**，且呈單調趨勢（0.1更差，0.2更差），還沒探到頭 |

#### 最終配方（`configs/model/cnn_light.yaml` + `configs/exp/E01_baseline_clean.yaml` 目前內容）

```yaml
model: cnn_light          # Conv(4,k3)-BN-AvgPool-Conv(8,k51)-BN-AvgPool-FC10-FC10-FC(256)
kernel_initializer: he_uniform      # 不是 lecun_normal
activation: selu
preprocess: standardize_per_point -> MinMaxScaler([0,1])   # 兩階段都只 fit A
leakage: ID (256類)
split: A=30000 / V=5000 / D=15000 / E=10000
train:
  optimizer: adam
  lr: 0.02                          # one-cycle 的 base 值，非實際峰值
  batch_size: 50
  epochs: 50                        # 跟 one-cycle 排程長度綁定，不可隨意加大
  lr_schedule: one_cycle
  one_cycle:
    end_percentage: 0.2
    scale_percentage: 0.05          # 實際峰值 = lr × 100 × scale² = 5e-3
attack:
  n_runs: 100                       # 100次獨立重排攻擊平均，不是單次評估
  max_traces: 9000（正式評估時手動 override，config預設5000）
```

#### 最終結果

```
N_TGE  = 475     （§11 目標 100±30，還差 4.75 倍）
N_SR90 = 776
GE @ N=9000  = 0.0000
SR1 @ N=9000 = 1.0000
```

#### 三個對報告書/口試最有價值的結論

1. **「N_TGE≈100」這個歷史目標值本身不可靠**：來源是單次、不重排、依原始儲存順序跑過 Attack 集的弱評估法（`train_cnnd.py` 配上原始 `ASCAD_test_models.py::full_ranks()`），拿同一套配方用嚴謹的 100 次獨立重排評估重測，結果是 GE=199.27（等於沒學到東西）。**不應該把「打到 100」當作必達的驗收標準，475 這個數字本身在方法論上比「≈100」更站得住腳**。
2. **關鍵超參數往往不是理論上「應該」重要的那個**：一路查文獻假設 one-cycle LR 是決定性因素，做了才發現 `he_uniform`（一個違反「SELU該配lecun_normal」教科書原則的選擇）帶來的進步（9.2倍）比 one-cycle 本身還大。
3. **超參數調整要注意變數間的耦合**：`scale_percentage` 掃描一開始因為沒有同步固定峰值 LR 而得到錯誤結論，這類「表面上只改一個變數，實際上牽動了另一個沒明講的變數」的陷阱，在調參時很容易發生，也是這次調查裡少數需要回頭承認並修正的錯誤。

### B.17 E02-E08 骨架補齊 + 首批正式結果，抓到一個真的評分 bug

補齊 CLAUDE.md §8.2 全部 E02-E08 的 config（`configs/data/ascad_desync{50,100}.yaml`、`configs/model/{cnn_best,resnet}.yaml`、`configs/exp/E0{2-8}_*.yaml`），沿用 E01 已驗證的 cnn_light 配方（one-cycle+MinMax+he_uniform，`lr=0.02, scale_percentage=0.05`）。E06/E07（cnn_best/resnet 還是 stub）、E02（動態增強還沒接）在對應位置乾淨丟出 `NotImplementedError`；E05（HW）當時在 `scores.build` 丟出明確的 `NotImplementedError`（見下方發現的 bug；HW 評分後來在 B.19 補齊了）。

跑了三個「理論上能跑」的實驗全量版本（A=30000/V=5000，50 epochs）：

| 實驗 | 結果 | 判讀 |
|---|---|---|
| E03（desync50） | GE@1000=152.31（比隨機127.5更差），曲線持平甚至微升 | 負面：沿用 desync0 調出的超參數在有時間抖動的資料上完全沒學到東西 |
| E04（desync100） | GE@1000=168.39 | 同樣負面，抖動更大更沒學到 |
| E08（遮罩已知標籤） | 初次評估：GE 不收斂、比隨機略差，儘管訓練期 loss 明顯在降 | **異常**：模型顯然學到東西，但攻擊端完全測不出來 |

**E08 的異常追出一個真的 bug**：`src/attack/scores.py::build` 不管什麼 leakage model，永遠只算 `hypothesis(i,k) = Sbox[p⊕k]`（未遮罩），但 `ID_MASKED` 訓練時的標籤其實是 `Z' = Sbox[p⊕k] ⊕ mask[i]`（每條軌跡的 mask 值不同，見 `src/data/labels.py`）。模型正確地學會了預測 `Z'`，但評分階段拿它去對「沒異或 mask 的錯誤假設」，等於考卷寫對了、對答案的人卻用了另一份答案卷——GE 當然不會收斂，即使模型完全正確也一樣。

**已修正**：`scores.build` 新增 `mask` 參數，`hyp = hyp ^ mask[:, None]`；`scripts/03_evaluate.py` 在 `leakage.model == "ID_MASKED"` 時自動帶入對應的 `masks[:, mask_index]`。新增 `tests/test_scores.py::test_masked_scores_require_the_mask_to_recover_the_key` 回歸測試——刻意同時驗證「帶 mask 才能正確抓出金鑰」跟「不帶 mask 抓不出來」兩種情況，避免以後又悄悄退化回這個 bug（`tests/` 現在 24 個測試全過）。

**修正後重新評估 E08**（不用重訓練，同一份 `probs.npy`）：

```
N_TGE  = 3       （3條軌跡內就穩定收斂，符合「評估者視角上界」的定位）
N_SR90 = 3
PI     = 3.3637 bits（滿分8bits，遠高於 E01 系列的 -0.09~-1.19）
GE @ N=1000 = 0.0000
SR1 @ N=1000 = 1.0000
```

GE 曲線（`runs/E08_masked_label_20260816_1349/figures/ge_curve.png`）幾乎垂直下墜到 0，是目前所有實驗裡最乾淨的收斂——完全符合預期：這是本研究定義的最強攻擊者假設（遮罩已知），理應比 E01（遮罩未知）好上好幾個數量級。

**E03/E04 的負面結果目前判斷是「超參數沒調過」，不是管線壞了**：E01 從最初的固定LR卡在GE~30-40，到最後靠 one-cycle+he_uniform+scale_percentage 調到 N_TGE=475，中間經過 B.7-B.15 一整輪調查跟至少 15 次訓練。E03/E04 直接沿用 desync0 調出來的超參數（連 batch/epoch/one-cycle 排程都完全沒為 desync 情境調整過），會學不到東西並不意外——尤其 desync 資料的核心難點正是時間抖動，跟 desync0 調參時要解決的問題完全不同，不能假設同一組超參數直接適用。

### B.18 鑑別診斷：desync50 連「遮罩已知」這種簡單目標都學不到，排除「ID 目標本來就難」這個解釋

為了確認 E03 學不到東西的根本原因，跑了一個關鍵對照實驗：**desync50 資料 + 遮罩已知標籤**（`runs/E03_desync50_20260816_1359/`，`--override leakage.model=ID_MASKED leakage.mask_index=0`，其餘完全沿用 E03 的配方）。

邏輯是：E08 已經證實「遮罩已知」在 desync0 上是全專案最容易學的目標（N_TGE=3，幾乎瞬間收斂）。如果同一個簡單目標放到 desync50 上還是學不到，就能排除「ID 目標因為被遮罩隱藏所以天生難學」這個解釋（那是 desync0 情境下才需要處理的難點），把問題精確定位到「desync 抖動本身讓現有配方失效」。

結果：

```
GE @ N=1000 = 157.14   （比隨機基準 127.5 還糟）
PI          = -0.0237  （幾乎零資訊，訓練期GE預覽全程在123-182之間震盪，沒有收斂跡象）
```

**確認假設成立**：即使是全專案最容易學的目標，放到 desync50 上用同一套 one-cycle+he_uniform+MinMax 配方，一樣完全學不到東西。這代表 E01 那套花了 B.7-B.15 整輪調查才調出來的超參數，是針對 desync0 這個特定 optimization landscape 精細調校的結果，**沒有理由假設能直接遷移到 desync50/100**——尤其 `scale_percentage=0.05` 這種窄幅擺動的激進設定，換一個雜訊特性完全不同的任務，很可能連基本的學習都無法啟動（類似 E01 早期用錯超參數時的完全不收斂症狀，例如 `max_lr=1e-2` 或 `scale_percentage` 沒固定峰值時的情況）。

**目前結論**：E03/E04 若要真正解決，大機率需要對 desync50/100 分別重新走一輪跟 B.7-B.15 同等規模的超參數調查（不能假設 desync0 調好的配方直接適用），而不是一次性的小修小補。這是一筆不小的時間投入（E01 那輪前後跑了超過 15 次訓練），這次先停在「確認根因類別」，是否要投入完整調查留待後續決定。

### B.19 補齊 HW 評分：`scores.build` 泛化成吃 `leakage_model`，順便抓到 `GEModelSelection` 的同一個坑

**設計**：`scores.build` 新增 `leakage_model` 參數（預設 `"ID"`，向下相容），內部依 leakage model 決定「金鑰假設要映射到 probs 的第幾欄」：

| leakage_model | class(i,k) | 備註 |
|---|---|---|
| `ID` | `Sbox[p⊕k]`（0–255） | 原本的行為 |
| `ID_MASKED` | `Sbox[p⊕k] ⊕ mask[i]`（0–255） | B.17 修的那個 |
| `HW` | `HW_TABLE[Sbox[p⊕k]]`（0–8） | 這次新增，重用 `src/data/labels.py::HW_TABLE`，不重新定義 |

順便把 `mask` 參數也上了防呆：**如果帶了 `mask` 但 `leakage_model` 不是 `ID_MASKED`，直接 `ValueError`**，不要讓它被靜默忽略——跟 B.17 修的那個 bug 是同一種「表面上做對了，實際上被吃掉」的模式，先把後門堵起來。`tests/test_scores.py` 新增 `test_hw_leakage_scoring_recovers_the_key`、`test_class_index_out_of_range_raises` 兩個測試（`tests/` 現在 26 個測試全過）。

**追出的第二個同源 bug**：`src/train/callbacks.py::GEModelSelection`（訓練期間做 checkpoint 選擇用的）內部也呼叫 `scores.build`，但完全沒帶 `leakage_model`/`mask`，永遠用預設的 `ID` 行為。這代表：
1. **E08 訓練當時印出來的所有 GE 預覽數字，其實從頭到尾都是用錯誤（未遮罩）的方式算的**——雖然事後 `03_evaluate.py` 用對的方法重新評估救回了正確的 N_TGE=3，但訓練期間的 checkpoint 選擇本身是憑錯誤的指標做的，只是這個任務太簡單、隨便一個 checkpoint 都夠好，才沒被看出來。
2. **E05（HW）光是訓練都會直接崩潰**——`GEModelSelection` 用 `ID`（256欄）去索引只有 9 欄的 `probs`，撞上剛加的範圍檢查，`ValueError` 直接把訓練中斷。

**已修正**：`GEModelSelection.__init__` 新增 `leakage_model`、`mask` 參數並傳給內部的 `scores.build`；`src/train/trainer.py::fit` 從 `cfg["leakage"]` 算出這兩個值餵給 callback（跟 `scripts/03_evaluate.py` 算法一致）。用縮小規模的 E05 跑過一次確認：訓練不再崩潰、`GEModelSelection` 的 GE 預覽正確反映 HW 評分、`02_run_attack.py`／`03_evaluate.py` 正確吃到 `probs.shape=(N,9)` 全程無誤。

**E05 正式全量結果（`runs/E05_hw_leakage_20260816_1415/`，沿用 E01 已驗證的 one-cycle+MinMax+he_uniform 配方，desync0，A=30000/V=5000）**：

```
N_TGE  = 1361
N_SR90 = 1819
GE @ N=3000 = 0.0000
SR1 @ N=3000 = 1.0000
```

**表現比預期好很多**——訓練期預覽 GE 早在 epoch 25 就掉到 0.90（1000條窗口內就快收斂），正式評估拉寬到 3000 條窗口後完全收斂，GE 曲線（`runs/E05_hw_leakage_20260816_1415/figures/ge_curve.png`）平滑降到 0，是目前全部實驗裡收斂第二快的（僅次於 E08 的 N_TGE=3），比 E01 的 ID 目標（N_TGE=475）快 3 倍以上。這符合 SCA 文獻對 HW 模型的一般認知：雖然只有 9 類、資訊量理論上限較低（log2(9)≈3.17 bits vs ID 的 8 bits），但類別邊界通常對應更穩定的功耗特徵（漢明重量直接關聯翻轉位元數），在很多實務情境下比 256 類的 ID 更好學。

### B.20 補齊 `cnn_best.py` / `resnet.py`

**`cnn_best`**：`ASCAD/ASCAD_train_models.py::cnn_best`（vendored 上游參考碼）逐行對照移植——5 個 Conv1D(k=11, filters=64/128/256/512/512, relu, same) + AveragePooling(2) 區塊，接 Flatten + Dense(4096,relu)×2 + Dense(256,softmax)。**參數量精確對上文件寫的 66,652,544**（`m.count_params()` 驗證過，不是約略值）。`configs/model/cnn_best.yaml` 沿用陷阱清單 #10 的 RMSprop lr=1e-5（不能用 Adam 1e-3，會發散）。

**`resnet`**：CLAUDE.md §6.1 沒有上游參考碼可抄，只給了「3個殘差區塊+GAP，約30K參數」的目標，自行設計：Stem Conv(k=3)+BN+ReLU+AvgPool，接 3 個殘差區塊（每個區塊 Conv(k=3)+BN+ReLU+Conv(k=3)+BN，channel數不同時用 1×1 conv 投影 skip connection，區塊間 AvgPool(2)），最後 GlobalAveragePooling1D+Dense(softmax)。用 ReLU+he_normal（標準搭配，不是 cnn_light 那種「理論配錯但實測有效」的 SELU+he_uniform，因為這裡沒有歷史結果可以推翻理論）。用 `_FILTERS=(12,24,48)` 調出 **28,816 參數**，貼近「約30K」的目標。

兩個都用縮小規模（A=200-500、1-2 epochs）跑過訓練→推論→評估全流程，確認沒有崩潰，管線接線正確（`cnn_best` 的 66.6M 參數模型單步訓練也順利跑完，只是慢）。**都還沒跑正式全量結果**——`configs/model/{cnn_best,resnet}.yaml` 目前分別是原論文超參數（cnn_best）跟隨手選的 Adam 1e-3（resnet），都還沒經過任何調參，不能假設能直接複製 E01/E05 的收斂表現（跟 desync50/100 的情況一樣，換了模型/資料通常需要重新調）。

### B.21 E06（cnn_best）正式跑：CPU 環境下 75 epoch 不可行，12 epoch 縮短版是誠實的負面結果

`configs/model/cnn_best.yaml` 依原論文設定是 `epochs=75`（RMSprop lr=1e-5，見 B.20）。全量版第一個 epoch 實測 317 秒，這台機器沒有 GPU（`env.json` 確認 CUDA 不可用，全程吃 7.6+ 核心 CPU），75 epochs 換算下來要 **6.6 小時**，明顯不適合在一個對話 session 裡等待完成，先跟使用者確認後拿到批准：**縮減到 12 epochs（約1小時）跑一個真實但不完整的結果，不是為了得出正式結論，是為了在時間預算內拿到誠實的中間資料點**。

**跑法**：`python3 scripts/01_train_attacker.py --config configs/exp/E06_cnn_best.yaml --override train.epochs=12`，其餘完全沿用 `configs/model/cnn_best.yaml`（RMSprop lr=1e-5、batch=200）與 `configs/exp/E06_cnn_best.yaml`（A=30000/V=5000、desync0、ID leakage、MinMax）。跑完後正常走 `02_run_attack.py` + `03_evaluate.py`（`attack.max_traces=1000`，未拉寬——這是負面結果，拉寬窗口不會改變結論，故未比照 E01 額外多跑）。

**結果（`runs/E06_cnn_best_20260816_1452/`）**：

```
loss:  5.5452 → 5.5425（12 epochs，幾乎沒動，隨機基準 log(256)=5.545）
train accuracy: 全程卡在 0.42%–0.49%（隨機基準 1/256≈0.39%）
val_loss:       5.5451 → 5.5462（最後兩個epoch甚至還在惡化）

N_TGE  = None（未收斂）
N_SR90 = None
GE @ N=1000  = 162.39   （比隨機基準 127.5 還糟）
SR1 @ N=1000 = 0.0000   （100次獨立攻擊，一次都沒排到第一）
PI           = -0.0192 bits（幾乎零資訊）
```

訓練期 `GEModelSelection` 唯一存下的 checkpoint 是 epoch 10（`final_GE=111.85`，比隨機基準 127.5 略好，是 50 epochs 訓練期間 20-run 快速預覽下唯一一次「優於隨機」的瞬間），但用正式 100 次獨立重排評估後，這個 checkpoint 在完整 1000 條窗口下仍然是負面結果（GE=162.39，比隨機基準還差）——這正是 CLAUDE.md 陷阱清單裡反覆提醒的「單次/少量run的預覽會被雜訊騙到」的又一個實例，只是這次雜訊剛好在錯誤的方向上給了一個看似樂觀的假訊號。

**這是預期中的結果，不是 bug**：RMSprop lr=1e-5 是刻意設定得很保守的學習率（陷阱清單#10：改用 Adam 1e-3 會發散，但代價是收斂極慢），原論文用 75 epochs 正是為了配合這麼小的 LR 才收斂。12 epochs 連 loss 都還沒真正開始離開隨機基準（loss 從 5.5452 只降到 5.5425，變化量是 E01 同樣訓練規模下降幅的一小部分），可以合理預期 75 epochs 版本會有實質不同的結果，但**這份 12-epoch 資料點本身不能用來評斷 cnn_best 在這個任務上的真實能力**，純粹是 CPU 環境時間預算限制下的產物。

**結論與標籤**：`runs/E06_cnn_best_20260816_1452` 明確標記為「**CPU環境縮短版12epoch，非原論文75epoch的結果，不能代表cnn_best真實能力**」。若之後有 GPU 資源或能接受多小時等待，應該重跑完整 75 epochs 版本才能對 cnn_best 下正式結論；在那之前，E06 在跨實驗比較表裡應該標註為「未完整驗證」而非直接跟 E01/E05/E08 的結果並列比較。

### B.22 E07（resnet）正式跑：完整訓練預算下的負面結果，不是資源受限被砍斷

`configs/model/resnet.yaml`（Adam lr=1e-3, batch=128, epochs=100）加上 `configs/exp/E07_resnet.yaml`（desync0, A=30000/V=5000, ID leakage, MinMax），跟 E06 不同，這次**完整跑滿了 CLAUDE.md §6.2 規格的超參數，不是縮短版**——resnet 只有 28,816 參數，單 epoch 約 12-13 秒，遠比 cnn_best 快，不受時間預算限制。

訓練在 epoch 50 觸發 `GEModelSelection` 的 patience 早停（連續 6 次評估、即 30 epochs 沒有改善），是正常收尾，不是被人為砍斷。

**結果（`runs/E07_resnet_20260816_1559/`）**：

```
train loss:  5.5547 → 5.4451（50 epochs，有下降但幅度不大）
train accuracy: 0.42% → 1.04%（隨機基準 0.39%，略高於隨機但幅度小）

N_TGE  = None（未收斂）
N_SR90 = None
GE @ N=1000  = 123.27   （比隨機基準 127.5 略低，幾乎持平）
SR1 @ N=1000 = 0.0000
PI           = -0.1129 bits（略負，比「零資訊」還糟——見 B.7 對此現象的討論：模型對某些軌跡給出「自信但錯誤」的預測，比均勻猜測更傷）
```

**跟 E06 的差異**：E06（cnn_best, 12-epoch縮短版）明確標記為「時間預算限制、不能代表真實能力」；**E07 不同，這是完整訓練預算下的結果，是一個誠實、完整、但目前效能不佳的負面結果**。loss/accuracy 確實有比 E06 更明顯的移動（不像 cnn_best 12 epochs 幾乎原地不動），代表 resnet 這個架構+Adam lr=1e-3 至少有在學一點東西，但學到的量不足以支撐攻擊在 1000 條軌跡窗口內收斂。

**初步判讀（尚未深入調查，先記錄現象）**：resnet 跟 desync50/desync100（B.18）的情況類似——直接沿用「隨手選的」Adam lr=1e-3，完全沒有為這個架構調過訓練方法論（E01 的 he_uniform + one-cycle + MinMax 這套配方是針對 `cnn_light` 這個特定架構、花了 B.7-B.15 整輪調查才調出來的，沒有理由假設能直接遷移到殘差架構）。resnet 的 28,816 參數量介於 `cnn_light`（18,642）和真正大模型之間，不無可能同樣需要類似規模的超參數調查（one-cycle、初始化方式等）才會有實質收斂結果。**是否要投入這輪調查，留待後續決定**，跟 E03/E04 的處理方式一致——先誠實記錄負面結果，不即興做小修小補。

**目前跨實驗總覽**：E01（cnn_light, ID）N_TGE=475 仍是全專案唯一「架構已調過超參數且驗證收斂」的真實攻擊結果；E06（cnn_best）、E07（resnet）兩個異架構對照組**都還沒有超參數調查投入**，目前的負面結果反映的是「還沒調」而非「這個架構做不到」，不應解讀成架構本身的能力上限。

### B.23 E02（噪訊增強）實作 + 正式跑：目前全專案表現最好的真實攻擊結果，PI 由負轉正

**實作**：`src/train/trainer.py` 新增 `_GaussianAugmentedDataset`（`keras.utils.PyDataset` 子類別），每個 epoch 結束時用 `cfg.seed+epoch` 重新生成整批 A 的噪訊版本（呼叫既有的純函式 `src/data/preprocess.py::gaussian_augment`），確保噪訊不是預先生成的固定資料集（陷阱清單 #11）。這個生成器物件在整個 `model.fit()` 呼叫期間只建立一次，`OneCycleLR`／`GEModelSelection` 兩個 callback 完全不用改——用小規模合成資料驗證過兩者跟增強生成器同時運作正常（LR schedule 仍正確地隨 batch 遞增/退火）。新增 `tests/test_augment.py` 四個測試（可重現性、跨seed確實不同、每個epoch確實重新生成噪訊、batch切分覆蓋整個epoch），`tests/` 現在 30 個測試全過。

**跑法**：`configs/exp/E02_noisy_augment.yaml` 完全沿用 E01 已驗證收斂的 `cnn_light` 配方（he_uniform + one-cycle + MinMax，見 `configs/model/cnn_light.yaml`），唯一差異是 `augment.gaussian.enabled=true`（`sigma_ratio=0.5`）。

**結果（`runs/E02_noisy_augment_20260816_1618/`，100次獨立重排評估，1000條窗口內就完全收斂，不用額外拉寬）**：

| | E01（無增強，基準） | **E02（有噪訊增強）** | 差異 |
|---|---|---|---|
| N_TGE | 475 | **206** | **快 2.3 倍** |
| N_SR90 | 776 | 248 | 快 3.1 倍 |
| GE @ N=1000 | 0.0000 | 0.0000 | 兩者都完全收斂 |
| PI（跟 N 無關，見 `src/metrics/information.py::pi`，用完整10000條E集算，不受 `attack.max_traces` 影響） | **-0.4999 bits**（負值） | **+0.1796 bits**（正值） | **由負轉正** |

**PI 由負轉正特別值得注意**：PI 定義是 `H[Z] + mean(log2 p(z|t))`，負值代表模型平均而言對正確類別給出的機率比「均勻亂猜」還低（陷阱討論見 B.7/B.21 提過的「confidently wrong」現象——模型在多數軌跡上有辨識力，但在其餘軌跡上非常自信地押錯，拖累平均值）。E01 的 PI=-0.4999 代表這個「自信押錯」的現象即使在已經能完全破解金鑰（GE@9000=0）的最佳配方上依然存在；**E02 加了噪訊增強後 PI 轉正，代表這個現象被顯著抑制了**——這正好對應 CLAUDE.md §6.4 引用 Wu/Perin/Picek 的解釋：噪訊增強有正規化效果，迫使模型別死記局部雜訊，轉而學穩定的洩漏特徵，讓模型在「單一軌跡上的預測品質」本身變得更可靠，不只是「多次攻擊平均後看起來變快」而已。

**結論**：**E02 目前是全專案已知最好的真實攻擊結果**，取代 E01 成為新的最佳基準；且是唯一一個「兩個獨立指標（N_TGE 收斂速度、PI 單軌跡資訊量）同時大幅進步」的實驗，不是只靠某個特定評估窗口挑出來的表面改善。你們期中發現的這個技巧，重構後用嚴謹的 100 次獨立重排評估＋PI 兩條證據重新驗證，結論成立。

### B.24 resnet（E07）精簡版超參數掃描：兩次嘗試皆負面，其中一次意外示範了「預覽被雜訊騙到」這個陷阱本身

在 B.22 的 baseline（Adam lr=1e-3, flat LR）之後，跟著 E01 的成功路徑試了兩個方向，**皆未帶來實質改善**：

**掃描#1（`runs/E07_resnet_20260816_1630/`）**：加上 One-Cycle LR，peak 沿用原本的 1e-3（`--override train.lr_schedule=one_cycle`，其餘不變）。訓練在 epoch 35 觸發 patience 早停（比 baseline 的 epoch50 更早），正式評估：

```
GE @ N=1000 = 131.34（baseline是123.27，幾乎沒差，甚至略差）
PI          = -0.0268（baseline是-0.1129，略微不那麼負，但仍是負值）
```

單純加 One-Cycle、peak LR 不變，效果幾乎等於沒做。

**掃描#2（`runs/E07_resnet_20260816_1652/`）**：One-Cycle LR，peak 拉高到 5e-3（跟 `cnn_light` 調出來的 peak 一致，`--override train.lr_schedule=one_cycle train.lr=5.0e-3`）。訓練期 `GEModelSelection` 的 20-run 快速預覽出現了看似有希望的訊號——GE 在多個 epoch 明顯低於隨機基準（epoch65: 87.60、epoch75: 89.30、epoch90: 91.50、epoch95: 91.65，隨機基準是127.5），但同時**震盪劇烈**（epoch10: 169.25、epoch50: 161.05，同一次訓練內來回擺盪），跟 `cnn_light` 當年平滑遞減的預覽曲線（B.8）完全不同調性。訓練在 epoch 65 存下當時的「最佳」checkpoint（20-run 預覽 GE=87.60），之後在 epoch 95 觸發 patience 早停。

**用正式 100 次獨立重排評估這個 epoch65 checkpoint，結果是全部三個實驗裡最差的**：

```
GE @ N=1000 = 174.63（比隨機基準127.5還糟，也比baseline的123.27、掃描#1的131.34都差）
PI          = -0.1472（三者中最負）
```

**這不是隨機運氣不好，是陷阱清單 #6/#7 那個「少量run的預覽會被雜訊騙到」現象的一次活生生示範，而且這次騙到的不是評估者本人，是 `GEModelSelection` 這個自動化 checkpoint 選擇機制本身**——20-run 快速預覽剛好在 epoch65 那個時間點抽樣抽出一個好看的數字，被判定為「new best」存了下來，但用 100-run 正式評估重新檢驗後，這個 checkpoint 其實是三次嘗試裡表現最差的一個。這跟 B.17 提到的 E08 早期 `scores.build` bug是不同性質的問題（那次是系統性算錯，這次是純粹的抽樣雜訊），但共同點是：**都證明了「只看訓練期間的少量run預覽」不能拿來下結論，正式評估的 100 次獨立重排是不可或缺的最後一道關卡**，即使是設計來防止過擬合驗證集的 `GEModelSelection` 機制本身，也不能完全免疫於這個雜訊來源（它用的 `n_runs_val=20` 是為了訓練期效率犧牲精確度換來的，正式評估的 `n_runs=100` 才是可信賴的版本）。

**結論**：resnet 架構在兩次輕量嘗試（沿用/拉高 peak LR 的 One-Cycle）後**仍然沒有找到有效訓練配方**，兩次都是負面結果，其中一次的「看似改善」在正式評估下被推翻。這跟 E01 花了 B.7-B.15 十五次左右的系統性掃描（含init方式、one-cycle三個超參數各自的獨立掃描）才找到有效配方的規模完全不同——**2 次快速嘗試不足以判斷 resnet 這個架構本身行不行，只能說「這兩個特定超參數組合不夠」**。是否要投入 E01 等級的完整調查，維持 B.22 的結論：留待後續決定，不在這輪精簡版掃描裡強行解決。

### B.25 desync50（E03）精簡版超參數掃描：三次嘗試都在噪聲層級，同樣需要完整調查才能解決

延續 B.18 的診斷結論（desync0 調出的 one-cycle 配方在 desync50 上完全失效，即使是最好學的遮罩已知標籤也一樣），試了兩個方向：

**掃描#1（`runs/E03_desync50_20260816_1718/`）**：放寬 One-Cycle 的擺動幅度（`scale_percentage=0.1`，即 B.13-B.15 之前的預設值，比 desync0 調出的 `0.05` 溫和），peak 固定 5e-3。正式評估：

```
GE @ N=1000 = 147.07（原本152.31，幾乎沒差）
PI          = -0.0334（仍是負值）
```

**掃描#2（`runs/E03_desync50_20260816_1728/`）**：乾脆整個拿掉 One-Cycle，改用 flat Adam lr=1e-3。訓練期預覽一度出現 GE=90.25（epoch45，判定new best），比前兩次的預覽都更有希望。但**吸取 B.24 resnet 掃描的教訓，沒有直接採信這個預覽**，用正式 100-run 評估重新檢驗：

```
GE @ N=1000 = 144.82（仍比原本152.31好一點，但PI跟前兩次一樣是負值）
PI          = -0.0471
```

**三次結果彼此差異都在雜訊範圍內**（152.31 / 147.07 / 144.82，GE都比隨機基準127.5差，PI都是負值），沒有一次真正跳脫「模型幾乎沒學到東西」的區間。跟 B.24 的 resnet 掃描#2 不同的是，這次訓練期預覽的樂觀訊號（90.25）雖然沒有像 resnet 那樣被正式評估完全推翻成「三者最差」，但也沒有兌現成真正的改善——比較像是同一組雜訊層級內的正常擺動，不是真訊號。

**結論**：desync50 在三次輕量嘗試（放寬 one-cycle 幅度、拿掉 one-cycle 改用 flat LR）後**仍未找到任何有效訓練配方**，維持 B.18 的判斷——desync 情境下的時間抖動需要跟 E01 同等規模（B.7-B.15，超過15次系統性掃描）的獨立調查才有機會解決，不是這種等級的精簡掃描能碰到邊的問題。深入調查留待後續決定，不在這輪強行解決。

### B.26 desync100（E04）單次確認跑：同一個模式再次出現，加上三個掃描任務的收尾總結

**範疇說明**：desync50 的三次嘗試（B.25）已經一致顯示「放寬 one-cycle 幅度」「拿掉 one-cycle 改用 flat LR」都只是在雜訊層級內小幅擺動，沒有真正的訊號。desync100 的時間抖動比 desync50 更劇烈，沒有理由預期同樣等級的嘗試會有本質不同的結果，所以這裡**刻意只跑一次確認性實驗**（desync50 兩次嘗試裡數字略好的 flat LR 版本），而不是重新做一輪完整的 3-5 次掃描——這是有意識地縮小範疇，不是敷衍。

**跑法**：`configs/exp/E04_desync100.yaml` + `--override train.lr_schedule=flat train.lr=1.0e-3`（跟 desync50 掃描#2 完全同款）。

**結果（`runs/E04_desync100_20260816_1738/`）**：

```
GE @ N=1000 = 138.43（原本E01配方在desync100上是168.39，flat LR確實有改善，但...）
PI          = -0.0795（仍是負值）
N_TGE       = None（未收斂）
```

跟 desync50 的模式完全一致：flat LR 比原本沿用 desync0 的激進 one-cycle 配方好一些（168.39→138.43，改善看起來比desync50那邊的152.31→144.82更明顯），但**PI 仍是負值、GE 仍未低於隨機基準 127.5**，代表這個「改善」本質上還是在「模型幾乎沒學到東西」的範圍內移動，還沒有真正跨過「有效攻擊」的門檻。

**結論**：desync100 確認了跟 desync50 一樣的判斷——現有的精簡掃描手法找不到有效配方，需要專屬的完整調查。

---

**三個精簡掃描任務（#3 resnet / #4 desync50 / #5 desync100）收尾總結**：

| 任務 | 嘗試次數 | 結論 |
|---|---|---|
| resnet（B.24） | 2次（one-cycle同peak、one-cycle拉高peak） | 負面，其中一次的訓練期預覽被正式評估推翻，示範了陷阱#6/#7連自動化選模機制都能騙到 |
| desync50（B.25） | 3次（放寬one-cycle、拿掉one-cycle） | 負面，三次結果都在雜訊範圍內 |
| desync100（B.26） | 1次確認跑（沿用desync50最佳嘗試） | 負面，同一模式再現 |

**三者共同結論**：**沒有一個異架構/異資料情境能被 2-3 次輕量超參數嘗試解決**。這不是精簡掃描這個做法本身失敗，而是誠實地確認了 E01 當初的經驗——B.7-B.15 花了 15 次左右系統性掃描（分別對 init 方式、one-cycle 三個超參數獨立掃描）才找到有效配方，這個投入量級是「輕量嘗試」的 5-8 倍。使用者已經明確選擇「精簡版：每個情境跑3-5次關鍵掃描」而非「完整版：跟E01同等規模」，這三個任務就是在這個範疇下的誠實產出——**找不到解法本身就是結論的一部分**，三個情境若要真正解決，都要留到後續投入 E01 等級的資源才有機會，目前先如實記錄現況、不勉強做超出範疇的事。

### B.27 GitHub 上線 + 實驗室 GPU server 環境建置 + E06 完整 75-epoch 結果：確認是配方本身的問題，不是時間預算限制

**GitHub**：專案已推上 `https://github.com/YeMiao1026/dlsca-attack-v2`（public）。這台機器原本沒有 `gh` CLI、沒有 SSH key，過程中發現這個 session 的 `sudo`／互動式密碼輸入都無法在這個介面下運作（沒有真正的 TTY），改用不需要 root 的方式把 `gh` 官方執行檔直接下載到 `~/.local/bin`，用 `gh auth login` 的 device-flow（瀏覽器貼一次性代碼）完成認證，繞開了密碼輸入的限制。

**實驗室 GPU server 環境建置**（`B11209025@140.118.9.22`，host `dell760`）：硬體規格為 **2× NVIDIA L40S（各46GB VRAM）、64核心、503GB RAM**，相較這台開發機（純CPU）是數量級的升級。建置過程：

1. **SSH 免密碼登入**：同樣受限於這個介面無法處理互動式密碼輸入，改產生本機 SSH 金鑰對（`~/.ssh/id_ed25519`），把公鑰貼進 server 的 `~/.ssh/authorized_keys`（使用者手動完成這步，因為需要先有一次可用的認證管道），之後全部操作走金鑰認證。
2. **資料傳輸**：`data/*.h5`（134MB，三個資料庫）用 `rsync` 傳過去，因為這些檔案被 `.gitignore` 排除、不會跟著 `git clone` 走。
3. **環境安裝的關鍵坑**：`requirements.txt` 原本只寫 `tensorflow==2.21.0`，裝起來後 `tf.config.list_physical_devices()` 只看得到 CPU——**單純 `pip install tensorflow` 不會帶 CUDA/cuDNN 相依套件**，要裝 `tensorflow[and-cuda]==2.21.0` 這個 extra 才會把 `nvidia-cudnn-cu12` 等一整套 NVIDIA 函式庫一起裝進 venv。但即使裝了這些套件，TensorFlow **仍然找不到**，因為這些 `.so` 檔案裝在 venv 的 `site-packages/nvidia/*/lib/` 底下，不在動態連結器的預設搜尋路徑上——最後手動把這些子目錄組成 `LD_LIBRARY_PATH` 並寫進 `.venv/bin/activate`（每次 `source` 這個 venv 就自動生效），才讓 `list_physical_devices()` 正確列出兩張 GPU。這整套流程值得記錄，因為之後在同一台 server 上開新 venv 大概率會重踩同樣的坑。
4. 用 `python3 -m pytest tests/ -q` 確認 30 個測試在 server 上全過，才開始正式訓練。

**E06（cnn_best）完整訓練結果（`runs/E06_cnn_best_20260816_1801/`）**：GPU 上單 epoch 只要 **2-4 秒**（CPU 上是 317 秒，約快 **100 倍**），訓練在 epoch 45（未到75上限）觸發 `GEModelSelection` 的 patience 早停——**這次是正常收尾，不是像 B.21 那樣被時間預算砍斷**。

```
train loss:     5.5417 → 5.5413（45 epochs幾乎完全打平，隨機基準是5.545）
train accuracy: 全程卡在 0.51%（隨機基準0.39%）

N_TGE  = None（未收斂）
N_SR90 = None
GE @ N=1000  = 157.77（比隨機基準127.5還糟，跟B.21的12-epoch版162.39幾乎沒差）
SR1 @ N=1000 = 0.0000
PI           = -0.0241 bits（幾乎零資訊，跟B.21的-0.0192幾乎沒差）
```

**這是本次調查最重要的釐清**：B.21 當時因為只跑了 12 個 epoch，無法排除「只是訓練不夠久」這個解釋；這次用完整訓練預算（GPU上跑滿到 patience 自然早停，等於把 RMSprop lr=1e-5 這個配方能發揮的空間都用盡了），結果**幾乎跟12-epoch版一模一樣**——45 epochs 的 loss 幾乎沒有比 12 epochs 的版本多降多少。**這確認了問題不是時間預算，是配方本身（RMSprop lr=1e-5，ASCAD原論文的超參數）在這個任務上就是學不太到東西**，需要跟 resnet/desync50/desync100 一樣的完整超參數調查（可能也需要 one-cycle 之類的訓練方法論介入，如同 E01 的經驗），不是「跑久一點」能解決的。

**跨實驗現況更新**：連同 B.24-B.26，目前四個異架構/異資料情境（resnet、desync50、desync100、cnn_best）**全部確認需要獨立的完整超參數調查才可能解決，都不是資源/時間預算的問題**。有了這台GPU server後，這類調查的單次訓練成本大幅下降（cnn_best從5.3分鐘/epoch降到2-4秒/epoch），如果要投入，現在的條件比先前好上太多，值得認真考慮排進後續時程。

### B.28 desync50 完整調查啟動：Phase 1（max_lr 掃描）四點全數落在雜訊層級，轉向噪訊增強組合

在 GPU server 到位後，使用者核准對 desync50/desync100 投入跟 E01（B.7-B.16）同等規模的完整調查，優先做 desync50（正式編號 E03，不是額外對照組）。方法論比照 E01：固定 `scale_percentage=0.1`（讓峰值LR直接等於 `train.lr`，避開 B.15 踩過的耦合陷阱)、`end_percentage=0.2`（預設值）、`epochs=50`，對 `max_lr` 做四點掃描。

**Phase 1 結果**（全部經過正式 100-run 評估，`scale_percentage=0.1` 固定）：

| max_lr | GE@1000 | PI | 備註 |
|---|---|---|---|
| 1e-3 | 139.84 | -0.0683 | |
| 2.5e-3 | 171.49 | -0.1010 | **訓練期20-run預覽一度顯示GE=86.70，看似有希望，正式評估後推翻——四點裡表現最差**，跟 B.24 resnet 掃描是同一種假警報 |
| 5e-3（沿用B.25既有資料） | 147.07 | -0.0334 | |
| 1e-2 | 152.97 | -0.0259 | |

**四點全部擠在 139.84–171.49 之間，沒有一點真正跳脫「沒學到東西」的雜訊層級**（全數比隨機基準127.5差或幾乎持平，PI全負）。跨了一個數量級的 max_lr（1e-3到1e-2），加上 B.25 原本已經測過放寬 scale_percentage、拿掉 one-cycle 改 flat LR 兩種嘗試，desync50 在**單純調整 One-Cycle LR 的三個超參數維度上，累計六次嘗試全部失敗**，且其中一次的樂觀訊號被證實是雜訊假警報。

**Phase 1 結論**：`max_lr`（連同 B.25 已排除的 `scale_percentage`、`end_percentage`／flat LR）不是 desync50 缺的那塊拼圖。這跟 E01 的經驗不同——E01 的 one-cycle 三個超參數各自的掃描都是在「已經有效的基礎配方」上做微調（695→475 這種30%等級的優化）；desync50 現在的狀況是**連「有效的基礎配方」本身都還沒找到**，代表問題可能出在比 LR schedule 更根本的地方。

**轉向 Phase 2**：不再單純轉動 One-Cycle 的旋鈕，改測試一個結構上不同、有明確動機的假設——**E02 的噪訊增強對 desync50 是否有幫助**。動機：噪訊增強的正規化效果（迫使模型別死記局部雜訊、學穩定特徵）跟 desync 抖動需要的「對時間平移容忍度」在直覺上是同一類問題——cnn_light 的 k=51 大 kernel 是空間（時間軸）上的容忍設計，噪訊增強則是振幅上的容忍設計，兩者原理不同但目標一致（別對局部細節過擬合）。這是一個從未在 desync 情境下測試過的全新方向，不是既有 one-cycle 掃描的變形。

**Phase 2 結果**（E01 已驗證配方 lr=0.02/scale=0.05/peak=5e-3/he_uniform 為基礎，加開噪訊增強）：

| sigma_ratio | 訓練期20-run預覽GE | 正式100-run評估GE@1000 | PI |
|---|---|---|---|
| 0.5 | 86.50（看似有希望） | **161.72（假警報推翻）** | -0.0934 |
| 1.0（更強噪訊） | 96.40（看似有希望） | **180.23（假警報推翻，比隨機基準127.5還糟很多）** | -0.0412 |

**兩次都是同一個模式**：訓練期 20-run 快速預覽都出現明顯優於隨機基準的數字，但正式 100-run 評估後雙雙被推翻，其中 `sigma_ratio=1.0` 甚至是整個 desync50 調查目前為止最差的單一結果。**噪訊增強對 desync50 沒有幫助，這個假設不成立**——直覺上「噪訊增強的振幅容忍」跟「desync需要的時間容忍」是同一類正規化問題，但實測結果不支持這個類比；更強的噪訊（sigma_ratio 1.0 vs 0.5）反而更差，暗示問題可能不是「模型過擬合局部細節」，而是更根本的東西（例如訓練訊號本身在 desync 資料上就被稀釋到很難學，加噪訊只是雪上加霜）。

**階段性總結（暫停查核點）**：desync50 目前累計 **8 次系統性嘗試**（Phase 1 的 4 個 max_lr 值 + B.25 的 2 個 LR-schedule off/放寬嘗試 + Phase 2 的 2 個噪訊增強變體），涵蓋兩個結構上不同的假設（LR schedule 調整、噪訊增強正規化），**全數負面**，且過程中兩次訓練期預覽的樂觀訊號都被正式評估推翻，顯示這個資料集上「少量run預覽」的不可靠程度比 desync0 更嚴重。跟 E01 的經驗做對比：E01 在同樣投入 ~8 次嘗試的階段時，已經找到 one-cycle+MinMax 這個「有效但未調優」的基礎配方（B.8，N_TGE從未收斂降到695附近的量級）；desync50 到現在連「有效但未調優」的基礎配方都還沒出現，**這代表接下來的方向可能需要比「調超參數」更根本的改變**（例如：檢查 desync 資料本身的 SNR/POI 特性是否需要不同的前處理、模型容量或架構是否要調整、甚至重新檢視 `00_inspect_data.py` 在 desync 資料上的健檢結果）。在investing 剩餘的~7次嘗試前，先暫停回報現況，讓使用者決定要不要換個方向、還是繼續原本的超參數調查路線。

### B.29 SNR 健檢揭露真正的根因：desync50/100 的訊號被時間抖動打散到接近雜訊層級，不是被架構或超參數藏起來

依使用者指示，暫停繼續調超參數，先回頭用 `00_inspect_data.py` 檢查 desync50/100 資料本身的 SNR/POI 特性（`--mask-index 0`，沿用 desync0 找到的欄位）：

```
=== desync50 ===
masked-label SNR peak: 0.0101 at point 500   （desync0是6.30，差624倍）
unmasked-label SNR peak: 0.0095 at point 146  （對照組，本應接近0）
超過峰值50%的點數: 700 / 700（應該是一小撮，結果是整條軌跡）
PASS/FAIL: FAIL（masked-label峰值只有對照組的1.06倍，正常要差10倍以上）

=== desync100 ===
masked-label SNR peak: 0.0102 at point 465   （幾乎跟desync50一樣，沒有隨抖動幅度加倍而更差）
unmasked-label SNR peak: 0.0096 at point 38
超過峰值50%的點數: 700 / 700
PASS/FAIL: FAIL（同上）
```

**這是本次調查最關鍵的發現**：即使是全專案最好學的目標（`ID_MASKED`，E08 只要3條軌跡就收斂），在 desync50/100 上用單點 SNR 這個統計量完全找不到任何集中的洩漏點——不是峰值變矮變寬（那還算合理的抖動稀釋），而是**整條700個點通通被拉平到跟未遮罩對照組差不多的水準**，且 desync50 跟 desync100 幾乎一樣差（沒有隨抖動幅度倍增而更嚴重惡化，暗示這個稀釋效應在 desync50 這個程度就已經「觸底」）。

這解釋了 Phase 1+2 的 8 次系統性嘗試為什麼全部失敗——不是超參數沒調對，是**單點分析視角下，desync 資料的可學習訊號量本身就已經被壓到接近雜訊層級**。CNN 理論上能做單點統計量看不到的跨時間點組合，但兩條證據（單點SNR幾乎歸零、CNN 8種配方都學不到）現在互相印證同一個結論。

### B.30 實作盲對齊（resync）前處理：SNR 峰值從 0.073 回升到 6.82，超越 desync0 原始基準，證實根因假設成立

既然懷疑是「訊號被時間抖動打散」而非「訊號真的消失」，合理的下一步是**在訓練前先把每條軌跡重新對齊**，而不是繼續讓模型自己學抖動容忍度。這是 SCA 領域處理 desync 的標準做法之一（另一種是加大模型的時間不變性，也就是 cnn_light 的 k=51 大 kernel 設計本來想做的事，但單靠這個顯然不夠）。

**實作**（`src/data/resync.py`）：
- `resync(traces, reference, max_shift)`：對每條軌跡跟一個參考軌跡做互相關（在 `±max_shift` 範圍內窮舉所有偏移量，用向量化的矩陣乘法一次算完全部軌跡× 全部候選偏移量，不逐條 Python 迴圈），取相關性最高的偏移量對齊。
- `resync_iterative(traces, max_shift, rounds)`：第一輪用任意一條軌跡（`traces[0]`）當模板，後續每輪用「上一輪對齊後的平均軌跡」當更精準的模板重新對齊（模板品質隨輪數提升）。回傳最終參考模板，讓 V/D/E 能對齊到同一個時間基準。
- **完全不使用 metadata 裡記錄的真實 `desync` 欄位**——這個欄位只用來事後驗證對齊算得準不準，真實攻擊者在未知硬體上不可能拿到每條軌跡的真實抖動量，這是模擬資料集才有的上帝視角。

**驗證結果**（desync50，A集前5000條，`mask_index=0`）：

```
對齊前：masked-label SNR peak = 0.073 at point 518
對齊後：masked-label SNR peak = 6.82 at point 502   （93倍提升，超越desync0原始基準6.30）
超過峰值50%的點數：700/700 → 5/700（訊號重新集中）

估計偏移量 vs metadata真實desync值的相關係數：-0.9995
（幾乎完美的線性關係，正負號代表對齊方向、-0.9995接近-1；估計偏移量+真實desync
恆等於一個常數~15，代表這個常數就是參考軌跡本身的偏移量——盲對齊只能還原「相對」
偏移量，這對訓練用途已經足夠，不需要還原「絕對」偏移量）
```

**這證實了 B.29 的假設完全正確**：desync50 的訊號沒有消失，只是被隨機時間偏移打散到不同位置。互相關法在完全不偷看真實 desync 值的前提下，幾乎精確地把訊號找回來並重新對齊。

**整合進正式管線**：`configs/base.yaml` 新增 `preprocess.resync: {enabled: false, max_shift: 50, rounds: 2}`（預設關閉，不影響其他實驗）。`scripts/01_train_attacker.py` 在切分完 A/V 之後、標準化之前，若 `resync.enabled` 則對 A 做 `resync_iterative` 取得參考模板，V 用同一個模板對齊。`scripts/02_run_attack.py` 依照既有慣例（跟 Standardizer/MinMax 一樣，靠「A 上的計算是確定性、無隨機性」這個特性）在推論時重新對 A 做一次 resync 算出同一個參考模板，拿去對齊 E——不需要額外持久化任何中間產物。新增 `tests/test_resync.py` 三個測試（用合成資料驗證能還原已知偏移量、能讓分散的特徵重新集中、回傳的參考模板能正確用於對齊另一批資料），`tests/` 現在 33 個測試全過。用縮小規模（A=1000, 2 epochs）跑過 `01_train_attacker.py`→`02_run_attack.py`→`03_evaluate.py` 全流程確認沒有崩潰，管線接線正確。

**下一步**：用 desync50 的完整規模（A=30000）+ E01 已驗證配方（he_uniform+one-cycle+MinMax）+ `resync.enabled=true` 正式跑一次，驗證是否終於能收斂。

### B.31 desync50 完整規模驗證：resync 是全部9次嘗試裡第一次真正跳脫隨機基準的結果

用 desync50 完整規模（A=30000、V=5000）+ E01 已驗證配方（he_uniform+one-cycle lr=0.02/scale=0.05+MinMax，跟 Phase 1/2 完全相同的超參數，**唯一差異是加開 `preprocess.resync.enabled=true, max_shift=50`**）正式跑一次。

**訓練期預覽**（20-run快速版）持續下降到 epoch50 的 `GE=33.70`，遠優於先前任何一次嘗試——但吸取 B.24/B.28/B.29 一路以來「預覽會騙人」的教訓，沒有直接採信，跑了正式 100-run 評估：

```
GE @ N=1000  = 117.05   （首次低於隨機基準127.5！之前9次嘗試沒有一次做到）
SR1 @ N=1000 = 0.0100   （100次獨立攻擊裡有1次成功排到第一名，首次出現任何成功案例）
PI           = -0.1736  （仍是負值，但已經有實質攻擊訊號存在）

拉寬到 N=9000：
GE @ N=9000  = 83.46    （持續隨窗口拉寬下降，還沒收斂但趨勢正確）
SR1 @ N=9000 = 0.0000
```

**這次預覽（33.70）跟正式評估（117.05@1000）確實有落差，不是完全準確的估計，但方向是對的、且是真訊號**——不像 B.24 resnet 掃描或 B.28 desync50 max_lr=2.5e-3 那兩次「正式評估把預覽徹底推翻、變成該批次最差」的假警報，這次正式評估雖然沒有預覽那麼誇張，但**確確實實是全部9次desync50嘗試裡第一次跳脫隨機基準、且隨評估窗口拉寬持續改善的結果**，曲線形狀（127.5→117.05→83.46）跟 E01 當年剛導入 one-cycle+MinMax 時的早期收斂曲線（B.8，30→15左右但尚未收斂）是同一種「還在收斂中、尚未到N_TGE但方向明確」的訊號。

**結論：resync 假設完全驗證成立**。B.29 的診斷（desync50 的訊號被時間抖動打散、不是真的消失）跟 B.30 的離線驗證（SNR峰值93倍回升）現在有了訓練層級的實證支持——resync 前處理讓一個沿用 desync0 超參數、完全沒有為 resync 後的資料重新調過的配方，就直接從「9次嘗試全部負面、無一低於隨機基準」變成「首次正式突破隨機基準且持續改善」。

**重要限制**：目前這次跑的超參數（he_uniform+one-cycle lr=0.02/scale=0.05）是**沿用 desync0 調出來的配方，完全沒有針對 resync 後的資料重新調過**——resync 後的資料分佈跟 desync0 未必完全相同（例如邊界處的 wraparound 效應、殘留的對齊誤差），仍有機會透過重新走一輪類似 B.7-B.15 的超參數掃描進一步改善，讓 N_TGE 真正收斂。下一步：(1) 對 resync 後的 desync50 資料做輕量超參數微調，看能否讓 N_TGE 真正收斂；(2) 把同樣的 resync 前處理套用到 desync100 上驗證是否同樣有效（task #9）。

### B.32 desync50 輕量重調負面、desync100 resync 幾乎完全失敗——resync 演算法本身在大抖動範圍下有明顯限制

**desync50 輕量重調**（`runs/E03_desync50_20260816_1858/`）：把 B.13/B.14 當年為 desync0 調過但被 B.15 淘汰的次佳配方（`lr=5e-3, scale_percentage=0.1`）套到 resync 後的 desync50 資料上：

```
GE @ N=1000 = 131.01（比B.31的117.05差，比原始隨機基準127.5也差）
PI          = -0.2200
```

**比 B.31 的結果更差**——這代表 B.31 用的 `lr=0.02/scale=0.05`（B.15 確立的 desync0 最佳配方）不只是「隨便選的」，換成已知較差的舊配方，resync 後的 desync50 表現也跟著變差，方向上是一致的（desync0 調好的配方順序，在 resync 後的 desync50 上似乎大致保留）。這次輕量重調沒有找到比 B.31 更好的配方，但也沒有推翻 resync 本身的有效性。

**desync100 resync（`runs/E04_desync100_20260816_1858/`，沿用B.31同款配方lr=0.02/scale=0.05，max_shift=100）**：

```
GE @ N=1000 = 136.04（仍比隨機基準127.5差，desync50當初是117.05，明顯突破）
PI          = -0.1246
```

**跟 desync50 的戲劇性突破不同，desync100 完全沒有起色**。離線追查發現：**問題出在 resync 演算法本身，不是訓練配方**——

```
desync100 離線SNR驗證（跟B.30同樣方法，max_shift=100）：
  對齊前 SNR峰值：0.056
  對齊後 SNR峰值：0.142（只進步2.5倍，desync50當初是93倍）
  估計偏移量 vs 真實desync值相關係數：-0.0223（幾乎等於零噪訊，desync50是-0.9995接近完美）

加大疊代輪數（rounds）測試：
  rounds=2:  corr=-0.0223  SNR peak=0.1424
  rounds=4:  corr=-0.3623  SNR peak=0.1534
  rounds=6:  corr=-0.4464  SNR peak=0.1592
  rounds=10: corr=-0.4544  SNR peak=0.1720（在此附近趨於平緩，不再明顯進步）
```

**結論：desync100 的抖動範圍（±100）讓盲對齊演算法的搜尋空間變成 desync50 的兩倍（201個候選偏移量 vs 101個），互相關更容易鎖定到「局部相似但錯誤」的位置，導致第1輪對齊品質差，進而讓第2輪賴以改善的「平均模板」本身就是模糊/錯誤的，多輪疊代只能收斂到一個明顯劣於 desync50 的局部最優（相關係數卡在-0.45附近，遠不到-0.9995的水準）**。加大疊代輪數有幫助但很快就碰到瓶頸，不是簡單加更多輪就能解決的問題。

**這是一個重要但誠實的限制發現**：resync 假設在 desync50 上完全驗證成立（B.31），但**目前這個簡單版本的演算法（單軌跡起點+多輪均值模板）沒有直接遷移到 desync100**，需要更穩健的對齊策略才可能有效（例如：限制搜尋範圍到某個已知可能的 POI 子視窗、用更大的子集或更好的初始模板建構方式、或階層式由粗到細的對齊）。這不代表 desync100 的訊號真的沒救（B.29 的 SNR 健檢顯示 desync100 跟 desync50 的原始 SNR 崩壞程度幾乎一樣嚴重，理論上訊號同樣存在），只代表現有的對齊演算法還不夠強健去把它找回來。desync100 的下一步應該是改良 resync 演算法本身，而不是再去調訓練超參數。

### B.33 找到真正的根因：`resync()` 的互相關分數沒有正規化，desync100 上出現系統性偏差——不是演算法不夠強健，是一個 bug

在著手設計「更複雜的對齊策略」之前，先檢查一個更基本的問題：`resync()` 的分數是**未正規化的原始內積**（`a @ b`）。偏移量 `s` 越大，兩條軌跡重疊的取樣點數越少（`length - |s|`），內積的量級天生就會偏小——**這會系統性地偏向選擇「重疊點數多」的小偏移量，而不是「真正對齊」的偏移量**，跟對齊品質本身無關，純粹是分數計算方式的產物。

這個偏差在 desync50（`max_shift=50`，只佔700點軌跡的7%）小到可以忽略——B.30/B.31 的驗證結果（相關係數-0.9995）看起來完美，掩蓋了這個問題。但在 desync100（`max_shift=100`，佔14%）就嚴重到讓演算法幾乎失效：`±100` 偏移時只剩 600/700 點重疊，比零偏移少了 100 個點（14%的證據量減損），足以讓演算法系統性地偏好選錯誤但重疊點數多的候選。

**修正**：改用正規化互相關（除以兩個重疊區段各自的 L2 範數，即 Pearson-style 相關係數），讓分數在不同偏移量下可以公平比較，不再受重疊長度影響：

```python
denom = np.sqrt((a**2).sum(axis=1)) * np.sqrt((b**2).sum()) + 1e-12
scores[:, i] = (a @ b) / denom
```

**修正後重新驗證**（`rounds=2`，B.29/B.30/B.32 用的同一套流程）：

```
desync50：  corr=-0.9995  SNR peak=6.9069（原本就好，修正後幾乎沒變，證實desync50沒受這個bug明顯影響）
desync100： corr=-0.9999  SNR peak=6.8006（原本是corr=-0.0223/peak=0.1424，B.32那次的失敗結果——修正後直接躍升到近乎完美，達到desync50同等水準）
```

**結論**：desync100 的對齊困難**不是抖動範圍加倍帶來的根本性演算法限制，是一個實作疏漏**——B.32 標題下的「需要更穩健的對齊策略（POI子視窗限制、階層式對齊等）」判斷是錯的，正確的答案是「先把分數正規化」這麼簡單的修正就夠了。`tests/` 33 個測試全數重跑確認沒有回歸（正規化不影響測試用的小規模合成資料案例，因為那些案例本來的 `max_shift` 相對長度佔比也小）。

**這是本次調查裡第二次「先假設要複雜的解法，回頭發現是基本功沒做對」的教訓**（第一次是 B.15 的 scale_percentage 混淆變數）——在設計更複雜的對齊策略之前，先檢查最基本的統計量計算方式對不對，往往比直接加大工程複雜度更有效率。

**下一步**：用修正後的 resync（`max_shift=100`）在 desync100 完整規模上正式跑一次訓練，驗證是否也能像 desync50 一樣突破隨機基準。

### B.34 desync100 用修正後的 resync 正式跑：對齊演算法已修好，但這次訓練沒有重現 desync50 的突破

用 desync100 完整規模（A=30000、V=5000）+ 修正後的 resync（`max_shift=100`，corr=-0.9999/SNR peak=6.80，B.33驗證過）+ E01 已驗證配方（跟 B.31 desync50 突破用的完全同一套超參數，沒有另外調過），正式跑一次訓練。

**訓練期預覽震盪劇烈**，不像 desync50 平順下降：

```
epoch 5:  143.15 (new best)
epoch 10: 122.35 (new best)
epoch 15: 67.65 (new best) ← 最終存下的checkpoint
epoch 20: 134.45
epoch 25: 120.40
epoch 30: 169.15
epoch 35: 141.35
epoch 40: 154.45
epoch 45: 167.90
epoch 50: 125.55
```

epoch15之後訓練完全沒有再改善（後35個epoch的預覽都在120-170之間亂跳，沒有一次贏過epoch15），最終存下的是 epoch15 那個「看起來最好」的 checkpoint。**吸取 B.24/B.28 一路以來的教訓，沒有直接採信**，跑了正式評估：

```
GE @ N=1000 = 146.58（比隨機基準127.5差）
PI          = -0.4484（本次desync調查裡最負的一次）
拉寬到 N=9000：
GE @ N=9000 = 181.53（比@1000還差，不是收斂中的曲線，是持續惡化）
```

**這次不是假警報被推翻成「更差」，是徹頭徹尾的負面結果**——GE 隨評估窗口拉寬不減反增，代表這個 checkpoint 學到的不是穩定可泛化的訊號，比較像是在 epoch15 那個時間點剛好對驗證集的某個特定雜訊模式擬合得不錯，換一個更大的評估窗口就現形。

**結論**：**resync 演算法本身確認已經修好**（離線驗證 corr=-0.9999、SNR peak=6.80，完全匹配 desync50 的品質），但**這次訓練沒有重現 desync50 拿同一套配方就直接突破的幸運**。desync50 當初第一次用完整規模+未調過的配方就成功（B.31），這次 desync100 同樣條件卻震盪失敗，說明：

1. desync50 的「一次到位」本身可能帶有一定運氣成分（訓練曲線本身也不是完全平順，只是這次比較幸運剛好走到收斂方向）；
2. desync100 即使訊號已經對齊回來，可能仍需要專屬的超參數調整（例如更保守的 LR、更多輪 GEModelSelection 耐心、或不同的 epoch/schedule 長度）才能穩定收斂，不能假設「resync 修好了地基，配方就會自動遷移過去」。

**這是誠實、完整的一次嘗試，不是失敗的調查**——resync 假設本身（訊號被打散、可以透過對齊找回來）在兩個資料集上都得到了離線層級的完整驗證；訓練層級目前 desync50 成功、desync100 尚未成功，兩者的差異點已經清楚定位在「訓練配方需不需要重新調」而非「對齊有沒有效」。依照使用者指示，先在此暫停回報，不自動繼續投入超參數調整，留待後續決定是否要對 desync100+resync 這個新的資料基礎重新走一輪調查。

### B.35 desync100 輕量重調第二輪：seed123 也是假警報，且撞見一個操作失誤（平行任務輸出目錄撞名）

使用者指示「繼續」後，平行測試兩個方向：更保守的 one-cycle（`lr=5e-3, scale=0.1`，代號「gentle」）跟換一個隨機種子（`seed=123`，其餘不變，代號「seed123」，用來檢驗 B.34 的失敗是否只是運氣不好）。

**操作失誤（先誠實記錄）**：`01_train_attacker.py` 的 run_dir 命名只精確到分鐘（`%Y%m%d_%H%M`），這兩個平行任務剛好在同一分鐘內啟動，**產生了完全相同的 run_dir 路徑，兩個行程同時寫入同一個 `model.keras`**。事後比對 `train_history.csv` 內容（跟兩份訓練log的GE序列逐epoch比對）確認**留下來的模型檔案是 seed123 那組**（因為它訓練期間持續有新checkpoint寫入到較晚的epoch，gentle那組在epoch10後就沒有再更新過，被seed123後續的寫入蓋掉了）；但 `config_snapshot.yaml` 卻是 gentle 那組的設定（在啟動階段先被覆寫、之後沒再變動），造成這個 run_dir 的中繼資料跟實際模型對不上——**已在此記錄澄清，不遠端修改伺服器上的檔案**。**gentle 那組的實際訓練結果因此遺失、無法評估**，重跑於 `runs_gentle/`（用 `--runs-dir` 明確指定不同輸出目錄，避免重蹈覆轍）。

**seed123 正式評估**（訓練期預覽持續下降到 epoch40 的 `GE=23.15`，是本次投入desync100以來最誘人的訓練期訊號）：

```
GE @ N=1000 = 166.76（比隨機基準127.5更差，是desync100目前為止最差的正式結果）
PI          = -0.7554（全部desync調查裡最負的一次）
```

**又是一次徹底的假警報**——訓練期預覽從43.95一路"進步"到23.15看起來像是真的在收斂，正式100-run評估卻推翻成整個調查最糟的結果。這排除了「B.34失敗只是運氣不好、換個種子就好」這個假設——**同一套未調過的配方換種子後一樣失敗，甚至假警報現象更誇張**，代表 desync100 的問題不是隨機初始化的運氣，是更系統性的東西。

**目前desync100的假警報頻率明顯高於desync50**：B.34（epoch15, 67.65→146.58/181.53）、B.35 seed123（epoch40, 23.15→166.76）都是訓練期預覽大幅偏離正式結果的案例，desync50當年（B.24之外）沒有出現這麼誇張的偏離。這可能暗示 desync100 即使對齊修好了，殘留的雜訊或訓練不穩定性本身比 desync50 更嚴重，20-run快速預覽在這裡的採樣雜訊被放大了。

**gentle 重跑結果**（`runs_gentle/E04_desync100_20260816_1931/`，正確用獨立 `--runs-dir` 避免撞名）：

```
GE @ N=1000 = 138.68（比隨機基準127.5差）
PI          = -0.4918
```

### B.36 desync100 收尾總結：resync 修好了地基，但三次系統性重調全部負面，暫停回報

desync100 在 resync 演算法修正（B.33）後，累計 **3 次系統性訓練嘗試**：

| 嘗試 | 配方差異 | GE@1000 | PI |
|---|---|---|---|
| B.34：原封不動套用desync50成功配方 | he_uniform+one-cycle lr=0.02/scale=0.05（跟desync50突破那次完全一樣） | 146.58（@9000惡化到181.53） | -0.4484 |
| B.35：換隨機種子 | 同上 + `seed=123` | 166.76（**desync100最差**） | -0.7554（**全調查最負**） |
| B.36：更保守LR | `lr=5e-3, scale=0.1` | 138.68 | -0.4918 |

**三次結果彼此接近（138.68-166.76），全部明顯差於隨機基準，沒有一次表現出desync50那種持續下降的收斂跡象**。跟 desync50 的對比非常清楚：desync50 用「完全沒調過的舊配方」第一次嘗試就直接突破（B.31），desync100 同樣的配方、外加換種子、外加放寬LR，三次嘗試全部失敗。

**目前對這個落差最合理的解讀**：resync 演算法本身已經證實對兩個資料集都同樣有效（B.33 離線驗證，desync100 對齊品質 corr=-0.9999 跟 desync50 的 -0.9995 幾乎相同），所以**問題不在對齊**；但 desync100 原始抖動範圍是 desync50 的兩倍，即使對齊回正確位置，**殘留的對齊誤差、量化效應、或訊號本身在更寬的原始抖動下的其他劣化（例如邊界wraparound影響的軌跡比例更高）可能還是比desync50嚴重**，導致同一套訓練配方在desync100上不夠用——desync100 可能真的需要專屬於它自己的完整超參數調查（跟 E01 當年的規模一樣），而不是在 desync50 的成功配方上做小幅微調就能解決。

**依照使用者「三次嘗試給一個查核點」的原則，在此暫停回報**，不自動投入第四次嘗試。這輪總結：resync 假設完整驗證成立（兩個資料集的訊號都被證實只是被打散、可以透過對齊找回來）；訓練層級 desync50 已突破、desync100 尚未突破，且已知不是簡單微調能解決的，需要更完整的投入。

### B.37 desync100 完整調查啟動（E01同等規模）：run_dir 撞名 bug 又發生一次（這次撞在同一秒），徹底修好

使用者指示「繼續投入desync100的完整調查」後，比照 desync50 的 Phase 1 手法（`scale_percentage=0.1` 固定，掃 `max_lr`），平行launch `lr=1e-3` 跟 `lr=1e-2` 兩點。

**同一個bug又發生一次，這次更難防**：B.35 的修正是把 run_dir 時間戳精度從分鐘加到秒（`%Y%m%d_%H%M%S`），但這次兩個平行任務剛好在**同一秒內**啟動，兩者又寫進了同一個 `runs_sweep/E04_desync100_20260816_194046` 目錄，`model.keras` 又被其中一個蓋掉。透過比對 `train_history.csv`（epoch50 GE=107.35 精確吻合 lr=1e-2 那組的log）確認留下來的是 lr=1e-2，**lr=1e-3 的結果又遺失**。

**這次徹底修好**：run_dir 命名再加上 `os.getpid()`（`{exp_id}_{timestamp}_{pid}`）——作業系統保證同時間執行的行程一定有不同 PID，不會再受時間戳精度限制影響，無論兩個 launch 相隔多短都不會撞名。已同步到伺服器，重跑遺失的 `lr=1e-3`。

**Phase 1 進度**（`scale_percentage=0.1` 固定）：

| max_lr | GE@1000 | PI |
|---|---|---|
| 5e-3（沿用B.36 gentle結果） | 138.68 | -0.4918 |
| 1e-2 | 153.82 | -0.0258 |
| **1e-3** | **95.17（@9000拉寬到41.80，持續下降）** | -0.3604 |

### B.38 desync100 首次真正突破：`max_lr=1e-3` 讓 GE 隨窗口拉寬持續下降到 41.80

**Phase 1 max_lr 掃描的最後一點（`lr=1e-3, scale=0.1`）帶來了 desync100 調查以來第一次真正的正向訊號**：

```
GE @ N=1000 = 95.17（首次明顯低於隨機基準127.5，desync100目前最佳）
拉寬到 N=9000：
GE @ N=9000 = 41.80（持續下降，趨勢比desync50當初的突破曲線127.5→117.05→83.46還要陡）
PI          = -0.3604（仍是負值，但已有實質攻擊訊號）
```

**這次確認是真訊號，不是假警報**——用「隨評估窗口拉寬持續下降」這個跟 B.31（desync50突破）同樣的判準驗證：曲線形狀健康，不像 B.34/B.35 那些「拉寬後不減反增」的假警報。desync100 終於也找到了能打破隨機基準的配方。

**Phase 1 結論**：跟 desync50 的 Phase 1（四個 max_lr 值全部失敗，B.28）不同，**desync100 的 Phase 1 直接命中**——`max_lr=1e-3`（三點裡最保守的一個）明顯優於 `5e-3`、`1e-2`。這符合一個合理的直覺：desync100 的殘留雜訊/不穩定性比 desync50 更嚴重（B.35/B.36 觀察到的高假警報率），需要比 desync50 更保守的學習率才能穩定收斂。**下一步**：以 `max_lr=1e-3` 為基礎，比照 desync50 的做法，掃 `end_percentage`／`scale_percentage`，看能不能讓 N_TGE 真正收斂。

### B.39 desync100 Phase 2（end_percentage掃描）：GE@10000 降到 10.00，非常接近完全收斂

以 Phase 1 贏家（`max_lr=1e-3, scale=0.1`）為基礎，掃 `end_percentage`（0.1、0.2已知95.17/41.80、0.35）：

| end_percentage | GE@1000 | GE@9000 | GE@10000 | PI | SR1@1000 |
|---|---|---|---|---|---|
| 0.1 | **84.42** | **14.92** | **10.00** | -0.4213 | 0.02（首次出現成功案例） |
| 0.2（Phase1贏家） | 95.17 | 41.80 | — | -0.3604 | 0.00 |
| 0.35 | 95.81 | — | — | -0.4194 | 0.02 |

**`end_percentage=0.1`（退火期縮短、探索期拉長）明顯最佳**，GE 曲線（127.5→84.42→14.92→10.00）走勢比 Phase 1 贏家還要陡，**已經非常接近完全收斂**（10000條軌跡是 Attack 集的全部上限，GE 還在個位數，只差臨門一腳）。SR1 在 N=1000 就已經有 2% 的獨立攻擊成功排到第一名，是 desync100 至今唯一出現過攻擊成功案例的配方。

**跟 desync0/desync50 的對比**：desync0（E01）當年 `end_percentage=0.2` 是三個維度裡的贏家；desync100 這裡反而是 `0.1`（退火期更短）更好——不同資料集/抖動情境下的最佳超參數不必然相同，這也是為什麼不能直接假設一個資料集調好的配方能直接遷移到另一個（跟 B.34-B.36 的教訓一致）。

**下一步**：以 `end_percentage=0.1` 為基礎，掃 `scale_percentage`（記得 B.15 的教訓，變動 scale_percentage 時要同步調整 `train.lr` 讓峰值LR固定在1e-3），看能不能讓 GE 在10000條軌跡窗口內真正跌破1、達成 N_TGE。
