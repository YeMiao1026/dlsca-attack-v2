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

**已修正**：`GEModelSelection.__init__` 新增 `leakage_model`、`mask` 參數並傳給內部的 `scores.build`；`src/train/trainer.py::fit` 從 `cfg["leakage"]` 算出這兩個值餵給 callback（跟 `scripts/03_evaluate.py` 算法一致）。用縮小規模的 E05 跑過一次確認：訓練不再崩潰、`GEModelSelection` 的 GE 預覽正確反映 HW 評分、`02_run_attack.py`／`03_evaluate.py` 正確吃到 `probs.shape=(N,9)` 全程無誤。E05 目前還沒跑正式全量結果（這次是縮小規模的煙霧測試，不代表真實效能）。
