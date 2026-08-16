# `runs/` 索引

`runs/` 目錄本身被 `.gitignore` 排除（訓練產物不進版控），這份索引移到 `docs/` 底下所以會進版控，作為導覽用，不是正式產物本身。每個子目錄的完整細節、調查脈絡見 `CLAUDE.md` 附錄 B.7–B.20；這裡只列重點方便快速定位。

## E01 系列（cnn_light、desync0、ID leakage、無 augmentation 的不同訓練配方）

| run_dir | 配方 | 評估窗口 | N_TGE | GE@窗口末端 | 備註 |
|---|---|---|---|---|---|
| `E01_baseline_clean_20260815_2215` | epochs=100，固定 Adam lr=1e-3，batch=128，lecun_normal，只標準化 | 1000 | None | 38.97 | 第一次嘗試，撞 epoch 上限時還在進步中被砍斷 |
| `E01_baseline_clean_20260815_2228` | epochs=500，固定LR，batch=128，lecun_normal，早停@175(最佳@145) | 1000 | None | 29.86 | 證實「單純加 epoch」沒用，GE 曲線已平緩 |
| `E01_baseline_clean_20260815_2256` | epochs=50，**one-cycle**，batch=50，lecun_normal，+MinMax | 1000 | None | 14.89 | one-cycle 首次見效，GE 曲線在窗口末端仍在下降 |
| `E01_baseline_clean_20260815_2312` | epochs=**150**，one-cycle，batch=50，lecun_normal，+MinMax | 5000 | None | 49.93 | 負面對照組：拉長 one-cycle 排程反而更差，證實 epochs=50 是跟排程長度綁定調過的 |
| `E01_baseline_clean_20260815_2336` | epochs=50，one-cycle，batch=50，lecun_normal，+MinMax（跟2256同配方，重跑拉寬評估窗口） | 5000 → 9000 | None → **6408** | 1.96 → 0.0 | `metrics.json`=9000條的最終結果；`metrics_maxtraces5000.json`=中途5000條的快照（兩者都是同一個模型，只是評估窗口不同，用 `03_evaluate.py --override` 重算，沒有重新訓練） |
| `E01_repro_original_recipe_20260816_0017` | **診斷用**，epochs=50，固定LR，batch=50，he_uniform，**無正規化**，A=44000 | 9000 | None | 199.27 | 重現歷史原始碼 `train_cnnd.py` 的確切配方，證實其「無正規化+固定LR」在嚴謹評估法下根本沒訓練起來（loss全程卡在隨機基準附近） |
| `E01_baseline_clean_20260816_0027` | epochs=50，one-cycle（`end_percentage=0.2`），batch=50，**he_uniform**，+MinMax | 9000 | **695** | 0.0 | **目前最佳結果**。跟 2336 唯一差異是初始化方式，N_TGE 進步 9.2 倍 |
| `E01_baseline_clean_20260816_1212` | 同上，`end_percentage`拉高到**0.35**（退火期拉長） | 9000 | 4253 | 0.0 | 負面：退火期拉長反而讓探索階段被壓縮，比 end_percentage=0.2 差6倍 |
| `E01_baseline_clean_20260816_1217` | 同上，`end_percentage`降到**0.1**（退火期縮短） | 9000 | None | 60.93 | 負面：完全沒收斂，比0.2更差，證實0.2是兩個方向都更差的局部最優 |
| `E01_baseline_clean_20260816_1232` | 同 0027 基準，`max_lr`拉高到**1e-2**（2倍） | 9000 | None | 167.93 | 負面：太激進，比隨機基準還糟 |
| `E01_baseline_clean_20260816_1236` | 同 0027 基準，`max_lr`降到**2.5e-3**（一半） | 9000 | None | 55.24 | 負面：太保守，卡在55左右不再進步 |
| `E01_baseline_clean_20260816_1248` | 同 0027 基準，`scale_percentage`降到**0.05**（**峰值LR沒固定，混淆實驗**） | 9000 | None | 128.17 | ⚠️ 峰值實際跌到1.25e-3，不是乾淨的scale_percentage測試，僅供對照 |
| `E01_baseline_clean_20260816_1255` | 同 0027 基準，`scale_percentage`拉高到**0.2**（**峰值LR沒固定，混淆實驗**） | 9000 | None | 215.42 | ⚠️ 峰值實際衝到2e-2，不是乾淨的scale_percentage測試，僅供對照 |
| `E01_baseline_clean_20260816_1302` | `lr=0.02, scale_percentage=0.05`（**峰值修正固定在5e-3**） | 9000 | **475** | 0.0 | **新的最佳結果**（修正混淆變數後）。比0027再進步32% |
| `E01_baseline_clean_20260816_1308` | `lr=1.25e-3, scale_percentage=0.2`（**峰值修正固定在5e-3**） | 9000 | None | 38.35 | 負面：擺動幅度變寬沒有幫助，確認 scale_percentage 越小越好的單調趨勢 |

## 目前最佳配方（截至此份索引）

`one-cycle LR (實際峰值LR=5e-3, end_percentage=0.2, scale_percentage=0.05, 對應 train.lr=0.02) + MinMaxScaler + he_uniform + batch=50 + epochs=50`，對應 `configs/model/cnn_light.yaml` + `configs/exp/E01_baseline_clean.yaml` 目前的內容。`N_TGE=475`，離 CLAUDE.md §11 目標（100±30）差 4.75 倍。

**重要提醒**：`OneCycleLR` 的實際峰值 LR = `train.lr × 100 × scale_percentage²`，只有 `scale_percentage=0.1` 時才會化簡成「峰值=train.lr」。單獨改 `scale_percentage` 卻不同步調整 `train.lr` 會意外把峰值也拉走（1248、1255 兩筆就是這樣的混淆實驗，已標註 ⚠️，不能當作乾淨的 scale_percentage 結論）。`end_percentage`（0.1/0.2/0.35）跟「峰值固定情況下的」`max_lr`／`scale_percentage` 都掃過，`end_percentage`、`max_lr` 兩個維度是局部最優（兩側都更差），但 `scale_percentage`（峰值固定後）呈現單調趨勢、還沒摸到反轉邊界，理論上還能繼續往更小的方向試（細節見 CLAUDE.md 附錄 B.13–B.15）。

## E02-E08 首批結果（沿用 E01 已驗證的 cnn_light 配方，A=30000/V=5000, epochs=50）

| run_dir | 實驗 | N_TGE | GE@1000 | 備註 |
|---|---|---|---|---|
| `E03_desync50_20260816_1337` | desync50 | None | 152.31（比隨機127.5更差） | 負面，直接沿用 desync0 調出的超參數沒學到東西 |
| `E03_desync50_20260816_1359` | desync50，**改用遮罩已知標籤**（`--override leakage.model=ID_MASKED leakage.mask_index=0`，其餘同上） | None | 157.14（一樣比隨機更差） | **鑑別診斷用**：連 E08 那種「全專案最好學」的目標放到 desync50 上都學不到，排除「ID 目標天生難」的解釋，確認是 desync 抖動讓現有 one-cycle 配方失效，見 CLAUDE.md 附錄 B.18 |
| `E03_desync50_20260816_1718` | 精簡掃描#1：放寬one-cycle幅度（scale=0.1，peak仍5e-3） | None | 147.07（跟152.31差不多） | 負面，PI=-0.0334。見 B.25 |
| `E03_desync50_20260816_1728` | 精簡掃描#2：拿掉one-cycle，改flat Adam lr=1e-3 | None | 144.82 | 負面，PI=-0.0471。訓練期預覽一度看似有希望（GE=90.25）但正式評估沒有兌現，跟其餘兩次都在雜訊範圍內。**三次嘗試皆未跳脫噪聲層級，desync50需要E01等級的完整獨立調查**。見 B.25 |

## desync50 完整調查（task #8，E01同等規模）

**Phase 1：max_lr掃描**（scale_percentage=0.1固定，其餘沿用cnn_light預設）

| run_dir | max_lr | GE@1000 | PI | 備註 |
|---|---|---|---|---|
| `E03_desync50_20260816_1809` | 1e-3 | 139.84 | -0.0683 | |
| `E03_desync50_20260816_1812` | 2.5e-3 | 171.49 | -0.1010 | 訓練期預覽GE=86.70看似有希望，正式評估推翻——四點最差 |
| （沿用B.25 `1718`） | 5e-3 | 147.07 | -0.0334 | |
| `E03_desync50_20260816_1820` | 1e-2 | 152.97 | -0.0259 | |

四點全在139.84-171.49雜訊層級，無一收斂。**Phase 1結論：max_lr不是缺的拼圖**，轉向Phase 2（噪訊增強+desync50組合，新假設）。細節見 CLAUDE.md 附錄 B.28。

**Phase 2：噪訊增強+desync50組合**（E01配方lr=0.02/scale=0.05為基礎+augment）

| run_dir | sigma_ratio | 預覽GE | 正式GE@1000 | PI |
|---|---|---|---|---|
| `E03_desync50_20260816_1828` | 0.5 | 86.50 | 161.72（假警報） | -0.0934 |
| `E03_desync50_20260816_1829` | 1.0 | 96.40 | 180.23（假警報，目前desync50調查最差結果） | -0.0412 |

兩次訓練期預覽皆被正式評估推翻，噪訊增強假設不成立。**desync50累計8次系統性嘗試（Phase1的4個max_lr + B.25的2個LR-schedule off + Phase2的2個增強變體）全數負面**，已暫停回報使用者是否換方向。細節見 CLAUDE.md 附錄 B.28。

## desync50 根因診斷 + resync 前處理（B.29-B.31）

`00_inspect_data.py` 健檢揭露：desync50/100 的per-point SNR即使對最好學的遮罩已知標籤也幾乎完全歸零（0.0101 vs 對照組0.0095，700/700個點都在峰值50%以上，無局部化POI）——這才是8次超參數嘗試全部失敗的真正原因，不是超參數沒調對。細節見 B.29。

新增 `src/data/resync.py`（互相關盲對齊，完全不用metadata的真實desync值），離線驗證：SNR峰值從0.073回升到6.82（93倍，超越desync0原始基準6.30），估計偏移量跟真實desync值相關係數-0.9995。細節見 B.30。

| run_dir | 配方 | GE@1000 | GE@9000 | SR1@1000 | PI | 備註 |
|---|---|---|---|---|---|---|
| `E03_desync50_20260816_1849` | E01配方（he_uniform+one-cycle+MinMax，沒為resync重調）+ **resync.enabled=true, max_shift=50** | **117.05**（首次低於隨機127.5） | **83.46**（持續下降） | **0.01**（首次成功案例） | -0.1736 | **全部9次desync50嘗試裡第一次真正突破隨機基準**，曲線形狀類似E01早期收斂曲線，尚未到N_TGE但方向明確。細節見 CLAUDE.md 附錄 B.31 |
| `E03_desync50_20260816_1858` | resync + 輕量重調（lr=5e-3, scale=0.1，desync0時代B.15淘汰的次佳配方） | 131.01（比1849差） | — | 0.00 | -0.2200 | 負面，不如B.31的配方，方向上跟desync0調參結果一致。見 B.32 |
| `E04_desync100_20260816_1858` | resync（max_shift=100，**舊版未正規化互相關，有bug**）+ 同B.31配方 | 136.04（仍差於隨機） | — | 0.00 | -0.1246 | **negative，resync演算法本身在desync100上幾乎失敗**（估計偏移量跟真實desync相關係數只有-0.0223，desync50是-0.9995）。事後查出根因是分數沒正規化，見B.33修正 |
| `E04_desync100_20260816_1910` | resync（max_shift=100，**修正後的正規化互相關**，corr=-0.9999完全修好）+ 同B.31配方 | 146.58（比隨機差） | 181.53（拉寬後更差，非收斂中曲線） | 0.00 | -0.4484 | **對齊演算法確認修好，但這次訓練沒重現desync50的突破**。訓練期預覽在epoch15震盪出一個false alarm（67.65），正式評估徹底推翻。見 CLAUDE.md 附錄 B.34 |
| `E04_desync100_20260816_1921`（seed123，⚠️輸出目錄跟gentle撞名，config_snapshot.yaml不準，已用train_history.csv比對確認模型身分） | resync + 同B.31配方 + `seed=123` | 166.76（desync100最差） | — | 0.00 | -0.7554（**全調查最負**） | 排除「B.34只是運氣不好」的假設。見 CLAUDE.md 附錄 B.35 |
| `E04_desync100_20260816_1931`（`runs_gentle/`，獨立目錄避免撞名） | resync + `lr=5e-3, scale=0.1` | 138.68 | — | 0.00 | -0.4918 | 見 CLAUDE.md 附錄 B.36 |

**desync100 resync後3次系統性重調（B.34-B.36）全部負面**，暫停回報後使用者指示繼續投入完整調查（task #10）。

## cnn_best（E06）完整調查（task #11）Phase 1：one-cycle峰值掃描

| peak | GE@1000 | PI |
|---|---|---|
| 1e-4 | 163.33 | -0.0183 |
| 1e-3 | 158.68 | -0.0273 |
| 1e-5 | 152.82 | -0.0257 |
| flat基準（B.27） | 157.77 | -0.0241 |

**四點全部無法區分，one-cycle對cnn_best完全沒有幫助**——跟cnn_light/desync100的經驗明顯不同。已暫停（非結案），使用者指示轉去resnet（task #12），見 CLAUDE.md 附錄 B.43-B.44。

## resnet（E07）完整調查（task #12）Phase 1：max_lr掃描

| peak | GE@1000 | GE@9000 | PI |
|---|---|---|---|
| **1e-4** | **107.87** | **70.93** | -0.0243 |
| 2.5e-4 | 116.82 | — | -0.0247 |
| 1e-3（既有,B.24） | 131.34 | — | -0.0268 |
| 5e-3（既有,B.24,假警報） | 174.63 | — | -0.1472 |

**`lr=1e-4` 首次真正突破隨機基準，且曲線健康持續下降**，呈現peak越小越好的單調趨勢。見 CLAUDE.md 附錄 B.45。

## resnet（E07）完整調查收尾（task #12）

| 階段 | 最佳設定 | GE@1000 | GE@9000 |
|---|---|---|---|
| 起點（B.24） | Adam lr=1e-3 flat | 123.27 | 未評估 |
| Phase1 max_lr | lr=5e-5 | 102.69 | 56.21 |
| Phase2 end_percentage | 0.1 | 101.17 | 51.32 |
| Phase3 scale_percentage | 0.1（預設） | 101.17 | 51.32 |
| Phase4 epochs | 100（預設） | 101.17 | 51.32 |

**四維度全部確認局部最優，最終結果 GE@9000=51.32**，唯一通過「隨窗口拉寬持續下降」健全性檢查的resnet配方。約14-15次系統性訓練，深度跟desync100調查相當。細節見 CLAUDE.md 附錄 B.45-B.49。task #12 在此告一段落，回報使用者。

## desync100 完整調查（task #10，E01同等規模）Phase 1：max_lr掃描

| run_dir | max_lr | GE@1000 | GE@9000 | PI | 備註 |
|---|---|---|---|---|---|
| （沿用B.36 gentle） | 5e-3 | 138.68 | — | -0.4918 | 負面 |
| `E04_desync100_20260816_194046`（lr1e2） | 1e-2 | 153.82 | — | -0.0258 | 負面，⚠️此目錄一度跟lr1e3撞名（同一秒啟動），已用train_history.csv比對確認身分 |
| `E04_desync100_20260816_195051_373600`（lr1e3） | **1e-3** | **95.17** | **41.80（持續下降）** | -0.3604 | **desync100首次真正突破，曲線形狀健康非假警報**。見 CLAUDE.md 附錄 B.38 |

**Phase 1 直接命中**（跟desync50 Phase 1全數失敗不同）：`lr=1e-3`（三點裡最保守）明顯優於`5e-3`/`1e-2`。下一步：以此為基礎掃`end_percentage`/`scale_percentage`。

**run_dir 撞名 bug 修過兩次**：第一次修分鐘→秒精度（B.35）不夠，第二次加上PID才徹底解決（B.37），`scripts/01_train_attacker.py` 現在的 run_dir 格式是 `{exp_id}_{timestamp}_{pid}`。

## desync100 Phase 2：end_percentage掃描

| run_dir | end_percentage | GE@1000 | GE@9000 | GE@10000 | PI | SR1@1000 |
|---|---|---|---|---|---|---|
| `E04_desync100_20260816_200044_391859` | **0.1** | **84.42** | **14.92** | **10.00** | -0.4213 | 0.02 |
| （沿用Phase1贏家） | 0.2 | 95.17 | 41.80 | — | -0.3604 | 0.00 |
| `E04_desync100_20260816_200044_391857` | 0.35 | 95.81 | — | — | -0.4194 | 0.02 |

**`end_percentage=0.1` 目前最佳，GE@10000=10.00，非常接近完全收斂**。細節見 CLAUDE.md 附錄 B.39。

## desync100 Phase 3：scale_percentage掃描（峰值固定1e-3）

| run_dir | scale_percentage | train.lr | GE@1000 | GE@10000 | PI |
|---|---|---|---|---|---|
| `..._201059_426350` | 0.05 | 4.0e-3 | 126.39 | — | -0.8260 |
| （沿用Phase2贏家） | 0.1 | 1.0e-3 | 84.42 | 10.00 | -0.4213 |
| `E04_desync100_20260816_201058_scale02` | **0.2** | 2.5e-4 | **69.97** | **7.00** | -0.6678 |

**跟desync0方向相反**（desync0是scale越小越好），desync100目前是越大越好，但 `scale=0.3`（`E04_desync100_20260816_201955_scale03`）反轉變差（GE@1000=87.32），確認 `0.2` 是局部最優。細節見 CLAUDE.md 附錄 B.40-B.41。

## desync100 目前最佳配方（截至此份索引）

`max_lr=1e-3（peak）, end_percentage=0.1, scale_percentage=0.2, epochs=50, train.lr=2.5e-4` + resync（max_shift=100）。**GE@10000=7.00**，非常接近完全收斂但尚未跌破1。跟desync50最佳配方（`lr=0.02, end_percentage=0.2, scale_percentage=0.05`）幾乎每個維度都不同。**四個超參數維度**（max_lr/end_percentage/scale_percentage/epochs）都已找到局部最優，見 CLAUDE.md 附錄 B.41-B.42 總結。

## desync100 Phase 4：epoch數/schedule長度掃描

| epochs | GE@1000 | GE@10000 | PI |
|---|---|---|---|
| 30 | 82.46 | 11.00 | -0.2568 |
| **50（贏家）** | **69.97** | **7.00** | -0.6678 |
| 75 | 111.71 | — | -0.0500 |

確認50 epochs也是局部最優，四維度全數摸到邊界。見 CLAUDE.md 附錄 B.42（含一次操作插曲：工具呼叫拒絕後底層SSH仍執行，造成重複訓練，已清理無資料損毀）。
| `E04_desync100_20260816_1343` | desync100 | None | 168.39 | 同樣負面，抖動更大更沒學到 |
| `E04_desync100_20260816_1738` | 單次確認跑（沿用desync50表現較好的flat LR，非完整掃描——scope說明見B.26） | None | 138.43（比168.39好一些） | 負面，PI=-0.0795仍是負值，同desync50模式再現。細節見 CLAUDE.md 附錄 B.26 |
| `E08_masked_label_20260816_1349` | 遮罩已知標籤，desync0 | **3** | 0.0 | **本專案所有實驗裡最快收斂**。初次評估異常（GE不收斂但loss明顯在降）追出 `scores.build` 沒處理mask的真bug，修正後才是這個數字，見 CLAUDE.md 附錄 B.17 |
| `E05_hw_leakage_20260816_1415` | HW 洩漏模型，desync0（9類） | **1361**（評估窗口拉到3000） | 0.0 @3000 | **第二快收斂**，比 E01 的 ID 目標（N_TGE=475）快3倍以上，符合 SCA 文獻對 HW 模型的一般認知。B.19 修完 `scores.build`/`GEModelSelection` 的 HW 支援後跑出來的正式結果 |

E03/E04 的負面結果經 B.18 鑑別診斷後，確認根因是「desync0 調出來的 one-cycle 超參數不適用於 desync 情境」，不是資料或管線問題，也不是 ID 目標本身難學。要解決大機率需要對 desync50/100 各自重新走一輪跟 B.7-B.15 同等規模的調查，尚未投入。E07（resnet，見B.20）模型已實作完成、能訓練+評估，但還沒跑正式全量結果、也還沒為這個架構調過 one-cycle 之類的訓練方法論；E02（噪訊增強）還缺動態增強的訓練迴圈才能跑。

## E06（cnn_best）：CPU 環境下的縮短版負面結果

| run_dir | 配方 | N_TGE | GE@1000 | 備註 |
|---|---|---|---|---|
| `E06_cnn_best_20260816_1452` | **CPU環境縮短版12epoch，非原論文75epoch**，RMSprop lr=1e-5，batch=200，desync0，ID leakage | None | 162.39（比隨機127.5更差） | 原論文75epoch在這台無GPU機器要6.6小時，經使用者同意縮成12epoch(~1小時)拿誠實中間資料點。loss全程幾乎沒動（5.5452→5.5425，隨機基準5.545），PI=-0.0192（幾乎零資訊）。**不能代表cnn_best真實能力**，純粹是RMSprop lr=1e-5配合75epoch才收斂、12epoch連基準線都還沒真正脫離。細節見 CLAUDE.md 附錄 B.21 |
| `E06_cnn_best_20260816_1801` | **實驗室GPU server上完整跑**（epoch45觸發patience早停，未到75上限，非資源限制），RMSprop lr=1e-5，batch=200 | None | 157.77（跟12epoch版162.39幾乎沒差） | **關鍵釐清**：完整訓練預算跑完後結果跟縮短版幾乎一樣，PI=-0.0241（跟-0.0192幾乎沒差）——**確認問題是配方本身學不到東西，不是訓練時間不夠**。GPU單epoch只要2-4秒（CPU是317秒，快100倍）。細節見 CLAUDE.md 附錄 B.27 |

## E02（噪訊增強）：目前全專案最佳真實攻擊結果

| run_dir | 配方 | N_TGE | GE@1000 | PI | 備註 |
|---|---|---|---|---|---|
| `E02_noisy_augment_20260816_1618` | 沿用E01配方（he_uniform+one-cycle+MinMax）+動態高斯噪訊增強（sigma_ratio=0.5，每epoch重新生成） | **206**（比E01的475快2.3倍） | 0.0 | **+0.1796**（E01是-0.4999，由負轉正） | **目前全專案已知最佳真實攻擊結果**，取代E01成為新基準。N_TGE跟PI兩個獨立指標同時大幅進步，不是評估窗口挑出來的表面改善。細節見 CLAUDE.md 附錄 B.23 |

## E07（resnet）：完整訓練預算下的負面結果

| run_dir | 配方 | N_TGE | GE@1000 | 備註 |
|---|---|---|---|---|
| `E07_resnet_20260816_1559` | **完整跑滿**，Adam lr=1e-3，batch=128，epochs=100（epoch50觸發patience早停，非資源限制），desync0，ID leakage | None | 123.27（接近隨機127.5，略低） | 跟E06不同，這是誠實完整的負面結果，不是縮短版。loss有下降（5.5547→5.4451）但幅度不足以讓攻擊收斂，PI=-0.1129（略負）。初步判讀：resnet架構完全沒調過訓練方法論（one-cycle/初始化等），跟E03/E04一樣「還沒調」不等於「做不到」。細節見 CLAUDE.md 附錄 B.22 |
| `E07_resnet_20260816_1630` | 精簡掃描#1：+One-Cycle LR，peak維持1e-3 | None | 131.34（比baseline還略差） | 負面，PI=-0.0268。單純加One-Cycle、peak不變幾乎沒差。細節見 B.24 |
| `E07_resnet_20260816_1652` | 精簡掃描#2：+One-Cycle LR，peak拉高到5e-3（同cnn_light） | None | **174.63（三者中最差）** | 負面，PI=-0.1472（三者中最負）。**訓練期20-run快速預覽一度顯示GE低至87.60（看似有希望，判定為new best存檔），但正式100-run評估推翻——是「少量run預覽被雜訊騙到」陷阱的活生生示範，這次騙到的是GEModelSelection本身**。細節見 B.24 |

## cnn_best（E06）完整調查收尾（task #11）

| 嘗試 | GE@1000 |
|---|---|
| flat基準（lr=1e-5） | 157.77 |
| one-cycle peak=1e-4 | 163.33 |
| one-cycle peak=1e-3 | 158.68 |
| one-cycle peak=1e-5 | 152.82 |
| flat lr=1e-3 | 159.51 |
| batch_size=50 | 158.91（訓練期預覽116.80看似有希望，正式評估推翻） |

**六個數字全部落在152.82-163.33窄範圍內，統計上無法區分**——LR schedule形狀、峰值大小（4個數量級）、batch size全部試過都無效。判定為66.6M參數對30000條訓練軌跡的優化困難本身，超出超參數調整範疇。task #11 結案（暫停），細節見 CLAUDE.md 附錄 B.43-B.50。

## 命名說明

- 目錄名格式 `{exp_id}_{timestamp}`，`exp_id` 對應 `configs/exp/*.yaml` 的 `exp_id` 欄位。
- `E01_repro_original_recipe` 不在 CLAUDE.md §8.2 的官方 E01-E08 編號內，是這次調查歷史數字來源時的診斷用實驗，config 檔案本身也有註記說明。
