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
| `E04_desync100_20260816_1343` | desync100 | None | 168.39 | 同樣負面，抖動更大更沒學到 |
| `E08_masked_label_20260816_1349` | 遮罩已知標籤，desync0 | **3** | 0.0 | **本專案所有實驗裡最快收斂**。初次評估異常（GE不收斂但loss明顯在降）追出 `scores.build` 沒處理mask的真bug，修正後才是這個數字，見 CLAUDE.md 附錄 B.17 |
| `E05_hw_leakage_20260816_1415` | HW 洩漏模型，desync0（9類） | **1361**（評估窗口拉到3000） | 0.0 @3000 | **第二快收斂**，比 E01 的 ID 目標（N_TGE=475）快3倍以上，符合 SCA 文獻對 HW 模型的一般認知。B.19 修完 `scores.build`/`GEModelSelection` 的 HW 支援後跑出來的正式結果 |

E03/E04 的負面結果經 B.18 鑑別診斷後，確認根因是「desync0 調出來的 one-cycle 超參數不適用於 desync 情境」，不是資料或管線問題，也不是 ID 目標本身難學。要解決大機率需要對 desync50/100 各自重新走一輪跟 B.7-B.15 同等規模的調查，尚未投入。E07（resnet，見B.20）模型已實作完成、能訓練+評估，但還沒跑正式全量結果、也還沒為這個架構調過 one-cycle 之類的訓練方法論；E02（噪訊增強）還缺動態增強的訓練迴圈才能跑。

## E06（cnn_best）：CPU 環境下的縮短版負面結果

| run_dir | 配方 | N_TGE | GE@1000 | 備註 |
|---|---|---|---|---|
| `E06_cnn_best_20260816_1452` | **CPU環境縮短版12epoch，非原論文75epoch**，RMSprop lr=1e-5，batch=200，desync0，ID leakage | None | 162.39（比隨機127.5更差） | 原論文75epoch在這台無GPU機器要6.6小時，經使用者同意縮成12epoch(~1小時)拿誠實中間資料點。loss全程幾乎沒動（5.5452→5.5425，隨機基準5.545），PI=-0.0192（幾乎零資訊）。**不能代表cnn_best真實能力**，純粹是RMSprop lr=1e-5配合75epoch才收斂、12epoch連基準線都還沒真正脫離。細節見 CLAUDE.md 附錄 B.21 |

## E02（噪訊增強）：目前全專案最佳真實攻擊結果

| run_dir | 配方 | N_TGE | GE@1000 | PI | 備註 |
|---|---|---|---|---|---|
| `E02_noisy_augment_20260816_1618` | 沿用E01配方（he_uniform+one-cycle+MinMax）+動態高斯噪訊增強（sigma_ratio=0.5，每epoch重新生成） | **206**（比E01的475快2.3倍） | 0.0 | **+0.1796**（E01是-0.4999，由負轉正） | **目前全專案已知最佳真實攻擊結果**，取代E01成為新基準。N_TGE跟PI兩個獨立指標同時大幅進步，不是評估窗口挑出來的表面改善。細節見 CLAUDE.md 附錄 B.23 |

## E07（resnet）：完整訓練預算下的負面結果

| run_dir | 配方 | N_TGE | GE@1000 | 備註 |
|---|---|---|---|---|
| `E07_resnet_20260816_1559` | **完整跑滿**，Adam lr=1e-3，batch=128，epochs=100（epoch50觸發patience早停，非資源限制），desync0，ID leakage | None | 123.27（接近隨機127.5，略低） | 跟E06不同，這是誠實完整的負面結果，不是縮短版。loss有下降（5.5547→5.4451）但幅度不足以讓攻擊收斂，PI=-0.1129（略負）。初步判讀：resnet架構完全沒調過訓練方法論（one-cycle/初始化等），跟E03/E04一樣「還沒調」不等於「做不到」。細節見 CLAUDE.md 附錄 B.22 |

## 命名說明

- 目錄名格式 `{exp_id}_{timestamp}`，`exp_id` 對應 `configs/exp/*.yaml` 的 `exp_id` 欄位。
- `E01_repro_original_recipe` 不在 CLAUDE.md §8.2 的官方 E01-E08 編號內，是這次調查歷史數字來源時的診斷用實驗，config 檔案本身也有註記說明。
