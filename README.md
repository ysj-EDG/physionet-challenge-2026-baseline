# PhysioNet Challenge 2026 — 基于 LSTM 的认知障碍筛查

> 从整夜多导睡眠图 (PSG) 中提取多模态 per-epoch 时序特征，训练 2-Layer LSTM 模型，筛查认知障碍风险。

---

## 目录

1. [项目概述](#1-项目概述)
2. [数据与预处理](#2-数据与预处理)
3. [特征工程](#3-特征工程)
4. [模型架构](#4-模型架构)
5. [训练流程](#5-训练流程)
6. [推理与评估](#6-推理与评估)
7. [使用方法](#7-使用方法)
8. [文件结构](#8-文件结构)
9. [实验结果](#9-实验结果)
10. [依赖环境](#10-依赖环境)

---

## 1. 项目概述

本项目针对 **PhysioNet Challenge 2026** 竞赛任务：基于整夜 PSG 记录，预测受试者是否存在认知障碍（Cognitive Impairment, 二分类）。

### 方法论总览

```
原始 PSG (EDF)
    │
    ├─ 通道标准化 (channel_table.csv)  →  双极推导  →  200 Hz 重采样
    │
    ├─ CAISR 算法标注 (睡眠分期 + 觉醒/呼吸/肢体事件)
    │
    └─ 特征提取:
         │
         ├── 全夜静态特征 (196 维)
         │     Demographic(10) + Algorithmic(186)
         │     → 拼接到 LSTM 输出后
         │
         └── Per-30s 时序特征 (476 维/epoch)
               Per-epoch: EEG(414) + EMG(24) + Resp(14) + OneHot(13)
               + ECG HRV(11) (滑动5分钟窗口, stride=30s)
               → LSTM 时序输入
```

### 模型

| 模型 | 输入 | 参数量 | 保存路径 |
|------|------|--------|----------|
| **2-Layer LSTM** | 变长时序 (476/epoch) + 196 静态 | ~1.0M | `lstm_model/lstm_model.pt` |

---

## 2. 数据与预处理

### 2.1 数据来源

PhysioNet Challenge 2026 训练集，包含多家医院 (Site) 的 PSG 记录。每条记录包含：

| 数据类型 | 文件 | 格式 |
|----------|------|------|
| 人口学信息 | `demographics.csv` | CSV |
| 生理信号 | `physiological_data/{SiteID}/{sub-..._ses-...}.edf` | EDF |
| 算法标注 | `algorithmic_annotations/{SiteID}/{sub-...}_caisr_annotations.edf` | EDF |
| 通道别名表 | `channel_table.csv` | CSV |

### 2.2 数据集划分

`split_dataset.py` 按 **7:1:2** 分层划分（按 `Cognitive_Impairment` 标签分层，seed=42）：

| 集合 | 比例 | 用途 |
|------|------|------|
| Train | 70% | 模型训练 |
| Val | 10% | 早停 / 超参选择 |
| Test | 20% | 最终评估 |

输出 `{data_folder}/splits/{train,val,test}_records.json`。

### 2.3 通道预处理

所有提取器共用同一套通道标准化 + 双极推导流程：

1. **通道名标准化**：原始通道名 → 小写 → 去除 `_pds`/`_eg` 后缀 → `:` 替换为 `-` → 通过 `channel_table.csv` 映射到标准名
2. **双极推导**：自动推导缺失的双极导联：

| 目标导联 | 公式 | 用途 |
|----------|------|------|
| f3-m2 | F3 − M2 | 左额 EEG |
| f4-m1 | F4 − M1 | 右额 EEG |
| c3-m2 | C3 − M2 | 左中央 EEG |
| c4-m1 | C4 − M1 | 右中央 EEG |
| o1-m2 | O1 − M2 | 左枕 EEG |
| o2-m1 | O2 − M1 | 右枕 EEG |
| e1-m2 | E1 − M2 | 左眼 EOG |
| e2-m1 | E2 − M1 | 右眼 EOG |
| chin1-chin2 | Chin1 − Chin2 | 颏 EMG |

3. **重采样**：全部通道通过分数有理数重采样到 200 Hz

---

## 3. 特征工程

### 3.0 特征维度总览

```
┌──────────────────────────────────────────────────────┐
│  时序输入 (per epoch, 变长)                           │
│  = EEG(414) + EMG(24) + Resp(14) + OneHot(13)        │
│    + ECG HRV(11, 滑动5分钟, stride=30s)                │
│  = 476 dims/epoch                                     │
├──────────────────────────────────────────────────────┤
│  静态拼接 (全夜, 固定)                                 │
│  = Demographic(10) + Algorithmic(186)                 │
│  = 196 dims                                           │
├──────────────────────────────────────────────────────┤
│  LSTM 输出拼接: 256 + 196 = 452 dims                  │
│  最终 FC: 452 → 64 → 1 (sigmoid)                      │
└──────────────────────────────────────────────────────┘
```

### 3.1 人口学特征 (10 维)

**文件**：`feature_extractor_demographic.py`  
**类别**：`DemographicMixin`

| 维度 | 特征 | 编码 |
|------|------|------|
| 1 | Age | 连续值 (岁)，缺失填 0 |
| 2–4 | Sex | One-hot: Female, Male, Other/Unknown |
| 5–9 | Race | One-hot: Asian, Black, Others, Unavailable, White |
| 10 | BMI | 连续值 (kg/m²) |

### 3.2 算法标注特征 (186 维)

**文件**：`feature_extractor_algorithmic.py`  
**类别**：`AlgorithmicMixin`  
**输入**：CAISR 算法标注 EDF（`stage_caisr`, `arousal_caisr`, `resp_caisr`, `limb_caisr`, 后验概率）

CAISR 睡眠分期编码：`1=N3, 2=N2, 3=N1, 4=REM, 5=Wake`

#### 3.2.1 睡眠结构特征 (84 维)

**基础睡眠参数 (7 维)**：TRT, TST, SE, SOL, REM_latency, wake_time, WASO

**各期比例与时长 (10 维)**：N1/N2/N3/REM/Wake 的百分比和绝对时长（秒）

**分期比值 (4 维)**：NREM_pct, N3/N1_ratio, N3/(N1+N2)_ratio, REM/NREM_ratio

**碎片化指标 (4 维)**：

| 特征 | 计算方式 |
|------|----------|
| `stage_transition_count` | 相邻 epoch 分期变化次数 |
| `transition_rate` | 转换次数 / (TST/3600)，单位 次/h |
| `wake_intrusions` | 入睡后由睡眠期进入清醒期的次数 |
| `short_bout_ratio` | 睡眠 bout 中时长 < 3 epoch 的比例 |

**Bout 构建算法**：
```
bout = 连续相同分期的一段
_diffs = diff(stages)
starts = [0] + {i+1 | _diffs[i] ≠ 0}
ends = starts[1:] + [n_epochs]
```

**Bout 统计 (39 维)**：对 W/N1/N2/N3/REM 五期各计算 7 个时长统计量（mean/median/max/P25/P75/P90/P95，共 35 维），加上 N1/N2/N3/REM 各期 bout 计数（4 维）。

**早晚动态 (8 维)**：

| 特征 | 说明 |
|------|------|
| early_N3_pct, late_REM_pct | 前/后半夜 N3、REM 占 TST 比例 |
| delta_N3, delta_REM, delta_W | 后半夜 − 前半夜的变化 |
| sleep_cycle_count | 完整 NREM-REM 周期数 |
| mean_cycle_duration_sec | 周期平均时长 |
| first_cycle_NREM_duration_sec | 第一周期 NREM 阶段时长 |

**NREM-REM 周期检测**：从入睡后开始扫描 bouts，每检测到 NREM→REM 序列计为一个完整周期。

**稳定性 (3 维)**：最长连续睡眠 bout、最长 N3 bout、最长 REM bout 的秒数。

**后验不确定性 (6 维)**：

| 特征 | 计算方式 |
|------|----------|
| mean_max_prob | 每 epoch 最大后验概率的均值 |
| std_max_prob | 最大后验概率的标准差 |
| mean_stage_entropy | 归一化分期熵 −Σ P(c)·ln(P(c)) / ln(5) 的均值 |
| high_entropy_ratio | 熵 > 1.2 的 epoch 比例 |
| low_conf_ratio | 最大概率 < 0.6 的 epoch 比例 |
| posterior_volatility | 相邻 epoch 后验向量 L1 距离的均值 |

**综合指数 (3 维)**：
```
sleep_fragmentation_index  = transition_rate + WASO/3600 + short_bout_ratio
deep_sleep_preservation_index = N3_pct + longest_N3_hours − N3_fragmentation
REM_integrity_index        = REM_pct + mean_REM_bout_hours − REM_fragmentation
```

#### 3.2.2 觉醒事件特征 (30 维)

**事件分割**：二值标签 (0.5s 分辨率) → diff 找边 → (start_sec, end_sec, duration)

| 类别 | 维度 | 内容 |
|------|------|------|
| 总负担 | 4 | arousal_count, duration_sec, AI(次/h TST), burden_ratio |
| 时长分布 | 6 | mean/median/max/P75/P90/P95 arousal duration |
| 间隔与爆发 | 4 | mean/std/min inter-arousal interval, burst_ratio(<30s) |
| 分期 AI | 8 | NREM/REM/N1/N2/N3/early/late Arousal Index + delta_AI |
| 概率统计 | 5 | mean/std/P90/P95 arousal_prob, high_prob_ratio(>0.5) |
| 耦合指标 | 3 | transition/respiratory/limb linked arousal ratio |

**分期 AI 计算**：将觉醒起始时间映射到睡眠分期 epoch (`stage = stages[floor(t/30)]`)，分类统计。

**耦合检测窗口**：
- 转换关联：觉醒与分期转换点重叠 ±15s
- 呼吸关联：觉醒与呼吸事件重叠 ±10s
- 肢体关联：觉醒与肢体事件重叠 ±10s

#### 3.2.3 呼吸事件特征 (42 维)

**事件分割**：多类标签序列 (1s 分辨率，0=无, 1=OA, 2=CA, 3=MA, 4=HY, 5=RERA)，连续同标签合并为一个事件。

| 类别 | 维度 | 内容 |
|------|------|------|
| 总负担 | 4 | count, duration_sec, REI(次/h TST), burden_ratio |
| 亚型分解 | 13 | OA/CA/MA/HY/RERA: 计数 + 指数(OAI/CAI/MAI/HYI/RERAI) + 比例 |
| 时长统计 | 8 | mean/max/P75/P90/P95，及 OA/CA/HY 各自 mean duration |
| 爆裂与聚集 | 2 | burst_ratio (间隔<30s), longest_event_cluster_sec |
| 分期 REI | 9 | NREM/REM/N1/N2/N3/early/late REI + rem/nrem ratio + delta |
| 耦合与综合 | 6 | post_event_arousal_ratio, resp_to_arousal_delay, transition_linked_ratio, AHI, obstructive_dominance, central_dominance |

**AHI 与主导度**：
```
AHI = (OA+CA+MA+HY)_count / (TST/3600)
obstructive_dominance = (OA + 0.5×HY) / total   # HY 按 0.5 权重
central_dominance = CA / total
```

**最长事件簇**：贪心聚类，相邻事件间隔 < 30s 视为同一簇。

#### 3.2.4 肢体运动事件特征 (30 维)

**事件分割**：多类标签 (1s 分辨率，0=无, 1=孤立性, 2=周期性 PLM)

| 类别 | 维度 | 内容 |
|------|------|------|
| 总负担 | 4 | count, duration_sec, LMI(次/h TST), burden_ratio |
| 亚型 | 5 | isolated/PLM count + isolated_index + PLMI + PLM_ratio |
| 时长 | 5 | mean/max/P75/P90/P95 limb duration |
| 间隔与爆发 | 3 | mean/std inter-limb interval, burst_ratio(<30s) |
| 分期 LMI | 7 | NREM/REM/N2/N3/early/late LMI + delta |
| 多重耦合 | 6 | limb→arousal, arousal→limb, resp→limb, transition→limb, periodic/non-periodic burden |

---

### 3.3 EEG Per-Epoch 特征 (414 维/epoch)

**文件**：`feature_extractor_eeg_coherence.py`  
**类别**：`EEGCoherenceMixin`  
**依赖**：`ref/eeg_sleep_features.py` 提供 `eeg_segment_coherence()`

> XGBoost 使用跨 epoch 均值（全夜聚合），LSTM 直接使用 per-epoch 特征。

#### 频谱特征 (54 维/epoch)

6 通道 (F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1) × 9 PSD 特征：

| 特征 | 公式 | 频段 |
|------|------|------|
| logP2P1 | log(P 9–20Hz / P 30–47Hz) | 高/超高频比 |
| logPsigmaPbeta | log(P 11–16Hz / P 20–30Hz) | 纺锤/ beta 比 |
| logPalphaPbeta | log(P 8–13Hz / P 20–30Hz) | alpha/ beta 比 |
| logPthetaPbeta | log(P 4–7Hz / P 20–30Hz) | theta/ beta 比 |
| logPdeltaPbeta | log(P 0.5–3Hz / P 20–30Hz) | delta/ beta 比 |
| SEF50 | 50% 频谱边缘频率 | Hz |
| logP0 | log(P 0.5–47Hz 总功率) | |
| logPbeta | log(P 20–30Hz) | beta 功率 |
| logP1 | log(P 30–47Hz) | 高频功率 |

#### 相干特征 (360 维/epoch)

15 导联对 × 24 特征：

**15 导联对**：F3-F4, F3-C3, F3-C4, F3-O1, F3-O2, F4-C3, F4-C4, F4-O1, F4-O2, C3-C4, C3-O1, C3-O2, C4-O1, C4-O2, O1-O2

**24 特征/对**（从 100-bin、0.5 Hz/bin 的幅度平方相干谱提取）：

| 类别 | 数量 | 特征名 |
|------|------|--------|
| 频带均值 | 5 | mean_delta, mean_theta, mean_alpha, mean_sigma, mean_beta |
| 频带积分 (AUC) | 5 | auc_delta, auc_theta, auc_alpha, auc_sigma, auc_beta |
| 频带 IQR | 5 | iqr_delta, iqr_theta, iqr_alpha, iqr_sigma, iqr_beta |
| 频带比值 | 4 | sigma/alpha/beta_delta_ratio, high_low_ratio |
| 谱形状 | 3 | spectral_centroid, spectral_entropy, spectral_bandwidth |
| Sigma 细分 | 2 | low_sigma_mean (11–13.5Hz), high_sigma_mean (13.5–16Hz) |

**频带定义**：delta=0.5–4, theta=4–8, alpha=8–13, sigma=11–16, beta=16–30 Hz

---

### 3.4 EMG Per-Epoch 特征 (24 维/epoch)

**文件**：`feature_extractor_emg.py`  
**类别**：`EMGMixin`

#### 全夜预处理

```
原始 EMG → NaN 线性插值 → 去中位数
    → 10–90 Hz 4阶 Butterworth 带通 → 整流
    → 5 Hz 低通包络 → 全夜 20% 分位数基线
```

#### 30s Epoch 特征 (8 维/通道)

| 特征 | 说明 |
|------|------|
| log_rms_norm | log(RMS / 全夜基线) |
| envelope_iqr_norm | 包络 IQR / 包络中位数 |
| active_fraction | 包络 > 2×基线的时间比 |
| tonic_fraction | 连续高活动 ≥5s 的时间比 |
| burst_rate_per_min | burst 频率 (下颏 0.1–5s, 腿 0.5–10s) |
| burst_duty_cycle | burst 时间占 epoch 比 |
| phasic_mini_epoch_fraction | 3s mini-epoch 有 burst 的比例 (仅下颏) |
| log_hf_lf_ratio | log(20–55Hz / 10–20Hz 功率比) |

3 通道 = chin(8) + lleg(8) + rleg(8) = **24 维/epoch**。

---

### 3.5 呼吸 Per-Epoch 特征 (14 维/epoch)

**文件**：`feature_extractor_resp.py`  
**类别**：`RespMixin`

#### 全夜预处理（每通道独立）

```
原始信号 → NaN 插值 → 25 Hz 重采样 → detrend
    → 0.05–1 Hz 4阶 Butterworth 带通
    → Hilbert 包络 → 0.4 Hz 低通平滑
    → 120s 滚动 80% 分位数基线 → 相对幅度
```

#### 30s Epoch 特征

| 通道 | 维数 | 特征 |
|------|------|------|
| Airflow | 7 | rate_bpm, cycle_cv, amp_local_norm, amp_iqr_over_median, reduction30_fraction, near_absent_fraction, longest_reduction_sec |
| Thorax | 2 | amp_local_norm, reduction30_fraction |
| Abdomen | 2 | amp_local_norm, reduction30_fraction |
| Joint | 3 | thorax_abd_corr, thorax_abd_lag_sec, thorax_abd_paradox_fraction |

**反常运动检测**：5s 滑动窗口内胸腹相关系数 < −0.25 的比例。

---

### 3.6 事件 OneHot Per-Epoch 特征 (13 维/epoch)

**文件**：`feature_extractor_event_onehot.py`  
**类别**：`EventOneHotMixin`

每 30s epoch 提取 13 类标签：

| 标签 | 内容 |
|------|------|
| stage_N1, N2, N3, REM, Wake | 睡眠分期 one-hot (互斥) |
| arousal | epoch 内觉醒事件覆盖的时间比 |
| resp_OA, CA, MA, HY, RERA | 各呼吸亚型覆盖时间比 |
| limb_isolated, PLM | 肢体运动亚型覆盖时间比 |

---

### 3.7 ECG HRV 特征 (11 维/5min-window)

**文件**：`feature_extractor_ecg_neurokit.py`  
**类别**：`ECGNeurokitMixin`  
**依赖**：neurokit2

#### 处理流程

```
ECG 原始信号 (200 Hz)
    → 5-min 滑动窗口, stride=30s
    → nk.ecg_clean(method="neurokit")
    → nk.ecg_peaks(method="neurokit")
    → nk.signal_fixpeaks(method="Kubios", iterative=True)  # 间期校正
    → nk.hrv_time() + nk.hrv_frequency(psd_method="welch")
      + nk.hrv_nonlinear() + nk.hrv_symbolic()
    → 每窗口 11 维
```

#### 11 核心特征

| 特征 | 域 | 说明 |
|------|-----|------|
| HRV_MedianNN | 时域 | RR 间期中位数 (ms) |
| HRV_MCVNN | 时域 | RR 变异系数 |
| HRV_CVNN | 时域 | NN 间期变异系数 |
| HRV_CVSD | 时域 | RR 差值变异系数 |
| HRV_pNN20 | 时域 | 相邻 RR 差 >20ms 比例 |
| log_HRV_LF | 频域 | log(LF 0.04–0.15 Hz 绝对功率 ms²) |
| log_HRV_HF | 频域 | log(HF 0.15–0.40 Hz 绝对功率 ms²) |
| HRV_HF_rel_LFHF | 频域 | HF / (LF+HF) |
| HRV_SD1SD2 | 非线性 | Poincaré SD1/SD2 ratio |
| Symbolic_EquProb4_0V | 符号动力学 | 0 变异模式比例 |
| Symbolic_EquProb4_2UV | 符号动力学 | 2 不同变异模式比例 |

#### 时间对齐

前 5 分钟 (epoch 0–9) 无 ECG 特征。从第 10 个 epoch 开始，`X_ecg[i]` 对齐第 `10+i` 个 epoch。mask 向量标记哪些 epoch 有完整 ECG。

---

### 3.8 PerEpochExtractor — 统一入口

**文件**：`per_epoch_extractor.py`  
**类别**：`PerEpochExtractor` (继承 DemographicMixin + AlgorithmicMixin + EEGCoherenceMixin)

```python
extractor = PerEpochExtractor()
X_seq, X_ecg, x_static, y, mask = extractor.extract_all(record, data_folder)

# X_seq:    (N_epochs, 465)   # per-epoch 时序
# X_ecg:    (N_5min_wins, 11) # 滑动 ECG HRV
# x_static: (196,)            # 全夜静态
# y:        int               # 标签 0/1
# mask:     (N_epochs,) bool  # 有 ECG 对齐的 epoch
```

---

## 4. 模型架构

### 2-Layer LSTM + Static Feature Concatenation

```
                    X_seq (T×465) ──┐
                    X_ecg (T×11) ───┤
                                    ├── concat → (T×476)
                                    │
    ┌───────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────┐
│  LSTM Layer 1 (128, batch_first)     │
│  LSTM Layer 2 (128)                  │
│  dropout = 0.3, bidirectional=False  │
│  pack_padded_sequence (变长)         │
│  权重初始化: Xavier(uniform) +       │
│              orthogonal(hh)          │
│              forget gate bias = 1    │
└──────────────┬───────────────────────┘
               │
               ▼
    h_n: (2 layers, B, 128)
    h_last = cat(h_n[0], h_n[1])  →  (B, 256)
               │
               ├── x_static (B, 196) ──┘
               ▼
         combined = (B, 452)
               │
               ▼
       FC(452 → 64) → ReLU → Dropout(0.3)
               │
               ▼
           FC(64 → 1)
               │
               ▼
         sigmoid → P ∈ [0, 1]
```

**参数量**：~1,027,713

**训练超参数**：

| 超参数 | 值 |
|--------|-----|
| Batch Size | 8 |
| Learning Rate | 1e-3 |
| Optimizer | Adam |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=8, mode='max') |
| Max Epochs | 80 (patience=15 早停) |
| Loss | BCEWithLogitsLoss |
| Gradient Clipping | max_norm=1.0 |
| Dropout | 0.3 |
| Hidden dim | 128 × 2 layers |
| FC hidden | 64 |

### Collate 策略

`collate_fn` 负责将变长序列组成 batch：

1. 过滤掉提取失败的记录
2. 取 batch 内最大长度 `max_len`
3. 零填充 X_seq 和 X_ecg 到 `max_len`
4. ECG 对齐：`X_ecg[i, 10:10+ecg_len]` — 前 10 个 epoch 无 ECG
5. mask_ecg 标记有效 ECG 位置

---

## 5. 训练流程

### 5.1 数据准备

```bash
python split_dataset.py
```

输出 `{data_folder}/splits/`：
```
train_records.json   (70%)
val_records.json     (10%)
test_records.json    (20%)
split_summary.json
```

### 5.2 训练 LSTM

```bash
python train_lstm.py
```

**特征缓存机制**：首次运行对每条记录调用 `PerEpochExtractor.extract_all()` 并写入 `lstm_cache/{train,val,test}/*.npz`，后续运行直接从缓存加载（毫秒级）。

**训练流程**：

1. 加载 `train/val/test` 的 records JSON
2. 构建 `PSGDataset`（首次提取特征 + 写入缓存，后续直接读缓存）
3. 构建 `DataLoader` + 自定义 `collate_fn`（变长序列 pad + ECG 对齐）
4. 特征诊断：对第一个 batch 打印各张量的 shape / nan / inf / min / max / mean / std
5. 训练循环：
   - 每 epoch 训练完在 val 集上评估 AUROC + TPR@5%
   - ReduceLROnPlateau（val AUROC 上 8 epoch 不涨则学习率减半）
   - 早停：15 epoch 无提升
6. 加载最佳 checkpoint，在 test 集上评估
7. 保存 `lstm_model/lstm_model.pt`（含 state_dict + val_auroc）

---

## 6. 推理与评估

### 6.1 推理脚本

```bash
python infer_lstm.py
# 或指定参数
python infer_lstm.py --data_folder /path/to/data --batch_size 256 --output_dir ./results
```

**优化策略**：

| 优化 | 说明 |
|------|------|
| **预加载** | 一次性将所有 `.npz` 缓存读入内存，消除 per-batch 磁盘 I/O |
| **长度排序** | 按序列长度降序排列，同 batch 内长度相近 → padding 浪费最小 |
| **大批次** | 默认 batch_size=256（仅推理无梯度，GPU 利用率充分） |
| **inference_mode** | `torch.inference_mode()` 替代 `no_grad()`，禁用 autograd 开销 |
| **cudnn benchmark** | GPU 上自动选择最优卷积算法 |
| **cache miss 跳过** | 缓存不存在的记录直接跳过，不实时提取 |

### 6.2 输出文件

| 文件 | 格式 | 内容 |
|------|------|------|
| `lstm_test_predictions.csv` | CSV | `BDSPPatientID, Cognitive_Impairment, Cognitive_Impairment_Probability` |
| `lstm_test_metrics.txt` | 文本 | 完整评估指标（分组+全文） |

### 6.3 评估指标

| 类别 | 指标 |
|------|------|
| **基本信息** | n_total, n_positive, n_negative, pos_ratio |
| **概率分布** | prob_mean, prob_std, prob_median, prob_min, prob_max |
| **核心指标** | AUROC, AUPRC, Accuracy, F1, Precision, Recall, Specificity |
| **混淆矩阵** | TP, TN, FP, FN |
| **TPR@容量** | TPR@1%, @5%, @10%, @20%, @30%, @50% (含对应 capacity) |
| **分类条件概率** | 正/负类的预测概率均值 |
| **最佳阈值** | Youden's J 最优阈值及对应 TPR/FPR |
| **性能统计** | preload time, inference time, total wall time |

---

## 7. 使用方法

### 7.1 环境配置

```bash
pip install -r requirements.txt
```

### 7.2 完整训练 + 推理流程

```bash
# 1. 划分数据集
python split_dataset.py

# 2. 训练 LSTM (~数小时 GPU)
python train_lstm.py

# 3. 测试集推理 + 输出指标
python infer_lstm.py
```

### 7.3 路径配置

训练脚本默认路径（根据实际环境修改）：

```python
# train_lstm.py
DATA_FOLDER = "/mnt/database/physionet2026/training_set"
```

推理脚本支持命令行参数：
```bash
python infer_lstm.py --data_folder /path/to/training_set --model lstm_model/lstm_model.pt
```

---

## 8. 文件结构

```
challenge2026/
│
├── 数据与划分
│   ├── split_dataset.py              # 7:1:2 分层划分
│   ├── channel_table.csv             # 通道别名映射表
│   └── requirements.txt              # Python 依赖
│
├── 特征提取器
│   ├── feature_extractor.py                    # XGBoost 用聚合提取器 (备用)
│   ├── per_epoch_extractor.py                 # LSTM PerEpochExtractor 主入口
│   ├── feature_extractor_demographic.py        # DemographicMixin (10 维)
│   ├── feature_extractor_algorithmic.py        # AlgorithmicMixin (186 维)
│   ├── feature_extractor_eeg_coherence.py      # EEGCoherenceMixin (414 维/epoch)
│   ├── feature_extractor_ecg_neurokit.py       # ECG HRV (11 维/5min-window)
│   ├── feature_extractor_emg.py                # EMG burst/tonic (24 维/epoch)
│   ├── feature_extractor_resp.py               # 呼吸 (14 维/epoch)
│   ├── feature_extractor_event_onehot.py       # OneHot (13 维/epoch)
│   └── feature_extractor_physiological.py      # 生理信号精细特征 (153 维, 备用)
│
├── 模型训练与推理
│   ├── train_lstm.py                 # LSTM 训练 + 特征缓存
│   ├── infer_lstm.py                 # LSTM 测试集推理 + 指标输出
│   └── train_xgboost.py              # XGBoost 训练 (备用)
│
├── 模型文件
│   └── lstm_model/
│       └── lstm_model.pt             # 训练好的 LSTM 模型 (~1.9MB)
│
├── 特征缓存
│   └── lstm_cache/
│       ├── train/                    # 训练集时序缓存 (.npz)
│       ├── val/                      # 验证集时序缓存
│       └── test/                     # 测试集时序缓存
│
├── 输出
│   └── lstm_results/
│       ├── lstm_test_predictions.csv
│       └── lstm_test_metrics.txt
│
├── 参考代码与算法
│   └── ref/
│       ├── python-example-2026/      # 官方示例 (team_code, helper_code, evaluate_model)
│       ├── eeg_sleep_features.py     # EEG 频谱/相干核心算法
│       ├── result.txt                # 历史 Site-CV 实验结果
│       └── output_*/                  # EEG 可视化输出
│
├── 文档
│   ├── README.md                     # 本文件 (技术总文档)
│   ├── FEATURES.md                   # LSTM 特征速查
│   ├── algorithmic_features_methodology.md   # CAISR 算法特征公式
│   └── physiological_features_methodology.md # 生理信号特征公式
│
└── __pycache__/
```

---

## 9. 实验结果

> 测试集 122 条记录 (正负各 61)，GPU (CUDA)，batch_size=256

| 指标 | 值 |
|------|-----|
| **AUROC** | **0.7221** |
| **AUPRC** | **0.6680** |
| Accuracy | 0.5492 |
| F1 | 0.6893 |
| Precision | 0.5259 |
| Recall | 1.0000 |
| Specificity | 0.0984 |
| | |
| TP | 61 |
| TN | 6 |
| FP | 55 |
| FN | 0 |
| | |
| **TPR@1%** (cap=1) | 1.0000 |
| **TPR@5%** (cap=6) | 0.5000 |
| **TPR@10%** (cap=12) | 0.5833 |
| **TPR@20%** (cap=24) | 0.7083 |
| **TPR@30%** (cap=36) | 0.6944 |
| **TPR@50%** (cap=61) | 0.6885 |
| | |
| 正类概率均值 | 0.8209 |
| 负类概率均值 | 0.7045 |
| Youden 最佳阈值 | 0.7242 |
| 最佳阈值 TPR | 0.8525 |
| 最佳阈值 FPR | 0.4262 |
| | |
| 预加载耗时 | 1.7s |
| 推理耗时 | 0.3s |
| 总耗时 | 2.1s |

完整指标见 `lstm_results/lstm_test_metrics.txt`。

---

## 10. 依赖环境

```
numpy, scipy, pandas, scikit-learn
torch
neurokit2
edfio          # EDF 文件读取
tqdm           # 进度条
joblib         # 模型序列化
```

完整依赖见 `requirements.txt`。

---

## 附录 A: 维度速查表

| 缩写 | 全称 | Per-Epoch 维度 | 全夜静态维度 | 说明 |
|------|------|---------------|-------------|------|
| demo | 人口学 | — | 10 | Age/Sex/Race/BMI |
| algo_sleep | 睡眠结构 | — | 84 | 分期统计 + Bout + 周期 + 不确定性 |
| algo_arousal | 觉醒事件 | — | 30 | AI + 时长 + 分期分布 + 耦合 |
| algo_resp | 呼吸事件 | — | 42 | REI + 亚型 + AHI + 耦合 |
| algo_limb | 肢体运动 | — | 30 | LMI + PLMI + 多重耦合 |
| eeg | EEG 频谱+相干 | 414 | — | 6ch×9 PSD + 15pairs×24 coh |
| emg | EMG burst/tonic | 24 | — | chin(8) + lleg(8) + rleg(8) |
| resp | 呼吸信号 | 14 | — | airflow(7) + thorax(2) + abd(2) + joint(3) |
| onehot | 事件 OneHot | 13 | — | stage(5) + arousal + resp(5) + limb(2) |
| ecg | ECG HRV | 11 | — | neurokit2 5-min window |
| **时序合计** | | **476** | — | 465 + 11 (concat per step) |
| **静态合计** | | — | **196** | demo(10) + algo(186) |

## 附录 B: CAISR 编码速查表

| 编码 | 含义 |
|------|------|
| **睡眠分期** (30s/epoch) | |
| 1 | N3 (慢波睡眠) |
| 2 | N2 |
| 3 | N1 |
| 4 | REM |
| 5 | Wake |
| 9 | Unavailable |
| **呼吸事件** (1s resolution) | |
| 0 | 无事件 |
| 1 | OA (阻塞性呼吸暂停) |
| 2 | CA (中枢性呼吸暂停) |
| 3 | MA (混合性呼吸暂停) |
| 4 | HY (低通气) |
| 5 | RERA (呼吸努力相关觉醒) |
| **肢体运动** (1s resolution) | |
| 0 | 无 |
| 1 | 孤立性肢体运动 |
| 2 | 周期性肢体运动 (PLM) |
| **觉醒** (0.5s resolution) | |
| 0 | 无觉醒 |
| 1 | 觉醒 |
