# P1 seed=7 单因素消融结果

## 结论

- **尾部限制（C, robust 列变换后 clip_z=5）**：验证 AC-AUROC 从 B 的 0.5905 降至 0.5474；test AC-AUROC/AUPRC 分别下降 0.103/0.228，external AUROC/AUPRC 分别下降 0.076/0.063。固定尾部限制没有收益。
- **去年龄（D, x_static[0]=0）**：验证 AC-AUROC 降至 0.4741，是四个消融中最低；test 与 external 的主要排序指标均低于 B，external F-measure 为 0（没有预测阳性真例）。年龄是当前模型的重要有效输入，不应删除。用于评分的 raw_age 在屏蔽前保存，未被改写。
- **去 BMI（E, x_static[9]=0）**：验证 AC-AUROC 仅比 B 低 0.0043；test AUROC/AUPRC 小幅增加 0.002/0.032，但 test AC-AUROC 下降 0.067，external AUROC/AUPRC 下降 0.106/0.099。结果混合，按预先规定的验证 AC-AUROC 选模规则不能认定去 BMI 有收益。
- **去相干性（F, X_seq[:,54:414]=0）**：验证 AC-AUROC 下降 0.0689，test AC-AUROC/AUPRC 下降 0.048/0.259；external AC-AUROC/AUPRC 相对 B 增加 0.104/0.095，但 external AUROC 下降 0.019。相干性对验证和 test 有帮助；external 的改善是预先固定实验的泛化观察，不据此继续调参。

四个 typed 变体在 external 的 AUROC、AC-AUROC 和 AUPRC 均未超过 legacy A。尤其不能因 Reward 或 Accuracy 单项较高就判定总体更好，因为它们明显受保存阈值与类别不平衡影响。

## 实际版本与执行口径

- 基准提交：`19e540101f9f1da66062df48c2bebf9b40b6a662`。
- 实际生产文件 SHA256：
  - `feature_scaling.py`: `88ee07403d2dff6fa77b16d16ea1d82b1d6c2183582ca38b8b8946b3be869dde`
  - `train_lstm.py`: `2594ad792c42b80546186acd960338895ea1ed34e7fa4bc3bf5a0bedd0b268a3`
  - `feature_rules_v1.json`: `b8624092b12e5a2800d7627413d6d4c791fec8c7fb77552a007e1d43342c3c90`
- 相对基准提交只有 `feature_scaling.py` 和 `train_lstm.py` 的 P1 实现改动；没有修改规则文件、原始 NPZ、特征提取器、网络、损失、采样器或训练超参数。
- `train_model.py`、`run_model.py`、`evaluate_model.py` 未修改；SHA256 分别为：
  - `train_model.py`: `9c4b294d6a8fef065311603175aa3d0ba40462e64631becf1dbacd45afe673a4`
  - `run_model.py`: `09941ccaeff383ee9911bc662d539522b080ea4599268dd07cb68afd3edd9021`
  - `evaluate_model.py`: `227e40a45a7ddb918812c17fd4a8e678334134c965d488db763409d245802970`
- C 保持原 P6000 训练进程；D/E/F 按后续指令在 H100、`eeg_env` 中顺序训练。数据、split、seed=7、B 预处理拟合状态和训练参数相同，但硬件不同，这是本轮成对性限制。
- C/D/E/F 均从头初始化网络，只复用 B checkpoint 中已拟合的 typed_v1 预处理状态；没有加载 B 的网络权重、校准器或阈值。
- 训练末尾只跳过 external NPZ 中 `-1` 标签的无效评分；验证评价、Platt 校准、Youden 阈值和 checkpoint 保存保持原逻辑。

## 官方评价流程

最终所有 train/test/external 结果均由仓库原有入口生成：

```bash
python run_model.py -d <data_folder> -m <model_folder> -o <output_folder>
python evaluate_model.py -d <labels.csv> -o <output_folder>/demographics.csv \
  -p <prevalence.csv> -s <scores.csv> -t <table.csv>
```

未使用 `run_model.py -f`，因此没有把失败预测静默写为 NaN。曾临时新增的自然训练集评价器及其 `train_metrics.json` 已停止并删除，以下训练集结果也完全来自上述官方流程。

- train：冻结 split 的自然733人，680阴性、53阳性；名单与 `/database2/physionet2026_kaggle/data/splits/train_records.json` 内容哈希一致。
- train 标签：直接从原始1103人 `/database2/physionet2026_kaggle/data/demographics.csv` 按冻结名单取子集，未读取 NPZ 标签；输入 SHA256 为 `98cb5be3b2287b32a548d78b71498da11cb6ab4cf17617c7c55b163a1512fee2`。
- test：158人；external：54人。两组使用上一轮相同真实标签和 prevalence 文件。
- 所有模型均从各自 checkpoint 自动恢复预处理、网络、校准器和阈值；没有在评价集重新 fit 或选阈值。

## 模型选择、校准与配置

| Arm | 输入配置 | Best epoch | Stop epoch | Best val AC-AUROC | Platt coef | Platt intercept | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|
| A_legacy_clip | legacy clip[-50,50] | 7 | 22 | 0.6810 | 0.545814 | -3.057671 | 0.075693 |
| B_typed_v1 | typed_v1, no mask, clip_z=None | 16 | 31 | 0.5905 | 0.028964 | -1.994392 | 0.073785 |
| C_typed_clip5 | typed_v1, robust transformed columns clip_z=5 | 13 | 28 | 0.5474 | 0.042965 | -1.896520 | 0.084143 |
| D_typed_no_age | typed_v1, x_static[0]=0 | 5 | 20 | 0.4741 | 0.011048 | -2.429449 | 0.083662 |
| E_typed_no_bmi | typed_v1, x_static[9]=0 | 13 | 28 | 0.5862 | 0.042134 | -1.876487 | 0.067580 |
| F_typed_no_coherence | typed_v1, X_seq[:,54:414]=0 | 16 | 31 | 0.5216 | 0.002222 | -2.455935 | 0.075556 |

## 自然训练集结果（不参与选模）

| Arm | AC-AUROC | AUROC | AUPRC | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_legacy_clip | 0.826 | 0.849 | 0.347 | 42 | 149 | 11 | 531 |
| B_typed_v1 | 1.000 | 1.000 | 1.000 | 53 | 217 | 0 | 463 |
| C_typed_clip5 | 1.000 | 1.000 | 0.999 | 53 | 126 | 0 | 554 |
| D_typed_no_age | 0.995 | 0.994 | 0.899 | 53 | 24 | 0 | 656 |
| E_typed_no_bmi | 0.999 | 0.999 | 0.976 | 53 | 307 | 0 | 373 |
| F_typed_no_coherence | 1.000 | 1.000 | 1.000 | 53 | 336 | 0 | 344 |

typed 模型的训练 AUROC 接近1，而验证 AC-AUROC仅为0.47–0.59，显示明显过拟合；该结果只用于诊断，不参与 checkpoint 选择。

## Test 七项官方指标（158人）

| Arm | Reward | AC-AUROC | Age-weighted AUROC | AUROC | AUPRC | Accuracy | F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_legacy_clip | 0.440 | 0.812 | 0.825 | 0.814 | 0.287 | 0.734 | 0.300 |
| B_typed_v1 | 0.349 | 0.800 | 0.797 | 0.843 | 0.566 | 0.639 | 0.260 |
| C_typed_clip5 | 0.023 | 0.697 | 0.714 | 0.813 | 0.338 | 0.747 | 0.286 |
| D_typed_no_age | 0.452 | 0.739 | 0.777 | 0.829 | 0.353 | 0.873 | 0.333 |
| E_typed_no_bmi | -0.130 | 0.733 | 0.704 | 0.845 | 0.598 | 0.532 | 0.213 |
| F_typed_no_coherence | 0.191 | 0.752 | 0.756 | 0.831 | 0.307 | 0.456 | 0.204 |

### Test：C/D/E/F 相对 B 的差值

| Arm | Reward | AC-AUROC | Age-weighted AUROC | AUROC | AUPRC | Accuracy | F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | -0.326 | -0.103 | -0.083 | -0.030 | -0.228 | +0.108 | +0.026 |
| D | +0.103 | -0.061 | -0.020 | -0.014 | -0.213 | +0.234 | +0.073 |
| E | -0.479 | -0.067 | -0.093 | +0.002 | +0.032 | -0.107 | -0.047 |
| F | -0.158 | -0.048 | -0.041 | -0.012 | -0.259 | -0.183 | -0.056 |

### Test：C/D/E/F 相对 A 的差值

| Arm | Reward | AC-AUROC | Age-weighted AUROC | AUROC | AUPRC | Accuracy | F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | -0.417 | -0.115 | -0.111 | -0.001 | +0.051 | +0.013 | -0.014 |
| D | +0.012 | -0.073 | -0.048 | +0.015 | +0.066 | +0.139 | +0.033 |
| E | -0.570 | -0.079 | -0.121 | +0.031 | +0.311 | -0.202 | -0.087 |
| F | -0.249 | -0.060 | -0.069 | +0.017 | +0.020 | -0.278 | -0.096 |

## External 七项官方指标（54人）

| Arm | Reward | AC-AUROC | Age-weighted AUROC | AUROC | AUPRC | Accuracy | F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_legacy_clip | 0.167 | 0.740 | 0.686 | 0.791 | 0.420 | 0.778 | 0.400 |
| B_typed_v1 | 0.041 | 0.521 | 0.548 | 0.611 | 0.262 | 0.537 | 0.242 |
| C_typed_clip5 | 1.130 | 0.521 | 0.515 | 0.535 | 0.199 | 0.556 | 0.250 |
| D_typed_no_age | -0.144 | 0.479 | 0.506 | 0.576 | 0.176 | 0.778 | 0.000 |
| E_typed_no_bmi | 1.010 | 0.562 | 0.528 | 0.505 | 0.163 | 0.444 | 0.211 |
| F_typed_no_coherence | 1.216 | 0.625 | 0.596 | 0.592 | 0.357 | 0.481 | 0.300 |

### External：C/D/E/F 相对 B 的差值

| Arm | Reward | AC-AUROC | Age-weighted AUROC | AUROC | AUPRC | Accuracy | F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | +1.089 | +0.000 | -0.033 | -0.076 | -0.063 | +0.019 | +0.008 |
| D | -0.185 | -0.042 | -0.042 | -0.035 | -0.086 | +0.241 | -0.242 |
| E | +0.969 | +0.041 | -0.020 | -0.106 | -0.099 | -0.093 | -0.031 |
| F | +1.175 | +0.104 | +0.048 | -0.019 | +0.095 | -0.056 | +0.058 |

### External：C/D/E/F 相对 A 的差值

| Arm | Reward | AC-AUROC | Age-weighted AUROC | AUROC | AUPRC | Accuracy | F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | +0.963 | -0.219 | -0.171 | -0.256 | -0.221 | -0.222 | -0.150 |
| D | -0.311 | -0.261 | -0.180 | -0.215 | -0.244 | +0.000 | -0.400 |
| E | +0.843 | -0.178 | -0.158 | -0.286 | -0.257 | -0.334 | -0.189 |
| F | +1.049 | -0.115 | -0.090 | -0.199 | -0.063 | -0.297 | -0.100 |

## 混淆矩阵

| Arm | Test TP/FP/FN/TN | External TP/FP/FN/TN |
|---|---|---|
| A_legacy_clip | 9 / 40 / 2 / 107 | 4 / 8 / 4 / 38 |
| B_typed_v1 | 10 / 56 / 1 / 91 | 4 / 21 / 4 / 25 |
| C_typed_clip5 | 8 / 37 / 3 / 110 | 4 / 20 / 4 / 26 |
| D_typed_no_age | 5 / 14 / 6 / 133 | 0 / 4 / 8 / 42 |
| E_typed_no_bmi | 10 / 73 / 1 / 74 | 4 / 26 / 4 / 20 |
| F_typed_no_coherence | 11 / 86 / 0 / 61 | 6 / 26 / 2 / 20 |

## Checkpoint SHA256

| Arm | SHA256 |
|---|---|
| A | `d91d18c42694e55041b4e852bba95c0c523bba757dac5ab344310744b98b8b8c` |
| B | `b4fb534bc1310d4b7ae132fdcfd7f04d6664c04cecf06028cf108e6af5342125` |
| C | `361a0748477bbf04b5b9a9da4b62ea6f235e685ad4486baa3cdb095b2fdef41b` |
| D | `e21f883707b671a3a0cb2383db46bd55c38ea4e643bdf39e50978d7a5289db5a` |
| E | `e252c44b02a875fe762fd8fb4a06e0e2808de0e17c4f9d6b17f79ef3cab24f52` |
| F | `03cd27a6c2a4ed43112bfdeba1eb7af06c799dfd41630a42d187b4eb40e72fac` |

## 运行异常与边界

- 19项轻量/回归测试通过；未发现非有限 loss、预测或缓存缺失。
- 自然训练集官方评价首次顺序命令把 F 目录名误写，C/D/E 已完成后在进入 F 前以非零状态退出；随后使用正确目录单独完成 F。没有修改模型或结果。
- 本轮只按验证 AC-AUROC 解释模型选择。external 仅作为固定实验的泛化观察，没有据此新增实验、调整阈值或改变训练。
