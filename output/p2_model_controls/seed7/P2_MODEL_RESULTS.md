# P2 seed=7：类别加权与紧凑静态分类器结果

## 结论

- **取消第二次正类加权（H 对 G）**：在同一 H100、`eeg_env`、seed、采样器和输入下，验证 AC-AUROC 从 0.526 升至 0.569，验证 AUROC/AUPRC 也分别提高 0.008/0.012；test AC-AUROC/AUROC 提高 0.109/0.028，但 AUPRC 降低 0.109；external 排序指标基本持平且混合。按预先规定的验证集选择口径，本配方支持保留平衡采样、取消 BCE 的第二次正类加权，但单次 seed 尚不能证明稳定收益。
- **增加固定 20 项 CAISR（J 对 I）**：验证 AC-AUROC/AUROC/AUPRC 分别提高 0.101/0.025/0.018，test 三项分别提高 0.331/0.059/0.188，external AUROC/AUPRC 小幅提高 0.030/0.006。固定 compact30 在同一 LR 下显示出相对 demo10 的增量。
- **扩展到全部 196 个静态特征（K 对 J）**：验证 AC-AUROC/AUROC 分别下降 0.091/0.066，test AC-AUROC/AUROC/AUPRC 分别下降 0.055/0.010/0.015，训练—验证差明显扩大。虽然 external AUROC/AUPRC 小幅提高 0.006/0.048，但按验证集不能认定全静态扩展有收益。
- 本轮验证选择中，H 的验证 AC-AUROC 为 0.569；静态 LR 中 J 最好，为 0.539。test/external 已被多次观察，只作为探索性泛化观察，不作为继续调参依据。

## 版本、数据与执行口径

- 基准及实际 Git HEAD：`e048052f033a62e95ee911fab151faa773cfffde`；本轮在未提交工作区改动上执行，未自动提交或推送。
- G/H 在同一 `NVIDIA H100 PCIe`、`eeg_env`、Python 3.10.12、PyTorch 2.5.1+cu121 中顺序从头训练；两组初始网络内容哈希相同：`d743c45fef9d70b6333faaf70412fa7bd93e96ee573d89ffe3212e78bdf897d3`。
- 冻结 B 预处理 checkpoint SHA256：`b4fb534bc1310d4b7ae132fdcfd7f04d6664c04cecf06028cf108e6af5342125`；规则 SHA256：`b8624092b12e5a2800d7627413d6d4c791fec8c7fb77552a007e1d43342c3c90`。
- split SHA256：train `5d5be75f...261e`、val `9ee4e400...7bcd`、test `bfc237e3...c31e`、external `52c10b1b...8de`；人数固定为 733/158/158/54。
- `npz_new` 共 1103 个缓存，评价前逐名单核对命中；没有重提特征、写回或覆盖 NPZ。
- train 标签来自上一轮冻结的官方原始标签子集，输入 SHA256 为 `98cb5be3b2287b32a548d78b71498da11cb6ab4cf17617c7c55b163a1512fee2`；val 同样从官方 1103 人 demographics 按冻结名单和顺序取得，SHA256 为 `85e67385fe8df41521ca4cb6fce165018995e44e1463dcb38a00b4628680a6d3`。test/external 沿用 P1 的真实标签和 prevalence，未使用 NPZ 中的 `-1`。
- `train_model.py`、`run_model.py`、`evaluate_model.py` 未修改，SHA256 分别为 `9c4b294d...3a4`、`09941cca...21e`、`227e40a4...970`。所有 20 个评价均由后两者生成，未使用 `run_model.py -f`。

生产代码改动限于：`train_lstm.py` 增加默认仍为 empirical 的 `LSTM_POS_WEIGHT_MODE`；新增 checkpoint 兼容的 `StaticLogisticModel` 和固定 LR 训练入口；`team_code.py` 按 `model_type` 加载静态模型，并提供默认关闭的 `LSTM_DECISION_OUTPUT` sidecar。该 sidecar 只在官方推理已计算 logit/概率后写文件，不改变返回结果。

## 固定实验与训练结果

| Arm | 模型与输入 | 训练不平衡处理 | 参数/非零系数 | Best epoch | Stop epoch / LR iter | Best val AC-AUROC |
|---|---|---|---:|---:|---:|---:|
| G_B_sameenv | 原 LSTM；495 时序 + 196 静态 | WeightedRandomSampler + BCE pos_weight=12.830189 | 481,153 | 10 | 25 | 0.5259 |
| H_single_balance | 与 G 相同 | WeightedRandomSampler + BCE pos_weight=1 | 481,153 | 6 | 21 | 0.5690 |
| I_LR_demo10 | elastic-net LR；静态前 10 列 | 自然 733 人；class_weight=balanced | 7/10 非零 | N/A | 10,000，未收敛 | 0.4375 |
| J_LR_compact30 | 同 LR；前 10 + 固定 20 CAISR | 同 I | 14/30 非零 | N/A | 43，收敛 | 0.5388 |
| K_LR_static196 | 同 LR；全部静态列 | 同 I | 34/196 非零 | N/A | 195，收敛 | 0.4483 |

I/J/K 固定使用 `penalty=elasticnet`、`solver=saga`、`l1_ratio=0.4`、`C=0.03`、`class_weight=balanced`、`max_iter=10000`、`tol=1e-4`、`random_state=7` 和截距；没有搜索超参数。I 的 `max_iter` 收敛警告如实保留，未修改配置重跑。

## 自然训练集与验证集排序指标

| Arm | Train AC-AUROC | Val AC-AUROC | Gap | Train AUROC | Val AUROC | Gap | Train AUPRC | Val AUPRC | Gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G | 1.000 | 0.526 | 0.474 | 1.000 | 0.567 | 0.433 | 1.000 | 0.101 | 0.899 |
| H | 0.998 | 0.569 | 0.429 | 0.997 | 0.575 | 0.422 | 0.943 | 0.113 | 0.830 |
| I | 0.609 | 0.438 | 0.171 | 0.792 | 0.664 | 0.128 | 0.256 | 0.241 | 0.015 |
| J | 0.723 | 0.539 | 0.184 | 0.833 | 0.689 | 0.144 | 0.320 | 0.259 | 0.061 |
| K | 0.838 | 0.448 | 0.390 | 0.894 | 0.623 | 0.271 | 0.424 | 0.277 | 0.147 |

G/H 仍显示严重过拟合；H 缩小了三项训练—验证差，但没有消除过拟合。K 相比 J 的容量扩展明显扩大差距。

## Test 七项官方指标（158 人）

| Arm | Reward | AC-AUROC | Age-weighted AUROC | AUROC | AUPRC | Accuracy | F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| G | 0.143 | 0.739 | 0.685 | 0.841 | 0.562 | 0.892 | 0.414 |
| H | 0.495 | 0.848 | 0.867 | 0.869 | 0.453 | 0.823 | 0.391 |
| I | -0.027 | 0.348 | 0.367 | 0.789 | 0.300 | 0.646 | 0.263 |
| J | 0.158 | 0.679 | 0.616 | 0.848 | 0.488 | 0.886 | 0.471 |
| K | -0.319 | 0.624 | 0.577 | 0.838 | 0.473 | 0.367 | 0.167 |

## External 七项官方指标（54 人）

| Arm | Reward | AC-AUROC | Age-weighted AUROC | AUROC | AUPRC | Accuracy | F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| G | -0.090 | 0.604 | 0.559 | 0.584 | 0.225 | 0.722 | 0.118 |
| H | -0.048 | 0.562 | 0.571 | 0.587 | 0.206 | 0.759 | 0.133 |
| I | 0.481 | 0.521 | 0.568 | 0.698 | 0.267 | 0.574 | 0.378 |
| J | -0.095 | 0.583 | 0.601 | 0.728 | 0.273 | 0.759 | 0.235 |
| K | 1.397 | 0.583 | 0.579 | 0.734 | 0.321 | 0.389 | 0.327 |

Reward、Accuracy 和 F-measure 依赖各 checkpoint 在验证集固定的 Youden 阈值，不能替代排序指标进行模型选择。

## 混淆矩阵

| Arm | Test TP / FP / FN / TN | External TP / FP / FN / TN |
|---|---|---|
| G | 6 / 12 / 5 / 135 | 1 / 8 / 7 / 38 |
| H | 9 / 26 / 2 / 121 | 1 / 6 / 7 / 40 |
| I | 10 / 55 / 1 / 92 | 7 / 22 / 1 / 24 |
| J | 8 / 15 / 3 / 132 | 2 / 7 / 6 / 39 |
| K | 10 / 99 / 1 / 48 | 8 / 33 / 0 / 13 |

## 三个预定义差值

验证集排序指标：

| 对比 | Δ AC-AUROC | Δ AUROC | Δ AUPRC |
|---|---:|---:|---:|
| H − G | +0.043 | +0.008 | +0.012 |
| J − I | +0.101 | +0.025 | +0.018 |
| K − J | -0.091 | -0.066 | +0.018 |

Test 七项差值：

| 对比 | Δ Reward | Δ AC-AUROC | Δ Age-weighted | Δ AUROC | Δ AUPRC | Δ Accuracy | Δ F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| H − G | +0.352 | +0.109 | +0.182 | +0.028 | -0.109 | -0.069 | -0.023 |
| J − I | +0.185 | +0.331 | +0.249 | +0.059 | +0.188 | +0.240 | +0.208 |
| K − J | -0.477 | -0.055 | -0.039 | -0.010 | -0.015 | -0.519 | -0.304 |

External 七项差值：

| 对比 | Δ Reward | Δ AC-AUROC | Δ Age-weighted | Δ AUROC | Δ AUPRC | Δ Accuracy | Δ F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| H − G | +0.042 | -0.042 | +0.012 | +0.003 | -0.019 | +0.037 | +0.015 |
| J − I | -0.576 | +0.062 | +0.033 | +0.030 | +0.006 | +0.185 | -0.143 |
| K − J | +1.492 | +0.000 | -0.022 | +0.006 | +0.048 | -0.370 | +0.092 |

## 校准、阈值与保存状态

| Arm | Platt coef | Intercept | Threshold |
|---|---:|---:|---:|
| G | 0.023656 | -2.115723 | 0.092702 |
| H | 0.037077 | -2.215469 | 0.092276 |
| I | 0.716344 | -2.357338 | 0.071760 |
| J | 0.703293 | -2.327795 | 0.144195 |
| K | 0.471872 | -2.257830 | 0.047575 |

- 五组均保存相同的 B typed_v1 拟合状态，显式 `clip_z=None`、空 mask。相对旧 B 状态唯一新增内容是空 `mask_config` 字段；规则、中心和尺度未变化。
- I/J/K checkpoint 均保存全部 196 列的额外 StandardScaler；三者状态哈希相同：`115f72d01e1ab61c8ffbb79605c40976daae12a079898d807375e0b36b25d483`。选列索引、名称、系数、截距、迭代数和收敛状态均分别保存在各 checkpoint。
- 每个 arm 的 train/val/test/external 目录都包含官方预测、`scores.csv`、`table.csv` 和 `decision_outputs.csv`。sidecar 校准概率与官方输出逐行相同；所有 logit/概率有限，五组 Platt 系数均为正，20 个集合均无概率 0/1 饱和，因此没有排序翻转。

Checkpoint SHA256：

| Arm | SHA256 |
|---|---|
| G | `f2376c82eac8fae05d833bdd005cde46d160f2fd37ed632192bc1ca61a7cd5b0` |
| H | `71868c4f58a6126e0fedf364796ef77cdbe7166c0790a40a416b4dc0edd24411` |
| I | `15bca8ef44cebcffb77a97c88611e56d9c2e88af3410c56bdcf9a0c8921d34f5` |
| J | `37c3ec0b1359871dbdd175c60bec23498d519c49bbbabb4840f8e7482328270a` |
| K | `f7e111344d6fbc2c7baa3df4494b33d14117822d8cdf2dbd06c4691335de4b77` |

审查时可直接结合 `h100_run_identity.json`（完整环境、代码/规则/split 哈希）、`evaluation_manifest.json`（官方评价输入和脚本哈希）、`evaluation.log` 及各 arm 的 `train.log`。H100 训练阶段 `team_code.py` SHA256 为 `97c2eb5eb2e3ee0af0b28af9119996ee7b63bf86e3f6fae8cc7f3734588f4740`；评价阶段仅增加默认关闭的 sidecar 写出后为 `b2159346109b4e585ececa0c7f4ca699f377de199ee10963532d7680784228e1`。

## 历史 A–F 参考（跨硬件，非严格单变量对比）

| Arm | Val AC-AUROC | Test AC / AUROC / AUPRC | External AC / AUROC / AUPRC |
|---|---:|---|---|
| A legacy | 0.6810 | 0.812 / 0.814 / 0.287 | 0.740 / 0.791 / 0.420 |
| B typed | 0.5905 | 0.800 / 0.843 / 0.566 | 0.521 / 0.611 / 0.262 |
| C clip5 | 0.5474 | 0.697 / 0.813 / 0.338 | 0.521 / 0.535 / 0.199 |
| D no age | 0.4741 | 0.739 / 0.829 / 0.353 | 0.479 / 0.576 / 0.176 |
| E no BMI | 0.5862 | 0.733 / 0.845 / 0.598 | 0.562 / 0.505 / 0.163 |
| F no coherence | 0.5216 | 0.752 / 0.831 / 0.307 | 0.625 / 0.592 / 0.357 |

A/B/C 在 P6000，D/E/F 与本轮 G/H 在 H100/eeg_env；旧实验还存在训练随机性。因此 A–F 只作历史背景，G/H 才是本轮损失配置的严格同环境配对。

## 运行异常与边界

- 23 项轻量兼容/回归测试通过；未发现缓存缺失、非有限 loss、非有限预测或患者漏预测。
- 第一次 Medex 启动在训练前因 Python 仓库导入路径失败，修复启动器后重新提交；没有产生正式模型。修复不改变实验配置。
- 远端 rsync 工作区不含 `.git`，LR 训练时打印了一条非致命的 `not a git repository`，因此 LR checkpoint 的 `code_head=None`；独立的 H100 身份文件已记录实际基准 HEAD 和生产文件哈希，三个 LR 模型均正常保存。
- 正式评分首次仅在评价器的单层缓存计数预检查处停止；没有产生预测。随后将只读检查修正为与生产路径相同的 split 子目录查找，失败日志保留为 `evaluation_precheck_failed.log`。
- I 达到固定 `max_iter=10000` 仍未收敛；按预注册设置保留该结果，没有调整 C、容差或迭代数。
- 本轮至 G–K 即停止；没有启动多 seed、LOSO、特征搜索、日期模型或新损失实验。
