# P3 v2：正确时间轴的分区 EEG 汇聚与重复分折结果

## 结论

- **人口学基线没有可靠数值收敛。** `P3_demo10` 的 9 折平均留出
  AC-AUROC 为 0.567，但每个重复各有一折达到 `max_iter=100000` 仍未收敛。
  因此它只作为低维参考，不作为已确认收敛的主参照。
- **固定 20 项 CAISR 的平均增量较小且不完全稳定。** `compact30-demo10`
  的 9 折平均差为 AC +0.0138、AUROC -0.0039、AUPRC +0.0236；三个重复
  的 AC 方向为正、负、正。它支持保留 compact30，但不能声称增量在所有
  分折重复中稳定出现。
- **阶段存在标记没有解释出额外收益。** `compact35-compact30` 为
  AC -0.0079、AUROC -0.0029、AUPRC -0.0045，三个重复 AC 均下降。
- **整夜低维 EEG 只显示有限、指标依赖的增量。** `global59-compact35`
  为 AC +0.0173、AUROC +0.0029、AUPRC -0.0217，三个重复中两次 AC
  提高、一次基本持平。global59 相对 compact30 的净 AC 增量仅 +0.0094，
  未达到预声明的 0.01 简约门槛，而且 AUROC 几乎相同、AUPRC 更低；本轮
  仍选择更低维的 **compact30**。
- **按阶段展开到 155 维明确恶化 CV。** `stage155-global59` 为
  AC -0.0965、AUROC -0.0349、AUPRC -0.0288，且训练—留出 AC 差扩大到
  0.332。当前四类谱特征×脑区×统计的分期表示在该 LR 配方下过拟合；
  这不能外推为“所有 EEG 无用”。
- test/external 是预先固定实验完成后的泛化观察，未用于增删方案或选模。
  stage155 的 test 较好但 external 和重复 CV 较差，不能据 test 反向选择它。

## 实际版本与执行边界

- 指令基准：`fc35c07f137cd1b26a277eaa2f70cfd8b0e8f4fc`；实际启动 HEAD：
  `c68f0010d13ad3a92afa2993b33873662165462c`。运行位于未提交工作区，
  没有自动 reset、提交或推送。
- 冻结输入为旧 `npz_new` 1103 例、固定 split 733/158/158/54、
  `typed_v1`、`clip_z=None` 和原 691 列规则；规则 SHA256：
  `b8624092b12e5a2800d7627413d6d4c791fec8c7fb77552a007e1d43342c3c90`。
- 未重新提取或覆盖 `npz_new`，未使用 data2 缓存，未把 test/external NPZ
  中的 `-1` 当真实标签。官方评分沿用 P2 的固定真实标签和 prevalence。
- 训练启动源文件和哈希保存在 `source_snapshot/`。CV 和最终拟合均使用
  `LogisticRegression(elasticnet, saga, l1_ratio=0.4, C=0.03,
  class_weight=balanced, max_iter=100000, tol=1e-4, random_state=7)`；
  每折只在折内训练人群拟合 FeatureScaler、缺失中位数和 StandardScaler。
- `run_model.py` 和 `evaluate_model.py` 未修改，SHA256 分别为
  `09941cca...9021` 和 `227e40a4...0970`。`team_code.py` 只新增明确的
  `pooled_logistic_v2` 加载/推理分支；旧静态 LR 和 LSTM 加载回归检查通过。
  新分支调用与训练相同的 `assemble_features`/`apply_imputer_scaler`，没有
  复制一套汇聚公式。

## 时间轴核对

- 1090/1103 条记录可按官方生理 EDF 起点后的连续 30 秒前缀可靠对齐；
  13 条缺少可读取 CAISR 标注，患者保留且阶段 EEG 摘要按缺失处理。
- 历史缓存 one-hot 对可读记录匹配“移除无效阶段后的压缩序列”，不能作为
  同行 EEG 的分期索引。本轮使用独立 sidecar 的原物理前缀 stage_code。
- 1103 条中 13 条没有有效阶段，另有 31 条至少缺少一个可追溯 EEG 导联。
  没有把全零特征猜作缺导联，也没有把 ECG mask 当 EEG 质量 mask。
- 完整边界见 `ALIGNMENT_NOTE.md`、`stage_sidecar/manifest.csv` 和
  `stage_sidecar/summary.json`。

## 3×3 折重复 CV

表中为先计算每折、再计算每个重复三折均值、最后对三个重复取均值；范围是
三个重复均值的最小—最大值，没有把不同折 logits 拼接成单个 AUROC。

| Arm | 维度 | Train AC | Holdout AC（范围） | Holdout AUROC | Holdout AUPRC | 未收敛折 |
|---|---:|---:|---:|---:|---:|---:|
| demo10 | 10 | 0.613 | 0.567（0.537–0.587） | 0.776 | 0.260 | 3/9 |
| compact30 | 30 | 0.743 | 0.581（0.544–0.637） | 0.772 | 0.283 | 0/9 |
| compact35 | 35 | 0.749 | 0.573（0.539–0.627） | 0.769 | 0.279 | 0/9 |
| global59 | 59 | 0.783 | 0.590（0.537–0.658） | 0.772 | 0.257 | 0/9 |
| stage155 | 155 | 0.826 | 0.494（0.453–0.530） | 0.737 | 0.228 | 0/9 |

| 固定比较 | Δ AC | Δ AUROC | Δ AUPRC |
|---|---:|---:|---:|
| compact30 − demo10 | +0.0138 | -0.0039 | +0.0236 |
| compact35 − compact30 | -0.0079 | -0.0029 | -0.0045 |
| global59 − compact35 | +0.0173 | +0.0029 | -0.0217 |
| stage155 − global59 | -0.0965 | -0.0349 | -0.0288 |

逐折训练/留出 logits、checkpoint 和分数在 `cv/`；完整分折名单、站点构成、
阳性数、gap=2 年龄配对数与分组无重叠检查分别在
`cv_fold_manifest.csv`、`cv_fold_checks.csv`。每个 seed 的 733 人都恰好
作为留出样本出现一次，训练/留出 BDSPPatientID 无交叉。

## 全 733 人最终拟合与原 val

以下排序指标来自最佳最终 checkpoint 的自然名单推理；val 只用于各模型的
Platt 校准和 Youden 阈值，不参与上述 CV 候选选择。

| Arm | Train AC / AUROC / AUPRC | Val AC / AUROC / AUPRC | Train−Val AC |
|---|---|---|---:|
| demo10 | 0.614 / 0.792 / 0.256 | 0.468 / 0.670 / 0.250 | 0.146 |
| compact30 | 0.723 / 0.833 / 0.320 | 0.539 / 0.689 / 0.259 | 0.184 |
| compact35 | 0.726 / 0.834 / 0.313 | 0.547 / 0.691 / 0.257 | 0.179 |
| global59 | 0.750 / 0.853 / 0.303 | 0.405 / 0.631 / 0.187 | 0.345 |
| stage155 | 0.786 / 0.869 / 0.343 | 0.487 / 0.644 / 0.216 | 0.299 |

## Test 官方七项指标（158 人）

| Arm | Reward | AC-AUROC | Age-weighted | AUROC | AUPRC | Accuracy | F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| demo10 | -0.047 | 0.352 | 0.365 | 0.792 | 0.277 | 0.627 | 0.253 |
| compact30 | 0.158 | 0.679 | 0.616 | 0.848 | 0.488 | 0.886 | 0.471 |
| compact35 | 0.116 | 0.691 | 0.622 | 0.851 | 0.492 | 0.899 | 0.467 |
| global59 | -0.134 | 0.745 | 0.762 | 0.848 | 0.441 | 0.544 | 0.217 |
| stage155 | 0.086 | 0.752 | 0.707 | 0.859 | 0.523 | 0.791 | 0.353 |

## External 官方七项指标（54 人）

| Arm | Reward | AC-AUROC | Age-weighted | AUROC | AUPRC | Accuracy | F-measure |
|---|---:|---:|---:|---:|---:|---:|---:|
| demo10 | 0.481 | 0.552 | 0.589 | 0.705 | 0.269 | 0.574 | 0.378 |
| compact30 | -0.095 | 0.583 | 0.601 | 0.728 | 0.273 | 0.759 | 0.235 |
| compact35 | -0.099 | 0.583 | 0.601 | 0.728 | 0.273 | 0.778 | 0.143 |
| global59 | 1.168 | 0.604 | 0.610 | 0.745 | 0.302 | 0.426 | 0.311 |
| stage155 | 0.397 | 0.542 | 0.562 | 0.690 | 0.256 | 0.741 | 0.462 |

Reward、Accuracy 和 F-measure 使用各 checkpoint 在 val 上冻结的阈值；
排序指标不依赖该阈值。external 仅作预先固定的泛化观察，不据此继续调参。

## 混淆矩阵

| Arm | Test TP / FP / FN / TN | External TP / FP / FN / TN |
|---|---|---|
| demo10 | 10 / 58 / 1 / 89 | 7 / 22 / 1 / 24 |
| compact30 | 8 / 15 / 3 / 132 | 2 / 7 / 6 / 39 |
| compact35 | 7 / 12 / 4 / 135 | 1 / 5 / 7 / 41 |
| global59 | 10 / 71 / 1 / 76 | 7 / 30 / 1 / 16 |
| stage155 | 9 / 31 / 2 / 116 | 6 / 12 / 2 / 34 |

## 最终 checkpoint 状态

| Arm | 非零系数 | LR iter | Platt coef / intercept | 阈值 | Checkpoint SHA256 |
|---|---:|---:|---|---:|---|
| demo10 | 4/10 | 13186 | 0.723919 / -2.354099 | 0.071779 | `46b00f37...799f` |
| compact30 | 14/30 | 43 | 0.703293 / -2.327795 | 0.144195 | `580b8e35...5a1d` |
| compact35 | 16/35 | 53 | 0.707425 / -2.326847 | 0.156122 | `02f23909...a439` |
| global59 | 23/59 | 73 | 0.485976 / -2.303361 | 0.065397 | `68e2e4a4...b384` |
| stage155 | 45/155 | 90 | 0.541558 / -2.260388 | 0.099759 | `95cce63f...fe1` |

五个最终模型均收敛，且 checkpoint 明确保存 `model_type=pooled_logistic_v2`、
字段名、typed 参数、填补、额外标准化、LR、校准、阈值、训练名单和 sidecar
profile。单记录保存/加载回归中，官方推理与训练路径 logit 的绝对差为
`3.99e-16`。20 个官方评价组合均覆盖完整名单、顺序一致、logit/概率有限，
decision sidecar 概率与官方输出逐值一致；未使用 `run_model.py -f`。

## 异常与停止边界

- 初版 sidecar 键名错误和初版重复转换造成的失败/中断证据分别保留在
  `stage_sidecar_failed_badkey_20260910`、`cv_failed_badkey_20260910` 和
  `cv_failed_repeated_transform_20260910`；它们未混入正式结果。
- 正式 45 个 CV 模型中只有 demo10 的 3 折未收敛；未修改 C、tol、solver
  或迭代上限绕过。五个最终模型均收敛。
- data2 `timegrid_v2` 同步/缓存任务独立运行，未混入 P3。本轮到此停止，
  不自动启动特征搜索、新网络、LOSO 或调参。

机器可读汇总见 `p3_results.json`、`cv_results.json`、
`official_evaluation_manifest.json`；完整运行日志为 `run_optimized.log` 和
`official_evaluation.log`。
