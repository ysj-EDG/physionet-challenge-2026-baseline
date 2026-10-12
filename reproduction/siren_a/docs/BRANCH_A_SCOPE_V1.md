# SIREN Branch A Scope V1

状态：G-1 已冻结范围；G1 之前不得执行特征计算、训练、NPZ 改写或任何下一个 Gate。

## A. 实验目标

本实验仅实现 SIREN-inspired Branch A。利用既有 1103/6600 NPZ，将逐 epoch 多模态特征转化为患者级、固定维度的全夜特征。

本实验不复现 Branch B，不进行自监督预训练，不使用 CAISR20 低容量残差。

## B. 数据来源

输入 NPZ 的既定字段和维度为：

- `X_seq`：483 维 / epoch；
- `X_ecg`：12 维 / 滑动窗口；
- `x_static`：196 维 / 患者。

保留上游提取逻辑与现有数据版本。EEG 频谱来自原有 100 Hz 特征计算流程；本项目不重新重采样 EEG，也不改为 SIREN 的 128 Hz。读取时必须使用对应版本的分期及模态有效性元数据。

本项目不修改或覆盖任何已有 NPZ。P1 审计确认 6600 v2.0 与 6600 v2.1 缓存可在旧工作区只读访问；1103 缓存当前为 `UNAVAILABLE`。不得声称已经找到 1103 缓存，不得自行搜索替代缓存或重新提取。

## C. 分阶段 EEG 频谱

使用现有每 epoch 六导联、每导联九维的 EEG54。

固定睡眠阶段集合：

- Wake；
- REM；
- N2；
- N3；
- NREM：N1 + N2 + N3；
- All-stage：所有有效睡眠阶段。

每一阶段仅对有效 EEG 特征计算 Mean 和 IQR。每阶段输出 108 维，六阶段共 648 维。不得自行加入 Median、STD、Min、Max 或其他统计量。

## D. EEG 空间特征

采用相对于宽频段总功率 `P0` 的四频带对数相对功率：

- Delta；
- Theta；
- Alpha；
- Sigma。

其值由现有 EEG54 中的频带功率比、`logPbeta` 和 `logP0` 派生。

空间特征的生理定义为：

- **Interhemispheric Asymmetry**：同源左右 EEG 导联的相对功率差异；
- **Anterior–Posterior Gradient**：双侧额叶平均相对功率与双侧枕叶平均相对功率的差异。

空间特征必须在同一有效 epoch 上计算，再按睡眠阶段汇聚。当前候选方案为 36 维，但不强求与 SIREN 原论文的 20+16 精确一致。最终列名、适用阶段及计算细节由 G1 审核冻结。

Codex 不得自行增加导联对、频带或其他空间比较公式。

## E. 非频谱特征

使用现有 NPZ，不新增原始 EDF 特征提取：

- CAISR20：原有 20 维整夜特征；
- HRV11：仅使用 11 个 HRV 指标，Mean + IQR，得到 22 维；
- EMG24：Chin、Left leg、Right leg，每路 8 维，Mean + IQR，得到 48 维；
- Resp14：Mean + IQR，得到 28 维；
- Demo5：保留 SIREN 风格人口统计学变量，不将年龄作为模型输入。

非频谱生理特征暂定 118 维，Demo5 独立计算。Demo5 的种族编码和缺失值规则在 G1 审核后冻结，本 Gate 不自行决定。

## F. 明确排除的特征

第一版不使用：

- Coherence360；
- BSR18；
- SpO2 新提取特征；
- Circadian cosine；
- ECG predicted age gap；
- Age residualization；
- 新的 Spindle、Slow Oscillation、SO–Spindle Coupling 提取器；
- 自监督 EEG Embedding；
- CAISR20 预测残差。

## G. Feature Views

采用 SIREN 五视图组织：

- V1：Wake EEG + Demo5；
- V2：REM EEG + REM 空间特征 + Demo5；
- V3：N2、N3、NREM EEG + 对应空间特征 + Demo5；
- V4：All-stage EEG + Demo5；
- V5：Non-spectral + Demo5。

当前候选独立特征池约 807 维，五视图总计约 827 列。以上包含尚待 G1 冻结的空间特征和 Demo 编码，不得将估计列数当成可以自行调整的硬目标。

## H. 双层缓存设计

第一层是现有原始 NPZ，永久只读。

第二层是新建的 Night-level Feature Cache，由 G2 生成。Night-level 缓存保存：

- 五视图的原始全夜特征；
- 每列有效性/缺失状态；
- 阶段与模态有效样本数量；
- Feature Manifest；
- 来源 NPZ 版本和指纹。

Night-level 缓存是确定性特征产物，不包含通过疾病标签学习的变换。不得在全夜缓存生成阶段拟合标准化器、缺失值填补器、站点分位数映射或分类模型。训练折相关的特征变换和相关产物必须单独保存。

1103 和 6600 数据的缓存来源必须分别记录，禁止混用不同 NPZ 版本或覆盖原始文件。

## I. 下游模型

后续使用 SIREN Branch A 的：

- L2 Logistic Regression；
- HistGradientBoosting；
- LightGBM LambdaRank。

实验组织为五视图 × 三学习器 × 15 Seeds。使用模型预测结果的百分位排名进行 Branch A 内部集成。

学习器超参数、缺失值处理、站点百分位映射及训练集选择的具体实现，需分别在后续 Gate 审核冻结。当前没有 Branch B，因此不执行 A/B = 0.6:0.4 融合。

## J. Gate 顺序

- G-1：Git 分支、范围文档、提交与远端核验；
- G1：固定 Feature Manifest、空间特征、有效性及缺失规则；
- G2：读取原始 NPZ，生成独立 Night-level Feature Cache；
- G3：实现已审核的站点分位数及 Fold-specific 输入处理；
- G4：冻结三种学习器和训练协议，完成 Branch A 训练；
- G5：LOSO 结果、融合评估和特征消融。

既有 Validity 审计视为前置依据，不重复开展新的完整数据审计。但实际缓存版本及字段仍需要在读取时执行断言。

每个 Gate 必须经用户与 GPT 审查通过才能继续。G-1 完成后立即停止，不进入 G1。
