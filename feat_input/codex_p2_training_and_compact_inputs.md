# P2：保留原始NPZ，检验类别加权与紧凑分类器

## 任务与边界

目标仓库：ysj-EDG/physionet-challenge-2026-baseline。
基准提交：e048052f033a62e95ee911fab151faa773cfffde。
参考报告：output/p1_ablation/seed7/P1_ABLATION_RESULTS.md。

请实际实现并训练下述五个固定实验，完成共享推理与官方评分；不要只提交方案或预检查。允许使用现有H100 Medex执行环境，沿用其已验证的数据挂载和命令。不要擅自增加实验。

本轮从“输入消融”推进到两个明确的问题：
1. 相同LSTM和typed_v1输入下，取消第二次正类加权是否改善泛化？
2. 基于现成静态特征的低容量分类器，是否比长序列网络更稳健？静态维度增加是否有增量？

原始特征提取器、原始NPZ、691列规则、数据成员和标签全部冻结。不重新提取、不覆盖NPZ、不增加日期/随访变量、不换data2或LOSO、不做SleepFM。不修改已有P0/P1报告或结果文件。

P1中的A/B/C在P6000，D/E/F在H100/eeg_env。因此本轮首先实际训练同环境G作为对照，不重新运行全面审计。G与H必须在同一H100、同一环境、同一数据、同一代码版本运行。原A/B及P1只作为历史参考，不能将跨环境差值描述为严格单变量效应。

## 一、固定实验表

| 实验 | 输入 | 分类器/训练目标 | 唯一问题 |
|---|---|---|---|
| G_B_sameenv | 完整typed_v1，无mask，无clip_z | 原LSTM、原平衡采样、原BCE pos_weight=680/53 | 在当前H100/eeg_env重跑B作为同环境对照 |
| H_single_balance | 与G完全相同 | 原LSTM、原平衡采样；仅BCE pos_weight改为1.0 | 第二次正类加权是否需要取消 |
| I_LR_demo10 | typed_v1处理后的静态前10列 | 固定正则逻辑回归 | 人口学基线 |
| J_LR_compact30 | 同10列人口学+预先固定的20列CAISR | 与I相同的逻辑回归 | 精简生理统计的增量 |
| K_LR_static196 | 全196列静态输入 | 与I/J相同的逻辑回归 | 扩展静态维度的增量 |

G/H是严格配对训练。I/J/K是相同分类器、缩放和正则下的特征集比较。LR与LSTM之间同时改变了输入表示和分类器，必须称作方案比较，不能声称仅改变网络结构。不要组合去BMI、去年龄、去相干性或clip5。

## 二、共同数据与预处理

1. 定位现有npz_new和冻结split，沿用P1真实的733训练、158验证、158test、54external名单；不按目录名重新划分。
2. 只读取现有B_typed_v1/lstm_model.pt中的input_preprocessing，复用其训练733人拟合得到的typed_v1规则、中心和尺度。优先使用P1现有来源路径与哈希记录，不凭空指定另一份文件。
3. 显式清空mask，clip_z=None。绝不加载B的网络权重、优化器、校准器或阈值作为新模型初始状态。
4. 同一原始NPZ仅在内存中转换一次。输入预处理状态随各模型保存。
5. G/H及I/J/K都沿用同一训练患者、标签和评价年龄。原始年龄单独保留，不能从标准化后列反推。
6. 沿用P1真实标签和prevalence文件，NPZ中的-1不得用于评分。train使用自然733人名单，而非平衡采样后的重复记录。
7. 不允许自动缺缓存后重提取或默默换用别的缓存。出现缺缓存时报告受影响步骤，保留已完成产物，不改数据来绕过。

## 三、G/H：只增加一个明确的损失配置

在train_lstm.py中为正类权重增加简洁选项，例如：
LSTM_POS_WEIGHT_MODE=empirical 或 unit。
默认empirical必须保持当前行为；unit使用BCEWithLogitsLoss(pos_weight=torch.ones(1, device=DEVICE))或数学等价实现。

G设置empirical；H设置unit。两者都保留原WeightedRandomSampler。

不得同时修改自然采样、网络结构、输入维度、dropout、batch、Adam、lr、梯度裁剪、调度、80 epoch上限、patience15、验证AC-AUROC选模及Platt/Youden规则。不得添加weight decay或focal/ranking loss。

两者从头初始化，同seed=7，独立进程顺序训练。保留目前训练末尾跳过external无效标签评分的补丁。保存checkpoint时附实际损失配置，推理不需要使用该训练配置。

只做必要的轻量检查：同seed下初始权重相同；同一batch除pos_weight外的输入/标签相同；两个loss均有限。不要建立新的预检查框架，不额外遍历DataLoader污染正式训练随机状态。通过后立即执行完整训练。

注意：G/H优化的loss不同，原始训练loss数值不能直接横向比较。使用最终自然训练集和验证集的AUROC、AC-AUROC、AUPRC判断泛化差距。

## 四、I/J/K：从缓存静态列构建固定基线

### 4.1 固定输入列

以下均为x_static的零基索引。人口学为0:10，保持现有编码，不重新编码，不添加缺失指示。

J额外的20列预定义为：

| 索引 | 现有特征名 |
|---:|---|
| 11 | caisr_sleep_tst_sec |
| 12 | caisr_sleep_se |
| 13 | caisr_sleep_sol_sec |
| 14 | caisr_sleep_rem_latency_sec |
| 16 | caisr_sleep_waso_sec |
| 17 | caisr_sleep_n1_pct |
| 18 | caisr_sleep_n2_pct |
| 19 | caisr_sleep_n3_pct |
| 20 | caisr_sleep_rem_pct |
| 32 | caisr_sleep_transition_rate |
| 34 | caisr_sleep_short_bout_ratio |
| 87 | caisr_sleep_mean_stage_entropy |
| 96 | caisr_arousal_arousal_index |
| 97 | caisr_arousal_arousal_burden_ratio |
| 98 | caisr_arousal_mean_arousal_duration |
| 126 | caisr_respiratory_respiratory_event_index |
| 127 | caisr_respiratory_respiratory_burden_ratio |
| 141 | caisr_respiratory_mean_resp_duration |
| 168 | caisr_limb_limb_movement_index |
| 173 | caisr_limb_plmi |

索引列表：
[11,12,13,14,16,17,18,19,20,32,34,87,96,97,98,126,127,141,168,173]

用已冻结feature_schema/feature_rules验证名称和索引对应。若拼写格式不同但来源公式/位置完全一致，记录对应关系；若语义或索引实质不符，报告该分支，不按相关性临时替换列。不要重建全部schema。

I：range(10)；J：range(10)+上面20列；K：range(196)。按此顺序保存selected_indices与名称。不要数据驱动挑选20列。

这里的30维方案是受CinC论文启发、适配现有缓存的对照，不是对某篇论文的逐特征复现。不能将transition_rate称为transition_entropy，也不能将含多种事件的respiratory_event_index直接重新命名为临床AHI。

### 4.2 线性模型的缩放和超参数

1. 从每个患者取typed_v1后的196维静态向量，不引入时间序列汇聚。
2. 仅在自然733人训练数据上额外拟合一个StandardScaler，对全部196列逐列中心化/标准化，然后按I/J/K各自列集选取。
3. 三个LR共用同一组逐列mean/scale，因此共同列的数值完全一致。零方差按StandardScaler的正常scale=1处理，不擅自删列。
4. 额外标准化是为了线性正则的尺度可比性。它属于LR分支自己的预处理，不改变typed_v1，也不回写LSTM输入。必须保存在checkpoint中；验证/推理只transform。
5. 三个模型均使用scikit-learn LogisticRegression：elastic-net，solver=saga，l1_ratio=0.4，C=0.03，class_weight=balanced，max_iter=10000，tol=1e-4，random_state=7，fit_intercept=True。按照现有安装版本使用语义等价参数，不升级环境。
6. 自然训练733行只出现一次，不额外WeightedRandomSampler，不再叠加sample_weight。class_weight仅用于训练一次。
7. 不搜索C、l1_ratio或特征数，不因test/external分数差而更改方案。收敛警告、全零系数要如实报告，不能据此声称该特征集普遍无用。
8. 使用验证集decision_function输出拟合既有Platt校准和Youden阈值；不使用test/external。逻辑回归没有best epoch，写N/A。
9. 保存非零系数和对应特征名称、截距、训练迭代数。系数不等于因果或临床解释，不能据它自动删特征。

## 五、统一保存与官方推理

G/H沿用现有LSTM checkpoint与推理。

LR新增一个很小的StaticLogisticModel或等价共享推理模块，载入选列、额外StandardScaler参数、LR权重与截距，输出未经校准的decision logit。不要将sklearn训练分类器的predict_proba再当作logit校准。

为保持run_model.py和evaluate_model.py不变，可在team_code.load_model中按checkpoint.model_type选择模型；字段不存在默认原lstm。尽量让新线性模型兼容现有forward(X_seq,X_ecg,x_static,lengths)签名，只使用x_static，从而复用现有_sample和校准/阈值流程。

checkpoint保存输入预处理状态、model_type、selected_indices及名字、线性缩放、系数、校准和阈值。可沿用现有lstm_model.pt文件名作为兼容容器，但日志和报告必须标明实际是LR，不冒称LSTM。

不要改现有LSTM分支的网络定义或A-F checkpoint行为。不修改原始提取、官方run_model.py/evaluate_model.py，不复制另一套指标函数。

必要回归测试仅包括：
- 旧B checkpoint在关闭新选项时结果不变；
- LR保存/加载后的decision logit与sklearn一致（合适浮点容差）；
- 同一NPZ训练侧与推理侧的选列/尺度一致；
- 未在验证或测试拟合缩放；
- 模型加载不会混用分类器/预处理状态。

这些测试是接入测试，不应演变为新的多轮全量审计。完成后实际训练、推理。

## 六、运行与评价纪律

1. 沿用P1 H100/eeg_env的已有执行方式，G/H始终相同软硬件。LR用同一软件环境，CPU训练即可。
2. 固定seed=7；五组独立输出，不覆盖A-F，也不覆盖已经存在的新结果目录。
3. 记录实际HEAD、允许的代码差异、输入manifest/规则/预处理状态标识、环境版本。简洁复用已有记录方式，不重造全仓库指纹系统。
4. 五组训练/验证选择和校准结束后，再一次性执行train、val、test、external的原有run_model.py+evaluate_model.py。明确LSTM_NPZ_CACHE指向同一份冻结缓存。
5. 不用run_model.py的失败补齐选项，不遗漏失败患者。预测覆盖需与固定名单一致。
6. train采用自然733人；val/test/external不做重新校准、选阈值或调参。训练过程中G/H必要的val选模保持原有逻辑。
7. 额外记录原始decision logit和校准后概率，若Platt系数非正或发生数值饱和，报告原始/校准后排序差异；不要暗中翻转评分。
8. 遇到数值或数据错误，保留已有结果并说明，不私自改batch、lr、clip、特征集合或训练标签。
9. 本轮不自动扩展多seed、LOSO、日期模型、特征搜索或新的原始信号提取。

## 七、结果交付

建议目录output/p2_model_controls/seed7，五个子目录分别对应G/H/I/J/K。
每组保存checkpoint、训练日志、train/val/test/external预测与官方scores/table。

统一P2_MODEL_RESULTS.md包含：
- 每组的实际输入、模型、损失、维度、参数量/非零系数数目；
- G/H最佳epoch、停止epoch；LR迭代与收敛情况；
- 自然训练和验证的AUROC、AC-AUROC、AUPRC，以及训练-验证差；
- test/external七项官方指标和TP/FP/FN/TN；
- H-G、J-I、K-J三个预定义对比；
- 历史A-F参考表，并明确跨硬件限制；
- 各组额外StandardScaler和typed_v1状态的保存情况；
- 运行异常和缺失结果，N/A不伪造数字。

模型选择只依照验证集。现有test/external已多次被观察，明确为探索性结果，不能作为新的独立最终检验。

解释限定：H优于G才支持本配方去除第二次正类加权；J优于I支持当前精简CAISR在相同线性模型下的增量；K不优于J只说明本设置下扩展静态列未显示增益。不得将一次结果推广成某模态/某生理特征普遍无效。

完成以上五组就停止，下一轮再根据结果决定保留哪个训练目标、是否压缩时序表示，以及是否进入预先固定的数据2站点验证。

## 依据（用于理解，不要求重新检索或下载）

- P1固定提交报告：https://github.com/ysj-EDG/physionet-challenge-2026-baseline/blob/e048052f033a62e95ee911fab151faa773cfffde/output/p1_ablation/seed7/P1_ABLATION_RESULTS.md
- CinC 384：49项统计/人口学特征、强正则LR，以及复杂模型的隐藏验证对照。https://cinc.org/2026/Program/accepted/384_Preprint.pdf
- CinC 463：精简CAISR与站点可辨识性分析。https://cinc.org/2026/Program/accepted/463_Preprint.pdf
- CinC 59：特征组的正/负增量不能依据数量推断。https://cinc.org/2026/Program/accepted/59_Preprint.pdf
- CinC 116：可供下一阶段参考的特征族表示与分期汇聚；本轮不实现。https://cinc.org/2026/Program/accepted/116_Preprint.pdf