# P3 v2：正确时间轴上的分区EEG汇聚与重复分折

本文件替代尚未执行的codex_p3_compact_pooling_and_repeated_cv.md，不同时执行两版。
仓库：ysj-EDG/physionet-challenge-2026-baseline。
基准：fc35c07f137cd1b26a277eaa2f70cfd8b0e8f4fc。

## 目的

回答三个问题：
1. 紧凑CAISR相对人口学的增量能否重复出现？
2. 控制阶段存在信息后，低维EEG统计是否增加预测信息？
3. 相同EEG统计按睡眠阶段组织，是否优于整夜混合？

不重训G/H，不调裁剪、不新增日期、随访/诊断输入、SleepFM、树模型或新的神经网络。当前实验只读旧npz_new，不能覆盖NPZ、A–K权重和报告。允许读取原始CAISR标注及EDF头恢复可靠的阶段时间坐标，不重提EEG或其他生理特征。

data2的timegrid_v2缓存任务独立执行，不混入本轮733人CV，也不拿其全局拟合参数用于本实验。

## 1. 先处理上一版P3隐含的时间轴假设

参考源码中，_extract_per_epoch_onehot先删除stage_caisr不属于1–5的行，再将结果与原时间轴的生理特征前缀拼接；静态CAISR也有类似时间压缩。不能默认历史NPZ第470:475列一定与同一行EEG对应。

只做本任务所需的时间核对，不开展全特征审计：
- 从既有记录名单定位每条CAISR标注及生理EDF头，读取起始时间、采样率和原始stage码。
- 根据历史缓存真实生成入口确定X_seq每行的物理时间：通常是从生理EDF起点开始的30秒连续前缀，但必须通过源码/元数据确认，不能仅凭shape猜测。
- 对照缓存one-hot是否匹配原始时间序列或压缩后的序列，记录是否受影响；不能把压缩索引当作EEG索引。
- 为P3单独生成sidecar：record_id、epoch_start_sec、真实对齐stage_code、stage_valid、时间映射来源及对齐状态。
- 对可确认的连续前缀，在原有N行范围内使用正确的原始阶段；历史裁掉的生理尾部不能凭空恢复，也不为此重算PSD。
- 缺失/无法可靠对齐时，该记录仍保留，但阶段相关EEG摘要为缺失；不能删患者以改善结果。报告无法对齐人数和原因；若任一来源整体无法对齐，暂停EEG两臂并继续静态基线，明确不能检验分期假设。
- 不把ECG mask当作EEG质量mask，不把所有特征0猜为缺失。当前缓存没有可追溯EEG质量时，保留既有值并报告限制。
- 不使用人工作者分期代替CAISR分期。

本轮x_static仍使用旧缓存原值，所有方案一致；不偷偷把新版CAISR静态值替换进部分方案。报告说明：本轮检验已缓存生理信息的新汇聚，未整体修正旧缓存的静态定义。新版本data2上的模型需另行重新拟合。

## 2. 五个预先固定方案

同一LR目标，只改变输入集合，不在结果出来后继续增删候选：

P3_demo10：P2的I，人口学10维；用于补充数值收敛的参考。
P3_compact30：P2的J，相同10维人口学＋相同20项CAISR。
P3_compact35：compact30＋5个阶段存在标记。
P3_global59：compact35＋24项整夜EEG摘要。
P3_stage155：compact35＋120项分期EEG摘要。

增加compact35是为了隔离“阶段存在信息”与“EEG信息”，避免将这两者一起加入后声称所有增益来自EEG。

核心对照：compact30-demo10；compact35-compact30；global59-compact35；stage155-global59。
后三个方案具有相同5个存在标记。155维不预设优于59维；它是一个必须证明收益的较丰富候选。

## 3. 24/120项EEG摘要的定义

### 3.1 固定特征与空间分组

使用X_seq前54列（六导联×每导联九项），核对冻结FEATURE_NAMES与schema：
- log10(Ptheta/Pbeta)，导联内索引3；
- log10(Pdelta/Pbeta)，索引4；
- log10(Psigma/Pbeta)，索引1；
- SEF50，索引5。

不得把log(theta/beta)命名成theta相对总功率，不得把sigma比值叫纺锤波密度。这里只借鉴论文的信息组织方式，不是逐列复现。

六导联保留三组，不再全部折成一个中位数：
- F：F3-M2、F4-M1；
- C：C3-M2、C4-M1；
- O：O1-M2、O2-M1。

导联顺序按当前提取器核对，不用字符串模糊匹配猜索引。

### 3.2 时间汇聚

每个CV训练折拟合typed_v1后，对记录执行一次转换，clip_z=None，空屏蔽配置。已log特征不再log。

对每个导联、每个目标时间集合、每项特征计算两个量：时间中位数和IQR=p75-p25。再分别对同一脑区左右两个导联的该统计取中位数。必须先在每个导联内计算时间IQR，不能把左右导联和时间混成一个数组计算IQR。

- global：所有带有效CAISR阶段的可用历史epoch，4项×3脑区×2统计=24。
- stage：N1、N2、N3、REM、Wake分别计算，4×3×2×5=120。
- 两组使用相同的基础时间范围，不用global额外包含未知阶段而stage排除。
- 一个目标集合少于3个epoch时，该集合的EEG统计设为缺失。这是本轮预先固定的最低数值稳定条件，不是临床阈值，不根据分数调整。
- 阶段存在标记表示该记录原有N行物理范围内是否存在至少一个有效的该阶段epoch；它不代表EEG干净率。
- 已知缺导联可由可追溯metadata或时间核对时读取的EDF头确认，并排除该导联占位值。不能仅因九项均0就推断缺导联。没有可核验信息时使用旧缓存值并注明限制。
- 整夜无可靠对齐阶段时，五个存在标记为0，EEG摘要全缺失；不删除患者。

新特征命名需包括stage/global、F/C/O、实际指标名、median/iqr，顺序固定并写入checkpoint。

## 4. 为什么不同于旧P3

旧P3只取时间中位数并合并六导联，会主动丢掉时间波动及脑区差异。CinC 365报告N2额区theta相对功率波动、特定周期慢振荡等重要性；59报告beta波动、theta相对功率波动和REM觉醒率。

本轮因此保留少量脑区和IQR，但不新增原始信号特征，不直接添加全部6×9×阶段×统计组合。不能将本轮结果解释为对论文所述相同特征或特定阶段因果作用的验证。

## 5. 重复分折：不是733折，也不是选最高的一折

仅使用原train的733人，三折重复三次，分折seed=7、17、29。
- 按BDSPPatientID分组；确无该字段时使用可验证的患者ID。不同session不能跨折。
- 尽量保持SiteID×label组合分层；使用StratifiedGroupKFold或当前环境等价实现。五组共用完全相同的9套折。
- 每折打印训练/留出人数、阳性数、站点构成、有效年龄配对数。不得搜索一个分数高的划分seed。
- 原val158、test158、external54不参与候选选择、折内拟合或收敛决策。
- 这些折来自历史训练来源的混合分布，不能称为跨院验证；data2三站点LOSO是后续独立阶段。

每折仅用折内训练记录拟合typed_v1的中心/尺度、汇聚后中位数填补和StandardScaler。不能加载B在全部733人上拟合的state进行折内验证。
冻结规则而不是冻结跨折参数。同一折五模型共享对应列同一处理，避免重复拟合或细微差别。
静态列选择复用P2的确切20项名单。无需fit任何label相关特征筛选。

汇聚后的全缺失列固定填0并保留列，记录原因。其他缺失用该折训练中位数。额外标准化也只拟合训练记录。合法类别编码不改定义。

## 6. 固定线性目标与收敛

全部五组：
LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.4,
                   C=0.03, class_weight='balanced', fit_intercept=True,
                   max_iter=100000, tol=1e-4, random_state=7)
使用float64矩阵，不另加sampler或sample_weight。

与P2相同统计目标，只提高最大数值迭代预算；不是根据验证分数决定训练轮次。
出现ConvergenceWarning时保留该折状态、系数及结果，明确未达到数值收敛标准。不要临时改C、tol或换solver并声称同一实验。其他已收敛折继续。未收敛模型不能作为已确认生理增量的参照。

固定C在不同训练规模下的实际正则作用不必完全相同，报告中如实说明本轮是固定配方下的分折实验，不声称估计了精确的样本数因果效应。

## 7. 评价与选择

- 每折保存训练及留出患者的raw decision logit；用仓库官方实现计算AC-AUROC(gap=2)、AUROC、AUPRC。
- CV留出折不拟合Platt或阈值。排序评价直接用logit。
- 先算每折指标，再算每次重复的三折均值，最后报告三次重复均值和范围。
- 逐个配对报告差值和方向。九折相互相关，不当作九个独立试验计算显著性；不把所有不同折logit拼起来算一个总AUROC代替折均值。
- 站点子组无法计算时写N/A，不填0.5。
- 选择主指标为重复CV平均AC；差值小于0.01时优先较低维模型，该0.01只是预声明的简约决策规则而非统计显著性阈值。
- 不能用原val、test或external决定是否增加阶段、脑区或统计量。

无需先加多组神经网络或小网格搜C。五个预定义方案全部跑完再评价原外部集合，途中不看外部结果改变方案。

## 8. 全733人最终拟合和部署路径

重复CV结束后，五方案分别用全部733人重新拟合本轮变换/汇聚填补/标准化和LR，不继承B的网络权重或校准状态。
使用原val158拟合各自Platt及Youden阈值，再一次性评价原train/val/test/external；真实标签与prevalence沿用P2，不读取NPZ中的-1当真值。

新增明确model_type，例如pooled_logistic_v2。checkpoint保存：
- 缓存/时间映射profile、字段顺序、脑区/阶段/统计定义；
- 折内或最终typed参数、填补、额外标准化、LR系数；
- Platt、阈值及训练名单；
- 对旧缓存所需sidecar的来源与校验，缺少时间证据时不得静默退回可疑one-hot。

训练与推理调用同一个共享汇聚函数。旧A–K模型行为保持不变。
官方评估入口继续可用；读取旧缓存时显式提供P3时间sidecar，读取新timegrid_v2缓存则使用其metadata，但不得把两个profile混在同一训练/评价结果中。未经验证，不用新缓存评估旧checkpoint后声称同一实验。

## 9. 最小测试后直接运行

只测试：
- 人工不可用阶段不移位，且真实缓存时间映射可追溯；
- 实际维度10/30/35/59/155；
- median/IQR计算正确且F/C/O不互混；
- 缺阶段、缺导联和少于3个epoch不删患者；
- 原年龄评价输入不被改变；
- 训练/推理、保存/加载一致；
- 修改折外值不改变折内拟合参数。

通过后实际运行重复CV与最终拟合，不再生成新的全特征审计体系。运行文件与data2长任务使用独立快照和目录，不能运行中改共享源文件。

## 10. 交付与停止

建议output/p3_pooled_v2/：
- 五组固定配置与清晰列名；
- 时间sidecar及ALIGNMENT_NOTE.md（只说明对齐问题及范围）；
- 九套折manifest、每折分数/患者logit；
- 最终五个checkpoint及官方预测/scores；
- P3_V2_RESULTS.md。

报告回答：人口学是否收敛；精简CAISR是否稳定增量；阶段存在标记是否解释增量；EEG是否在该参照上增加信息；分期汇聚是否优于整夜；训练—留出差距如何变化。
未观察到增益时，只能说本轮选定的四类谱特征/脑区/统计与LR没有显示增量，不能说整个EEG无用。

本任务到此结束，不自动删除核心缓存列，不自动启动data2训练、LOSO、学习曲线或GRU。data2缓存可以与P3并行生成；得到本轮结果后再冻结少量候选进入站点验证。

## 来源

P2：https://github.com/ysj-EDG/physionet-challenge-2026-baseline/blob/fc35c07f137cd1b26a277eaa2f70cfd8b0e8f4fc/output/p2_model_controls/seed7/P2_MODEL_RESULTS.md
CinC 116（特征族和阶段汇聚）：https://cinc.org/2026/Program/accepted/116_Preprint.pdf
CinC 365（阶段、脑区、波动的重要性探索）：https://cinc.org/2026/Program/accepted/365_Preprint.pdf
CinC 59（REM觉醒率、频谱波动等）：https://cinc.org/2026/Program/accepted/59_Preprint.pdf
CinC 175（稀疏EEG相对CAISR的站点消融）：https://cinc.org/2026/Program/accepted/175_Preprint.pdf