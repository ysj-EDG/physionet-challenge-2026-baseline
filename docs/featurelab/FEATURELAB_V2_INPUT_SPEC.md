# FeatureLab v2 输入规格（1444维候选池）

状态：CACHE_ONLY_QC 完成；本文件描述已验证的旧807维与Batch1新637维候选池，不授权训练、筛选、Rank256或迁移到Branch A。

## 输入行与来源

- 行数：6,600，顺序沿用已验证的G2-B1旧缓存索引。
- 记录对齐键：(site_id, record_id)；不按文件名排序配对，不使用内连接丢弃记录。
- 旧块：807个冻结独立Feature ID；新块：六个已批准Batch1家族共637个Feature ID。
- 每个列同时有数值、valid_mask和support_count；有效数值保留真实值（包括0），无效位置为NaN且mask为False。没有插补、缩放或整条记录删除。
- 本次只读取既有G2-B1和六家族最终缓存，不读取EDF，不运行旧提取器。

## 固定列顺序

[old807] + [eeg_tail288] + [eeg_temporal96] + [eeg_coherence_stage90] + [caisr_stage35] + [emg_stage72] + [resp_stage56]，总计1,444个独立特征。Demo5只在旧独立块出现一次。

实验集合列数：E0=807，E1=1,191（旧+EEG尾部+EEG时间组织），E2=1,060（旧+Coherence+CAISR+EMG+Resp），E3=1,444。

## 五视图候选位置

旧视图位置保持原顺序并追加新列：V1=161、V2=177、V3=493、V4=257、V5=376，总计1,464个位置。去重后的独立特征数仍为1,444。新列规划为：V1 Wake尾部48，V2 REM尾部48，V3 N2/N3/NREM尾部144，V4 All-stage尾部48加Temporal96，V5 Coherence90加CAISR35加EMG72加Resp56。

## 质量与隐私边界

候选池质控只公开家族/特征级聚合计数和合成规格；真实行值、支持分布、记录键、源SHA和私有路径保留在私有缓存目录。该输入规格不包含患者标签、TTE、预测、评分或模型产物。
