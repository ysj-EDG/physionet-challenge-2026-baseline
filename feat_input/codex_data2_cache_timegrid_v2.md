# data2：最小时间轴修复、有效性信息保留与全量特征缓存

基准仓库：ysj-EDG/physionet-challenge-2026-baseline。
参考提交：fc35c07f137cd1b26a277eaa2f70cfd8b0e8f4fc。
本任务与P3模型实验分开执行。旧npz_new、A–K模型和历史报告不得覆盖。

## 目标

data2约6600条记录尚无同版本NPZ。准备一个有版本号、时间轴正确、可续跑的完整特征缓存，随后在用户已有H100/Medex计算环境中实际启动全量提取。

不是重新设计生理特征，不因文献重要性删列，不启动data2模型训练。保持核心输出X_seq=483列、X_ecg=12列、x_static=196列，保留全部现有特征。三者可以具有不同时间长度，与已有滑动ECG语义一致。

此次允许的生产修改仅限：
1. 修正CAISR不可用阶段导致的时间轴压缩及事件时间伸缩。
2. 保存计算中已经存在的有效性信息和原始阶段信息。
3. 新增明确的提取版本和可靠批量缓存入口。

不得改变EEG频带、PSD算法、滤波、重采样、thr=100/dthr=45、相干性计算公式、EMG/HRV公式、标签定义或分类模型。不要迁移到新的GPU数值实现。

## 1. 先确认一个已知的、具体的问题，不开展全面审计

当前per_epoch_features/per_epoch_extractor.py的_extract_per_epoch_onehot使用：
raw_stages[valid]
删去不可用阶段后，又在extract_all中按最小长度将one-hot与原时间轴的EEG/EMG/呼吸前缀拼接。
feature_extractor_algorithmic.py的多个模块同样压缩stage，然后用有效阶段数×30秒反推事件采样间隔。

检查本地代码是否仍有上述逻辑。仅读取data2的CAISR标注及必要EDF头，统计未知阶段、标注与生理记录起始时间/长度差异。不要读取全部生理波形来做这项检查。此统计不使用结局标签选病例，不是全特征审计。

先用合成序列验证：N2、Unavailable、REM、N3四个30秒窗，必须仍有四个窗，REM仍对应60–90秒，而不能被移到30–60秒。

如果所有真实记录均无不可用阶段且起始时间一致，也要保留时间轴安全实现；报告说明该修复对这批记录实际值是否产生变化，不宣称它解释了此前A/B性能。

## 2. 实现独立提取模式timegrid_v2，默认旧行为仍可复现

通过明确参数或版本化入口启用timegrid_v2。旧checkpoint、旧API默认路径及旧缓存不得被静默切换。

### 2.1 时间坐标

- 保留原始stage_caisr位置，不删除9、NaN或其他非法码。
- 生理epoch时间范围以真实信号和EDF时间基准确定；占位模态不得决定整夜只有一个epoch。
- 正常CAISR为30秒阶段、0.5秒觉醒、1秒呼吸/肢体标注。优先读取各通道实际采样率；只在元数据不可得且确认符合官方格式时使用上述固定值，并记录来源。
- 读取并使用标注与生理EDF的起始偏移；不能用demographics中的CreationTime代替采集时间。
- 对于不能可靠确定的时间对应，明确记录失败/缺失，不猜测通过裁剪或插值拉伸来对齐。
- 只允许按真实时间范围裁掉超出部分；标注较短时对应区域是未知，而不是压缩其余时间。
- 维持现有ECG窗口定义和训练端offset=10。本任务不同时调整ECG窗口语义。

### 2.2 one-hot及事件覆盖

- 输出每个生理epoch的五阶段编码，顺序仍为N1/N2/N3/REM/Wake，对应CAISR码3/2/1/4/5。
- 未知阶段的五项全0，同时保存stage_valid=False；未知不等于Wake。
- 八项事件覆盖比例使用事件真实时间与30秒窗的交叠计算。
- 已知事件不因同一时段stage未知而消失；只有需要分期条件的统计才排除未知分期时间。
- 不用有效stage数×30/事件数组长度推算事件采样间隔。

### 2.3 静态CAISR的时间统计

保持186列的顺序、名称和物理单位，修复其时间基准：
- 记录总时间与有效分期暴露时间分别计算；TRT采用明确记录范围的时间，而非删除未知后的数组长度。
- TST来自可确认的睡眠阶段时间；分期占比、每小时事件率沿用相应睡眠时间分母。
- 潜伏期、前后半夜、事件onset及周期位置使用原时间坐标。
- bout和转换不能跨未知区间直接连接；未知处是边界而非一个可训练睡眠阶段。
- 分期事件率只使用在该阶段可确认的事件及暴露时间，不将未知暴露当正常睡眠。
- 没有足够观测的量保留可识别缺失/有效标记；不要将无法定义的数值伪装成临床正常0。
- 与本问题无关的CAISR后验概率变换、复合指标公式、阈值和字段不顺便重写。

这些修复即使列数不变，也构成新的提取语义版本。不要将timegrid_v2与历史缓存称为相同版本。

## 3. 必须保留的辅助信息，不直接扩大模型输入

在新NPZ附加键或独立数值sidecar中保存以下信息，消费者通过显式键读取，不改变核心数组列顺序：
- extraction_version、代码SHA、配置/通道表哈希、依赖版本及记录身份；
- epoch_start_sec、epoch_duration_sec及用于对齐的EDF时间偏移；
- 原始/对齐后的stage_code、stage_valid；有五阶段概率时一并保存及对应有效标记；
- EEG六导联channel_available；
- 每epoch、每导联的实际clean_subsegment_count和total_subsegment_count或其比例；
- HRV窗口计算成功标记；原mask继续只表示ECG时间对齐，不重命名为EEG质量；
- 可取得的阶段/事件标注可用性、实际模态长度和失败状态。

注意：当前eeg_segment_coherence的第二个返回量pvalues虽然注释写有效比例，但实际初始化为0后没有更新。不能直接将它保存为真实质量率。质量应来自已经用于PSD计算的art_flags/clean_counts；缺失导联不得因零信号计算产生的数值被标成有效。

以可选return_metadata或等效轻量方式透传，保证关闭该选项时旧数值路径不变，不重复滤波/FFT，不额外计算新特征。暂不添加纺锤波、慢波检测、CAP、Hjorth、SpO2、SleepFM或完整PSD大缓存。

## 4. 版本与标签

- 输出到新的data2缓存目录，例如npz_data2_timegrid_v2，不写入npz_new。尤其注意npz的回传结果和h100的进度检测，新建/database/home/gaohaojie/workspace/python-example-2026/npz_data2
- 不把typed_v1缩放或任何全数据拟合参数写进原始特征数组。缓存是原始提取输出，包含本来就定义为log的特征。
- 标签取自data2官方demographics，以明确0/1值保存/关联；不要用旧train/val/test文件决定data2哪些患者有标签。
- BidsFolder+SessionID标识记录；BDSPPatientID用于未来患者级分组，缺失时有明确替代规则。
- 1103小集是大集子集，不把两份同人记录相加。旧缓存只有在提取语义、输入来源和全部配置可证实一致时才可复用；本任务默认不复用旧npz_new。
- data2完成后仍不能套用B的缩放中心/尺度；训练时按各折训练成员重新拟合。

## 5. 只做有限验收，然后启动全量

合成测试：
1. 不可用阶段不移位；缺分期不等于Wake；事件时长不被压缩。
2. unknown前后的bout不连通，事件的时间定位不变。
3. 正常完整分期、采样率和时长一致的合成输入，新旧输出在未涉及修复处一致。
4. 质量比例来自真实flags，不是常数pvalues；关闭metadata输出不改变生理特征值。

实际试跑：每站点2条，共6条；仅按ID/时长/通道完整性选取，不按模型成绩筛选。优先覆盖一个不可用分期案例（若存在）。使用真实官方通道名及EDF单位，核对X_seq/X_ecg/x_static形状、时间轴、有限性、阶段/质量统计，记录用时和峰值内存。不得因为有真实异常样本就自动放宽伪迹阈值。

通过后按已有Medex/集群方式实际启动全量任务，不只生成计划。该启动授权是执行本prompt时的任务要求，不要求GPT在对话中后台运行。

当前提取核心为edfio、NumPy/SciPy等CPU/I/O计算，H100不会自动加速它。根据6条实测确定CPU并行数、内存及磁盘预算；控制各worker的BLAS线程，避免进程数×内部线程过量。不要为提高GPU利用率改写计算算法。3天仅是用户的预估，报告实测吞吐后估算总时长。

每记录原子写入临时文件后rename；已有完整同版本缓存跳过；失败单独记录，可续跑；禁止用全0伪造成功。不因单个坏记录推倒重跑全部。记录实际提交的job ID、日志路径和恢复命令，不自动安排与本次无关任务。

## 6. 最小交付

- 可复现的版本化提取入口和批量命令；
- 六例试跑的DATA2_CACHE_START.md：修改范围、时间轴问题计数、实测吞吐/内存、真实启动状态；
- 全量manifest：每条成功/失败/已存在、路径、版本、时间长度、模态和质量摘要；
- 最终成功人数、站点/标签人数及异常计数。

不生成新的691列全面审计报告，不运行神经网络训练，不在任务中修改论文模型方案。P3在旧缓存上的分期对齐读取有单独prompt，两个任务的目录、分支/快照和日志互不覆盖。

## 核验来源

当前入口：https://github.com/ysj-EDG/physionet-challenge-2026-baseline/blob/fc35c07f137cd1b26a277eaa2f70cfd8b0e8f4fc/per_epoch_features/per_epoch_extractor.py
当前静态CAISR：https://github.com/ysj-EDG/physionet-challenge-2026-baseline/blob/fc35c07f137cd1b26a277eaa2f70cfd8b0e8f4fc/per_epoch_features/feature_extractor_algorithmic.py
当前EEG质量计算：https://github.com/ysj-EDG/physionet-challenge-2026-baseline/blob/fc35c07f137cd1b26a277eaa2f70cfd8b0e8f4fc/per_epoch_features/eeg_sleep_features.py
官方标注时间分辨率：https://moody-challenge.physionet.org/2026/