# Codex任务：P0原始输入裁剪修复（实施、验证，不启动完整训练）

## 0. 目标、范围和事实边界

目标仓库：ysj-EDG/physionet-challenge-2026-baseline。

先阅读当前仓库的 feat_input/AUDIT_REPORT.md、cache_inventory.json、feature_schema.json、feature_stats.csv、candidate_transform_summary.csv、audit_feature_scaling.py、run_log.txt，以及 train_lstm.py、team_code.py 和相关特征提取器。当前审计记录的提交是 0b8eccf4b9668d78d93e3882b0fb3b3f6e759e4f；这不代表执行时HEAD一定相同。记录实际HEAD、工作区差异和本次改动文件，不要 reset、clean、覆盖用户已有改动，不自动提交或推送。

任务：用显式逐列规则、仅训练集拟合、训练推理共享的输入变换器，替代三个输入分支的原始数值统一 clip(-50,50)。本轮完成代码、单元测试、离线变换报告和小规模前向/反向验证，不自动启动正式训练。

保持不变：特征提取公式、缓存原始数值、NPZ顺序、LSTM架构、损失、采样策略、split、训练超参数、官方评分公式、ECG的现有时间对齐。不要顺便修复双重类别加权、CAISR时间轴、导联阈值或增加新模态。不要给模型新增缺失指示维度。本次只修复输入接口及直接相关的缺失/占位处理。

已审计的是1103条历史缓存，不是6600条data2。代码必须支持任意合法训练manifest，但本次验证优先使用审计已确认的733条训练记录，不能把733或1103写死成缩放器限制。旧缓存没有完整提取版本元数据，不得声称历史缓存与当前提取器必然一致；把这一点保存在报告及变换器的来源说明中。

## 1. 必须认识到的审计结论

1. 1103条年龄原值范围50到88；94.8323%严格大于50，其余等于50。因此旧clip后年龄整列都是50，不只是部分记录发生改变。评估用的原年龄单独保留，不要改成标准化年龄。
2. HRV_MedianNN约99.0631%的窗口大于50ms；原中位数约930ms。pNN20约33.6929%的窗口大于50，其单位是百分数。
3. 多项CAISR总时长约98%以上被裁到50。CAISR的阶段pct通常已经是0到1，而BSR和pNN20是0到100，不能按pct字符串统一除100。
4. BMI约75.884%为0。当前load_bmi将无法读取/NaN回退为0；有效BMI不能等于0，拟合BMI统计时必须排除这些占位。
5. emg_chin_envelope_iqr_norm的原始极值约1.63e15，尽管p99约2.51、超过50的比例很小。原始均值/标准差会被异常尾部支配。log变换可以压缩数值，但不能证明极端比值是有效生理信息；上游分母/质量问题单独记录，暂不修改提取器。
6. 审计D没有实际执行log，而是回退C。因此不能把D列当成已验证的选择性log方案。
7. 审计脚本通过名称含ratio/pct/prob等决定identity，误放行了无上界比值，例如N3/N1。不得直接复用其自动类型推断作为生产策略。
8. IQR为0不等于整列常量，尤其是BSR、BMI和稀疏事件。审计的constant_or_tiny_scale不是充分的常量证据。
9. 旧mask主要代表ECG对齐，不能用来判定EEG质量。NaN已在部分提取路径被补零，NPZ的NaN率不等于原始缺失率。
10. 审计中的计数/极值是全量计算，分位数来自每条记录最多64行抽样；候选预览也不能替代全量变换后的最大值和有限性检查。

如以上事实与当前文件冲突，以实际代码和数据为准，列出差异，不要强行套用旧结论。

## 2. 代码组织与公共接口

建议新增一个轻量运行时模块 feature_scaling.py，及显式规则文件 feat_input/feature_rules_v1.json。也可采用等效结构，但禁止在推理时导入整个审计脚本、读取审计CSV、从目标患者/站点重新估计尺度。

建议接口：

- FeatureScaler.fit(train_records, schema, config)：只接受唯一训练记录列表；从原始缓存拟合。
- FeatureScaler.transform_arrays(X_seq, X_ecg, x_static, ...)：不原地修改输入；不拟合；返回三个变换后数组。mask保持原语义。
- state_dict()/from_state_dict()：使用可序列化基本类型保存/恢复规则、参数和版本。
- 训练和推理共用同一transform实现，而不是分别复制规则。

精确维度：X_seq=483，X_ecg=12，x_static=196。序列和ECG拼接后仍为495。保留时间长度，不在变换器内部做重采样或改变ECG偏移。

规则表必须覆盖全部691个源特征且恰好一次，每条包括：分支、索引、精确名称、特征族、公式/来源、原单位、固定换算因子、非线性变换、参照尺度、缩放策略、理论范围（有证据才写）、缺失规则、零值语义、审计来源与来源不确定性。

允许按精确特征族位置循环生成相同结构的规则，但须断言预期名称和顺序。运行时禁止以“名称含ratio/pct/prob/log”等宽泛字符串为兜底判定；任何未分类列都应报错，不能默认为identity或默认再log。

现有feature_schema.json可提供顺序，不能把其中基于名字猜出的语义视为已批准规则。先用当前提取公式校对后，输出独立的批准规则表。

## 3. 首选方案 typed_v1：固定规则，再拟合训练尺度

对需要缩放的连续列，定义：

u = x / unit_scale
v = g(u)
z = (v - center) / scale

unit_scale和g必须在规则文件中事先明确，不能通过看测试分布临时选择。默认g为恒等；只有本提示词列出的未取log的非负长尾类使用log1p。已经取log的值保持符号，不得重复log，不得统一abs。

默认统计缩放：center为训练中位数，scale为训练IQR。理论有界且要求直通的列不做中心化；百分数先转为比例。

不采用按患者单独z-score、不对整行196维静态向量做归一化、不按验证/测试站点重新拟合。不要加入分位数高斯化、CORAL、站点秩变换或标签相关特征选择。

### 3.1 EEG频谱：X_seq[0:54]

每通道9项，共6通道。依据eeg_sleep_features.py逐项确认：8项为已经取log10的功率或功率比，1项为SEF50（Hz）。这些列均采用恒等g，再按训练列稳健缩放；已log列的负号必须保留，不再log或abs。SEF50也不因为非负而自动log。

### 3.2 EEG相干性：X_seq[54:414]

共15对通道，每对24项，校对精确顺序：

- 各频带mean、IQR、低/高sigma均值，以及归一化谱熵：确认公式给出0到1后直通。
- 频带AUC是频率积分，不是0到1的均值；采用恒等g和训练稳健缩放，不裁到1。
- 频谱centroid、bandwidth（Hz）：恒等g和稳健缩放。
- sigma/delta、alpha/delta、beta/delta、high/low四类无上界比值：log1p(x/1)，再稳健缩放。

合法的单列零值保留。分母异常或safe_div回退产生的零值，若缓存已无法区分，不臆造缺失掩码；报告此限制。

### 3.3 BSR：X_seq[414:432]

18列百分数，固定除100后直通。0可以是真实无抑制，不作为缺失；不能因为IQR=0而放大稀疏非零值，也不能删掉整列。

### 3.4 EMG：X_seq[432:456]

下颏、左腿、右腿各8项：

- log_rms_norm、log_hf_lf_ratio：已经取log，保留符号，恒等g后稳健缩放。
- envelope_iqr_norm：log1p(x/1)后稳健缩放。
- burst_rate_per_min：log1p(x/(1 event/min))后稳健缩放。
- active_fraction、tonic_fraction、burst_duty_cycle、phasic_mini_epoch_fraction：直通0到1。

不要认为log已经解决了1e15比值的来源；在报告中保留原始极值及受影响记录数。不得修改信号QC阈值或提取器的分母。

### 3.5 呼吸：X_seq[456:470]

- airflow_cycle_cv、airflow_amp_local_norm、airflow_amp_iqr_over_median、thorax_amp_local_norm、abdomen_amp_local_norm：都是无上界非负比值候选，使用log1p(x/1)和稳健缩放。
- airflow_rate_bpm：恒等g和稳健缩放。
- airflow_longest_reduction_sec：恒等g，保留秒单位并稳健缩放。
- thorax_abd_lag_sec：保留符号，恒等g和稳健缩放。
- 各reduction/near_absent/paradox_fraction：直通。
- thorax_abd_corr：保留[-1,1]，直通。

### 3.6 阶段与事件：X_seq[470:483]

5项阶段one-hot和8项事件覆盖比例按当前定义直通，不要当成13项one-hot。全零StageEvent块的意义若无法从原数据恢复，只标记问题，不据此把整个EEG序列删除。

### 3.7 ECG：X_ecg[0:12]

逐项核对名称后使用以下规则：

0. model_HRV_MedianNN：ms除1000换秒，恒等g后稳健缩放。
1. model_HRV_MCVNN：log1p(x/1)，稳健缩放。
2. model_HRV_CVNN：log1p(x/1)，稳健缩放。
3. model_HRV_CVSD：log1p(x/1)，稳健缩放。
4. model_HRV_pNN20：百分数除100，直通。
5. model_log_HRV_LF：已经取自然对数，恒等g后稳健缩放，保留负号。
6. model_log_HRV_HF：同上。
7. model_HRV_HF_rel_LFHF：是HF/(LF+HF)，不是LF/HF，按0到1直通。
8. model_HRV_SD1SD2：比值不保证小于1，log1p(x/1)后稳健缩放。
9. model_HRV_Symbolic_EqualProb4_0V：确认公式后按0到1直通。
10. model_HRV_Symbolic_EqualProb4_2UV：同上。
11. hrv_circadian_cos：[-1,1]直通。

### 3.8 人口学：x_static[0:10]

- index0年龄：原始年龄先单独保存给评分；模型输入年龄用恒等g和稳健缩放。非有限或<=0视作不可用，不自定50/80等生理截断。
- index1:4性别、index4:9种族的原有编码直通，不重编码。
- index9 BMI：非有限或<=0视作缺失；仅对正的有限值拟合中位数和尺度，变换时使用训练中位数填补；不增加缺失维度。记录每个split实际缺失率，不把所有其他列的0照此处理。

### 3.9 CAISR：x_static[10:196]

这是最需要逐项校对公式的部分，必须输出186行完整规则，不允许“其余CAISR一律同一种变换”。以下分组决定第一版，不按测试集表现自动更改：

A. 大尺度睡眠总时长：trt_sec、tst_sec、sol_sec、rem_latency_sec、wake_time_sec、waso_sec、各stage的dur_*_sec，先秒除3600换小时，恒等g后稳健缩放。有效时长不统一log，也不裁到50秒。

B. 睡眠bout/cycle的均值、中位数、最大值和分位数时长、连续睡眠/阶段最长持续时长：log1p(seconds/60)，再稳健缩放。该处理是本次预注册的长尾压缩选择，不宣称由审计证明最优。

C. 各类event/bout/cycle计数：log1p(count/1)，再稳健缩放。

D. 各类事件率和正的阶段分层率：确认原单位后，统一到每小时（必要时固定换算），log1p(rate/(1 event/hour))后稳健缩放。不要对有符号early/late差值使用普通log1p。

E. 无上界正比值，包括n3_n1_ratio、n3_n1n2_ratio、rem_nrem_ratio、rem_to_nrem_ratio，以及分母为TST、但分子未必限制在睡眠期的各类burden_ratio：log1p(ratio/1)，再稳健缩放。不要因名称有ratio而假定<=1。

F. 单事件持续时间、事件间间隔、事件簇持续时间、事件关联延迟等非负秒数：log1p(seconds/60)，再稳健缩放。只有公式可证明非负才能用这一组；有符号延迟或差值归H。

G. 真正的阶段占比、效率、事件构成份额、覆盖比例、后验概率统计及其它由公式证明有界的量：保持已有尺度直通；CAISR的stage_pct已经是0到1，不再除100。mean_stage_entropy当前是未除log(5)的熵，不能假称已经归一到0到1；保持其[0,log(5)]尺度直通即可。posterior_volatility依公式保留其自身有界尺度，不擅自裁到1。

H. delta_n3、delta_rem、delta_w等有界占比差保留符号并直通；delta_ari、delta_resp_index、delta_limb_index等有符号率差用恒等g和稳健缩放。deep_sleep_preservation_index、rem_integrity_index等可以为负的复合量同样保留符号并稳健缩放，不abs、不普通log1p。

I. 非负复合burden/fragmentation量不等同于标准的概率或事件率。检查公式，如为非负混合指数则用log1p(index/1)和稳健缩放，并明确单位为原定义的复合指数；如可为负则归H。不要把复合指数伪标成events/hour。

若某列无法按公式确定归组，将其列为阻塞项，先解决语义映射再交付完整typed_v1；不得通过静默兜底把未知列纳入正式变换。

## 4. 拟合、抽样、尺度退化和数值边界

### 4.1 只拟合训练

从当前明确的train manifest读取唯一记录，在WeightedRandomSampler启用前拟合。验证/测试/外部记录、重复采样次数、标签比例不能参与中心和尺度估计。不得用全1103条的审计分位数直接填写生产参数。

序列和ECG只用模型实际消费的真实时间步。当前ECG偏移10个epoch，实际消费窗口数为max(0,min(T_ecg,T_seq-10))；与collate和_infer_one一致，不把被截掉的尾部ECG窗口纳入拟合。不要改变这个历史偏移。

静态特征每位患者计一次；序列拟合采用受试者等权的可复现抽样：默认每条记录、每个序列分支至多128个真实窗口，以固定seed及稳定record id派生随机种子，无放回抽取。每一列在某患者内有效抽样值的总权重为1，再合并患者；该列没有有效值的患者不参与该列。不得使用进程随机化的Python hash来派生种子。若采用等效精确实现，说明方法并保证不让长记录或重复采样支配尺度。

分位数可近似，但报告必须说明样本量、每记录上限、权重方式和seed。全量变换后的计数、有限性和最大值不能仅用抽样代替。

### 4.2 IQR退化

对需要中心化的列：center=训练加权中位数。

scale候选依次为：IQR、p95-p05、同一训练有效分布的标准差；只有有限且大于数值容差的候选才可使用。建议数值容差为1e-6*max(1,abs(center))，它是变换后坐标下的计算保护，不是临床阈值。全部候选退化则scale=1。

区分：真实全常量、样本IQR为0但存在少量非零、全缺失、近常量。常量列保留维度；不要强行除以1e-8，不因训练样本常量而把测试中的所有新值也覆写为0。全缺失列用确定性的回退并在报告中标红，不临时查看验证值拟合。

### 4.3 缺失填补和有效范围

先识别缺失，再拟合和变换。连续列在g变换后以训练中位数填补，中心化后为0。直通列无法识别的旧零值保持原语义；新出现非有限值使用规则中明确的固定/训练填补值并计数，不得统一把所有0当缺失。

对有公式证明的硬边界（BSR百分数、概率等），允许仅对浮点舍入级越界作容差修正；实质越界应单独标记无效、报告和按缺失规则处理，不能用硬clip掩盖单位错误。无上界比值不做0到1裁剪。普通log1p只适用于已批准非负列，实质负值报异常并按规则处理，不abs。

### 4.4 不立即引入另一个任意统一裁剪

typed_v1默认clip_z=None。不要把原始clip(-50,50)简单改成标准化后clip(-5,5)或(-10,10)。全量报告abs(z)>5、10、20、50的比例及最大值；这些是诊断刻度，不是预设删除阈值。

可以保留显式配置的标准化后数值安全边界接口，但默认关闭；若使用，必须写入checkpoint并逐列报告实际裁剪数量，且不能依据外层测试成绩自动选择。发现仍有数值灾难先列出具体列和原因，本轮不要自动试探阈值直到外部成绩变好。

统计和非线性变换使用float64内部计算，最终float32。检查转换后有限性；不把溢出静默转成大数或0。错误信息要含分支、特征名和记录标识。

## 5. 占位/失败块处理：有证据才做，不能扩大推断

1. 序列padding、ECG尚未对齐的前10个epoch、空ECG：保持模型输入0。这些占位不参与拟合。先变换真实窗口，后执行现有对齐和padding。
2. team_code._fallback_arrays生成的明确整段回退样本：保留序列占位0，不把零向量减去训练中心后伪造生理信号；静态中仍可用的人口学独立变换。
3. 历史HRV前11项同时精确为0：将其定义为本版显式的legacy_zero_hrv_sentinel处理假设，因为有效MedianNN不可能为0且该模式与失败补零一致。只对这11项排除拟合并输出0，第12项昼夜cos按其自身规则保留。记录数量，并明确这只是缓存级失败模式假设，不是已恢复原始质量真值。单个LF/HF log值等于0不能据此认定缺失。
4. CAISR整186项精确全0且TRT为0，可作为显式legacy_zero_caisr_block回退假设：整块不参与拟合、变换后保持0；记录数量和假设。不能把正常的零事件、零BSR、单独全零StageEvent块扩展为整夜缺失。
5. 不能把ECG mask=False用于删EEG、改睡眠阶段或判HRV质量。未知的EEG补零来源本轮保留并报告，不臆造逐通道质量标签。
6. 所有上述缺失策略都写入checkpoint，训练/推理保持完全一致，且与legacy_clip模式严格区分。

## 6. 接入现有训练代码

先查清PSGDataset、_build_tensors、collate_fn和main的现有职责。

- 从未变换x_static[0]保存raw_age，评分始终使用它；不能用填补或缩放后的年龄代替。
- train manifest确定后，读取原始数组拟合一次preprocessor，再把同一个冻结实例传给train/val及可用外部数据集。
- _build_tensors不再对typed_v1执行旧raw clip；同时避免在进入共享变换器前无条件nan_to_num而丢失刚读到的缺失信息。
- 不将已变换数组写回原始NPZ，不在每个epoch/worker重复fit。
- 采样器、pos_weight、LSTM层数、hidden、fc维度、dropout、optimizer、calibration策略和阈值流程本轮不改。
- checkpoint保存input_preprocessing完整状态、模式与版本、691列顺序hash、规则hash、拟合配置、train唯一ID集合hash及manifest文件hash、数据来源说明、拟合列有效数/退化标记、代码HEAD等。
- 优先以数值列表和基本字典保存参数，避免引入额外第三方Scaler对象的pickle兼容负担。
- 显式支持typed_v1与legacy_clip两种训练模式用于对照；新训练默认typed_v1，legacy分支必须复现旧nan处理和clip的实际顺序。

## 7. 接入推理和旧模型兼容

重点检查team_code中的_normalise_arrays、_sample、load_model、_infer_one、run_model和train_model。

- 数组读取/shape验证与数值变换分离。typed_v1路径在共享变换器之前不得先把所有非有限值抹成0。
- 缓存与临时inference_cache保持原始提取值；每条样本送网络前恰好transform一次。不得在_sample与_infer_one各做一次。
- load_model从checkpoint恢复对应preprocessor，不从CSV或目标患者拟合，不调用fit。
- 新版typed_v1模型缺参数、版本不支持、规则或列顺序不匹配：明确报错，不能静默identity、改走fallback样本或使用别的checkpoint的Scaler。
- 旧checkpoint没有preprocessor时，只允许显式legacy兼容开关恢复原始clip路径，例如LSTM_ALLOW_LEGACY_INPUT=1，并输出可见警告。不得自动把新Scaler接到旧权重上。
- typed_v1模型须重新训练；本次只做兼容与烟雾测试，不声称旧权重可直接迁移。
- 保留官方train_model/load_model/run_model签名、固定提交模式和临时缓存工作流。
- 不删除预测概率[0,1]裁剪、梯度裁剪等与本问题无关的clip；只替换三组原始特征的统一clip。
- 注意宽泛except：预处理版本/特征顺序错误不能被当作缓存读取失败而吞掉。

## 8. 测试：必须有可运行证据

至少实现以下测试，测试名和通过/失败结果写进报告：

1. 全部691列精确映射一次；483/12/196维度及输入495保持；错序、漏列、重复列立即失败。
2. 人工年龄60/70/80经新变换保持不同且有序；评估raw_age仍为60/70/80。不要用这些人工值替代真实拟合统计。
3. BSR=0/50/100得到0/.5/1；pNN20正确除100；CAISR_n2_pct=.5仍为.5。
4. NN=800/1000/1200ms、新时长序列保留区别；旧legacy确实会把>50值压成50。
5. 已取log的值不能再log或abs，x与-x的差别不能被合并；中心化后符号不必与原值相同，测试应检验信息/顺序保持而不是强制z与x同号。无上界ratio=2/100/1e6可以合法输入，且不被裁成1。
6. 构造一个1e15的EMG envelope ratio，确认typed_v1 log分支输出有限，正常0.3和0.6经float32变换后仍能区分。
7. IQR=0但存在稀疏非零时不除极小epsilon，非零信息不被清空；训练真常量列在新样本变化时仍按保存规则处理。
8. BMI=0/NaN不参与拟合，变换为中位数对应的0；有效BMI正常变换；BSR/event的合法0不被当BMI缺失。
9. 空ECG、前10个epoch、padding和明确fallback序列保持0；HRV全11零哨兵只作用于11项，昼夜cos不误删；mask不误用于EEG。
10. 变换前后时间长度和ECG对齐位置一致；拟合没有读入实际被截掉的ECG尾窗。
11. 修改val/test/ext输入为极端值，不改变拟合后的任何参数；不能增加这些记录来补某训练列缺失。
12. 同一record经训练数据路径与推理数据路径，先比较变换后三数组，再比较有效对齐张量；相同权重eval模式下logit一致（明确容差，例如atol=rtol=1e-6）。必要时固定CPU减少硬件差异。
13. 保存/加载preprocessor后数值等价；再次处理同一原始缓存结果可复现且缓存未改变；不得重复缩放。
14. typed_v1缺状态必须报错；显式legacy开关能复现旧输入，不能静默混合。
15. 小batch一次CPU前向和反向，loss、logits、所有实际得到梯度的参数均有限。dropout/训练和eval模式按测试目的设置。

不需要跑整轮训练即可完成这些测试。单元测试使用人工小数组；烟雾测试只取少量合法训练样本，不能混入验证样本更新权重。

## 9. 全量离线变换报告，不用模型分数代替工程验证

在当前可复核的train manifest拟合typed_v1后，对1103条现有缓存分别按train/val/test/external及site分组transform，仅统计，不训练。

生成feat_input/p0_fix/下的结果（可按项目规范调整名称）：

- IMPLEMENTATION_REPORT.md：改动、证据、测试、限制、下一步训练命令。
- feature_rules_v1.json：691列明确规则及来源。
- scaler_fit_summary.json：训练来源hash、参数/退化/缺失统计、抽样说明。
- transformed_feature_stats.csv：每列原始、legacy_clip、typed_v1的全量计数/有限性/极值，分位数可抽样但注明；包含abs(z)>5/10/20/50的比例、零/缺失/填补/无效/回退数量。
- unresolved_input_issues.md：无法恢复的零值来源、历史版本未知、极端EMG比值、未分类或越界列等，不隐藏风险。
- tests.log和可直接复现的命令。

特别展示年龄、BMI、MedianNN、pNN20、TST/TRT/WASO、BSR、EMG envelope ratio、N3/N1 ratio的前后对照；解释分位数变化而非只列“NaN=0”。不要把IQR=0统一写成constant。

审计显示test/external缓存标签为-1，不能把它们当正常二分类标签，不得报告伪造的AUROC或异常回退0.5。离线转换报告无须用标签；如验证脚本调用指标，先检查有效二分类标签及类别数，无法计算时明确N/A。不要修改官方评分公式来凑出结果。

只有单元测试、路径一致性、有限性、缓存未被覆盖、所有列完成映射且来源边界明确后，才能写“P0工程修复通过”。不能据此写“预测性能已经提高”。

## 10. 本轮结束时给用户的输出

先摘要实际发现和实现情况，再列变更文件、测试结果、当前训练命令（只展示不自动执行）、报告路径和剩余风险。

后续第一组正式实验只比较legacy_clip与typed_v1：同数据、同split、同seed、同采样和损失、同架构、同训练预算。改良标准化与其它模型结构留到独立消融，不同时修改。若之后比较StandardScaler，应该在同一已批准的单位/log/缺失规则之后替换尺度估计，而不是对存在1e15极值的原始数组直接z-score。

缩放规则、缺失处理和年龄恢复都属于这次输入修复；提升不能直接全部解释为EEG信息变多。后续可在同一验证协议下单独做人口学消融，本次不运行。

不得自动执行全量重提取、长时间训练、大规模调参、安装大型依赖、git提交/推送或移动数据。完成实现与验证后停止，等待用户根据报告决定下一轮实验。## 补充验收边界

- 历史NPZ没有feature_names或版本时，不能仅通过维度断言就证明历史列语义正确。本次允许以当前提取器顺序作为明确记录的legacy缓存假设继续实施；若NPZ自带名称/版本则必须逐项验证。报告必须区分“运行时schema一致”与“历史提取来源已验证”。
- 全量变换统计优先报告模型实际消费窗口上的分布；同时保留缓存原始窗口数作追溯。尤其ECG原始尾窗可能没有送入模型，前后裁剪比例比较须使用一致分母，不能把978740原始窗口与对齐后的有效窗口混算。
- 对第一版预定规则的任何偏离，都要写出具体特征、代码公式证据和原因；不要依据外层测试分布或分数自动选择更有利的变换。
- typed_v1并非只有尺度变化，还显式处理了BMI缺失和可识别失败块。正式对照应准确称为“输入预处理修复”；要隔离某一项的因果贡献需另做消融，不能把总增益全部归因于RobustScaler。
