# Matcha V18 Large6600 复现报告

状态：`complete`。源码、数据、E1、冻结 30 秒/192 维表征、三站点 LOSO、fold-local residual、全量最终模型和官方 `run_model.py` 端到端验证均已完成。

## 1. 固定源码、环境和数据

- 上游基准 tag/commit：`v18-final` / `7f2eae2fe4f170c87a21943b24ea4e217eabf67f`。
- 当前复现分支：`reproduction/v18-large6600`；本地工程提交为 `9d9485a`，仅包含缓存续跑、缓存指纹防混用、正式 cache 路径和远端阶段编排等工程改动，未改变模型结构、loss、采样、维度或训练 epoch。
- Conda：`sleepfm_env`；Python 3.10.20；torch 2.5.1+cu121；NumPy 2.2.6；pandas 2.2.3；scikit-learn 1.7.2；h5py 3.11.0；pyEDFlib 0.1.42；pyarrow 23.0.1。
- GPU：H100（E1、导出、LOSO、最终序列训练均在 H100 的 `eeg_env` 远端任务中完成）；本地验证使用双 Quadro P6000 环境。
- 原始数据：`/database2/physionet2026_kaggle/data2`；正式 prepared cache：`/database2/physionet2026_kaggle/matcha_cache/v18_large6600/prepared`，约 586G，未放置脚本/模型。
- 详细环境快照：`artifacts/v18_large6600/reports/environment.txt`。

## 2. Large6600 预处理

| 项目 | 结果 |
|---|---:|
| demographics 记录 | 6600 |
| 可用记录 | 6600 |
| 失败记录 | 0 |
| 阳性 | 498 |
| I0002 | 319（52 阳性） |
| I0006 | 1142（112 阳性） |
| S0001 | 5139（334 阳性） |
| prepared cache | 627,683,094,689 bytes（约 586G） |
| workers | 8 |

预处理保存的是原生 PSG/标注 HDF5 和 90 维 CAISR 汇总特征；它不是最终的 192 维冻结表征。冻结表征另存于下一节的永久目录。

## 3. E1 与冻结表征

- E1：seed `20260804`，15 epochs，正式使用 epoch 15 EMA teacher；训练完成，E1 总耗时约 77,692 秒。
- 原始正式 checkpoint：`h100_e1_results/e1_seed20260804_epoch15/checkpoint_epoch_15.pt`。
- checkpoint SHA256：`74261ac7451ef3f8710820e157733c72e97f4b112675ed520a1a9d161f1e47e8`。
- 永久冻结表征：`artifacts/v18_large6600/frozen/e1_seed20260804_epoch15/`。
- 记录数：6600；HDF5 分片：66；eligible 30 秒窗口：5,895,221；维度：192；存储 dtype：float16；分片约 2.3G。
- manifest、`window_index`、`complete_epoch_count`、`modality_valid`、stage/event logits、record/site 字段和 checkpoint 指纹均保留；全 66 分片校验通过，未发现 NaN/Inf 或临时分片混入。

## 4. 三站点 CI-label LOSO

外层站点为 I0002、I0006、S0001；inner split 15%，max epochs 20、minimum epochs 5、patience 4；pairwise 为 same-site、年龄差不超过 2 年、权重 0.15。LOSO 的原始 candidate 训练结果来自仓库现有脚本。下表的 `Raw` 是不加 residual 的 PSG sequence，`Residual` 是只用 outer-train 拟合日期/90D CAISR/follow-up 后按 `logit + 0.375 * adjustments` 的结果。

### 每个 seed、每个留出站点

| seed | held-out | n / pos | pairs | Raw AC / AUROC / AUPRC | Residual AC / AUROC / AUPRC |
|---:|---|---:|---:|---:|---:|
| 20260806 | I0002 | 319 / 52 | 1886 | 0.685578 / 0.729833 / 0.494938 | 0.888123 / 0.891530 / 0.781217 |
| 20260806 | I0006 | 1142 / 112 | 14493 | 0.655834 / 0.711148 / 0.225413 | 0.697164 / 0.750433 / 0.337407 |
| 20260806 | S0001 | 5139 / 334 | 182330 | 0.602764 / 0.676990 / 0.175257 | 0.689113 / 0.751157 / 0.300907 |
| 20260807 | I0002 | 319 / 52 | 1886 | 0.693001 / 0.738980 / 0.424474 | 0.906151 / 0.900389 / 0.777225 |
| 20260807 | I0006 | 1142 / 112 | 14493 | 0.694715 / 0.748149 / 0.308090 | 0.751259 / 0.793975 / 0.433833 |
| 20260807 | S0001 | 5139 / 334 | 182330 | 0.629737 / 0.684233 / 0.186403 | 0.699506 / 0.752755 / 0.302702 |
| 20260808 | I0002 | 319 / 52 | 1886 | 0.721633 / 0.740349 / 0.423118 | 0.896607 / 0.894339 / 0.780285 |
| 20260808 | I0006 | 1142 / 112 | 14493 | 0.682743 / 0.738436 / 0.270650 | 0.719451 / 0.774835 / 0.412777 |
| 20260808 | S0001 | 5139 / 334 | 182330 | 0.626364 / 0.694485 / 0.180447 | 0.694219 / 0.757576 / 0.312200 |

### 每个 seed 的 Macro/Worst AC

| seed | Raw Macro AC | Raw Worst AC | Residual Macro AC | Residual Worst AC |
|---:|---:|---:|---:|---:|
| 20260806 | 0.648059 | 0.602764 | 0.758133 | 0.689113 |
| 20260807 | 0.672484 | 0.629737 | 0.785639 | 0.699506 |
| 20260808 | 0.676914 | 0.626364 | 0.770092 | 0.694219 |

### 三 seed mean-logit ensemble

| ensemble | held-out | n / pos | pairs | AC / AUROC / AUPRC |
|---|---|---:|---:|---:|
| mean logits Raw / Residual | I0002 | 319 / 52 | 1886 | Raw 0.707317 / 0.749784 / 0.466394; Residual 0.900848 / 0.899453 / 0.793499 |
| mean logits Raw / Residual | I0006 | 1142 / 112 | 14493 | Raw 0.689919 / 0.745778 / 0.286365; Residual 0.728559 / 0.780062 / 0.410431 |
| mean logits Raw / Residual | S0001 | 5139 / 334 | 182330 | Raw 0.637471 / 0.705812 / 0.195215; Residual 0.698223 / 0.758418 / 0.314467 |

- 三 seed Raw Macro/Worst AC：`0.678236` / `0.637471`。
- 三 seed Residual Macro/Worst AC：`0.775877` / `0.698223`。
- pooled OOF（仅补充，不替代 Macro AC）：Raw AC/AUROC/AUPRC = `0.666516` / `0.725874` / `0.240510`；Residual = `0.721361` / `0.769930` / `0.379212`。

- 与 prompt 中的论文参考值比较：当前三 seed Raw Macro AC `0.678236`，比 M2 Raw 参考 `0.7273` 低 `0.049064`；当前 fold-local residual Macro AC `0.775877`，比 V14 residual 参考 `0.8039` 低 `0.028023`。两者训练条件和 E1 版本并非逐项相同，因此这里只作结果对照，不作为精确复现验收阈值。

LOSO 的原始 candidate gate 在三个 seed 均未通过（seed 20260806 的 Macro gain 为正但 S0001 有轻微下降；另外两个 seed 的 Macro gain 为负）。这被保留为真实结果，没有换 loss、换 seed 或重新调参。

由于仓库现有 LOSO parquet 保存的是 sigmoid 后概率而不是 raw logit，本报告的三 seed ensemble 使用 `logit(p)=log(p/(1-p))` 的确定性逆变换后求均值；没有概率平均。该协议差异已明确记录，未改变模型前向或训练逻辑。LOSO OOF 与 fold-local residual 文件：`artifacts/v18_large6600/loso/metrics/`。

## 5. 全量最终模型

- 复用同一份 E1 和冻结表征，不重新预训练或导出。
- seed `20260806`：2 epochs；seed `20260807`：4 epochs；seed `20260808`：1 epoch；三个任务均完成并有 `_SUCCESS`。
- 部署包：`artifacts/v18_large6600/final_model/raw_sequence_v18_full_training/`。
- 包含：`e1_encoder.pt`（EMA teacher）、`raw_sequence_seed_20260806.pt`、`raw_sequence_seed_20260807.pt`、`raw_sequence_seed_20260808.pt`、三个 residual JSON、`metadata.json`、`training_summary.json`。
- 全量残差由 6600 条正式记录拟合；运行时结构保持 `load_model` / `run_model` 接口。

## 6. 官方接口端到端验证

- 使用仓库原有 `run_model.py`，真实 EDF 各抽取一个记录覆盖三站点；执行于 tmux `matcha_runtime_validation`。
- I0002：`1024/1026` eligible epochs，Raw=ok，CAISR=ok。
- I0006：`741/748` eligible epochs，Raw=ok，CAISR=ok。
- S0001：`916/921` eligible epochs，Raw=ok，CAISR=ok。
- 返回码：`0`；输出：`artifacts/v18_large6600/runtime_validation/output/demographics.csv`；三条概率均有限。

这只是部署链路验证，不是独立泛化成绩；没有编造官方隐藏测试集指标。

## 7. 可重复运行与状态检查

从永久冻结 embedding 直接重跑 H100 cache-only LOSO（不会重跑 E1）：

```bash
cd /database/home/gaohaojie/workspace/repd/Matcha/psg4ci
tmux new-session -d -s matcha_h100_loso 'medex-run medex-run-h100-loso.yaml'
tmux attach -t matcha_h100_loso
```

全量最终 sequence（同样只上传/读取冻结 embedding）：

```bash
tmux new-session -d -s matcha_h100_final 'medex-run medex-run-h100-final.yaml'
```

查看状态/日志：

```bash
tmux ls
tail -f h100_loso_results/formal_loso.log
tail -f h100_final_results/formal_final_sequence.log
cat artifacts/v18_large6600/runtime_validation/rc
```

prepared cache 和永久冻结 embedding 均不应删除；H100 结果回传到 `h100_*_results/` 后可从对应阶段继续。

## 8. 结果文件索引

- E1：`h100_e1_results/e1_seed20260804_epoch15/`。
- frozen embeddings：`artifacts/v18_large6600/frozen/e1_seed20260804_epoch15/`。
- LOSO 原始输出：`h100_loso_results/`。
- LOSO residual/ensemble 汇总：`artifacts/v18_large6600/loso/`。
- final sequence：`h100_final_results/`。
- final deployment package：`artifacts/v18_large6600/final_model/raw_sequence_v18_full_training/`。
- official runtime validation：`artifacts/v18_large6600/runtime_validation/`。
- environment snapshot：`artifacts/v18_large6600/reports/environment.txt`。
