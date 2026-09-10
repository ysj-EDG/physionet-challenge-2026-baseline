# P3 v2 时间对齐说明

本轮只读取冻结的 `npz_new` 生理特征，并用官方原始 CAISR 标注和 EDF 头为
1103 条固定记录建立独立 stage sidecar；没有重提取或覆盖 NPZ。

- 1090 条记录可按官方生理 EDF 起点后的连续 30 秒物理前缀可靠对齐。
- 13 条记录缺少可读取的 CAISR 标注；患者仍保留，五个阶段存在标记置 0，
  stage/global EEG 汇聚值按缺失处理。
- 历史缓存的阶段 one-hot 对 1090 条可读记录匹配“删除无效阶段后的压缩序列”，
  并不等同于同一行 EEG 的物理时间；P3 未使用这些压缩列作 EEG 分期索引。
- 历史缓存裁掉的生理尾部没有恢复。global 与 stage 汇聚均只使用相同的、
  原缓存 `X_seq` 长度范围内且 CAISR 阶段有效的 epoch。
- 缺导联仅依据 EDF 头中可追溯的导联存在性排除；未把全零特征猜为缺导联，
  也未把 ECG mask 当 EEG 质量 mask。旧缓存缺少逐 epoch EEG 质量证据，
  因而本轮保留已有数值并把这点作为解释限制。

详细逐记录状态见 `stage_sidecar/manifest.csv`，计数与输入哈希见
`stage_sidecar/summary.json`。data2 的 `timegrid_v2` 缓存任务与本轮 733 人 CV
独立运行，其数据和拟合参数未混入 P3。
