# FeatureLab v2 候选池质控报告

状态：CACHE_ONLY_QC = COMPLETE；本轮仅修复原因编码、视图契约并从既有家族缓存发布独立修正版；未重提取、未执行训练、Rank256、特征选择、LOSO或Branch A迁移。

## 聚合结果

- 行数：6,600；独立特征：1,444；视图位置：1,464。
- 修正版保留原 `X_raw`、`valid_mask`、`support_count`、行顺序和 Feature ID 顺序；仅重建 missing-reason codebook。
- 公开逐列文件 `audit/featurelab_v2_feature_qc.csv` 含 Overall、I0002、I0006、S0001 四个 scope，共 5,776 行；支持量和数值分布仅保存在私有 QC。

## 分站点逐列 QC（聚合）

| scope | rows | valid cells | missing cells | all-missing columns | constant-valid columns | STATIC_BOUT_SEMANTIC_MISMATCH | caisr_sleep_mean_stage_entropy valid/total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Overall | 6600 | 9109798 | 420602 | 0 | 1 | 0 | 7/6600 |
| I0002 | 319 | 423903 | 36733 | 1 | 2 | 0 | 0/319 |
| I0006 | 1142 | 1563164 | 85884 | 0 | 2 | 0 | 1/1142 |
| S0001 | 5139 | 7122731 | 297985 | 0 | 1 | 0 | 6/5139 |

### 质量审核标记

- **Overall** 全缺失列：0；常数列（与全缺失分开）：1。
  - 主要缺失原因计数：{"ANNOTATION_UNAVAILABLE": 700, "CHANNEL_UNAVAILABLE": 12512, "INSUFFICIENT_DRIFT_SIDE_SUPPORT": 5868, "INSUFFICIENT_LEGAL_PAIRS": 2468, "INSUFFICIENT_REFERENCE_SUPPORT": 2468, "INSUFFICIENT_SUPPORT": 176800, "NO_LEGAL_TRANSITIONS": 4810, "NO_QUALIFYING_HOUR": 4756, "NO_STAGE_BOUT_SUPPORT": 1222, "OLD_SOURCE_MISSING": 204468, "PAIR_UNAVAILABLE": 4530}
  - 全缺失 Feature ID：none
- **I0002** 全缺失列：1；常数列（与全缺失分开）：2。
  - 主要缺失原因计数：{"ANNOTATION_UNAVAILABLE": 20, "CHANNEL_UNAVAILABLE": 5824, "INSUFFICIENT_DRIFT_SIDE_SUPPORT": 400, "INSUFFICIENT_LEGAL_PAIRS": 48, "INSUFFICIENT_REFERENCE_SUPPORT": 48, "INSUFFICIENT_SUPPORT": 8208, "NO_LEGAL_TRANSITIONS": 210, "NO_QUALIFYING_HOUR": 224, "NO_STAGE_BOUT_SUPPORT": 64, "OLD_SOURCE_MISSING": 18957, "PAIR_UNAVAILABLE": 2730}
  - 全缺失 Feature ID：caisr_sleep_mean_stage_entropy
- **I0006** 全缺失列：0；常数列（与全缺失分开）：2。
  - 主要缺失原因计数：{"ANNOTATION_UNAVAILABLE": 40, "CHANNEL_UNAVAILABLE": 48, "INSUFFICIENT_DRIFT_SIDE_SUPPORT": 972, "INSUFFICIENT_LEGAL_PAIRS": 268, "INSUFFICIENT_REFERENCE_SUPPORT": 268, "INSUFFICIENT_SUPPORT": 41724, "NO_LEGAL_TRANSITIONS": 1110, "NO_QUALIFYING_HOUR": 616, "NO_STAGE_BOUT_SUPPORT": 404, "OLD_SOURCE_MISSING": 40434}
  - 全缺失 Feature ID：none
- **S0001** 全缺失列：0；常数列（与全缺失分开）：1。
  - 主要缺失原因计数：{"ANNOTATION_UNAVAILABLE": 640, "CHANNEL_UNAVAILABLE": 6640, "INSUFFICIENT_DRIFT_SIDE_SUPPORT": 4496, "INSUFFICIENT_LEGAL_PAIRS": 2152, "INSUFFICIENT_REFERENCE_SUPPORT": 2152, "INSUFFICIENT_SUPPORT": 126868, "NO_LEGAL_TRANSITIONS": 3490, "NO_QUALIFYING_HOUR": 3916, "NO_STAGE_BOUT_SUPPORT": 754, "OLD_SOURCE_MISSING": 145077, "PAIR_UNAVAILABLE": 1800}
  - 全缺失 Feature ID：none

`STATIC_BOUT_SEMANTIC_MISMATCH` 在本次修正版原因字典中未作为有效原因出现时报告为 0；若未来出现，保留为审核标记，不删除列、不降支持阈值。无有效值的比例和分布在私有 QC 中记录为不可计算，而不是 0。

## 视图与候选集合

| V1 | V2 | V3 | V4 | V5 |
|---:|---:|---:|---:|---:|
| 161 | 177 | 493 | 257 | 376 |

E0/E1/E2/E3 列选择分别为 `[0:807]`、`[0:1191]`、`[0:807] + [1191:1444]`、`[0:1444]`；仅定义输入列，不生成训练矩阵。
