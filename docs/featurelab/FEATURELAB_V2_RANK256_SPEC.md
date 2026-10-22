# FeatureLab v2 Rank256 specification

## Scope

This extension ranks the immutable `candidate_pool_v2_fix1` 1,444-column matrix for three fixed LOSO holdouts under two protocols. It does not re-extract features, rebuild the candidate pool, impute, scale, select features, train models, score records, or modify Branch A.

The six private units are `I0002`, `I0006`, and `S0001` crossed with `transductive` and `inductive`. The train site mapping is I0002→S0001, I0006→S0001, and S0001→I0006; rows retain source order and the third site is recorded as unused.

## Frozen numerical contract

For each column, the reference is the stable sorted finite values selected by the source-valid mask. For a valid value `x`, with `L` reference values strictly below `x`, `E` reference values equal to `x`, and `n` reference values, the percentile is `(L + 0.5 E) / n`. Empty references produce NaN and a false rank-valid mask. Quantization is `floor(255*u + 0.5)` with integer levels 0–255 and missing level -1; quantized missing values remain NaN. Source validity and rank validity are stored separately.

The implementation reuses the frozen G3-A normalization functions and keeps NaN/mask semantics explicit. No patient labels, outcomes, predictions, or scores are read.

## Input and output identity

The input is the byte-verified `candidate_pool_v2_fix1` completion, including ordered Feature IDs, row/site identity, source-valid mask, support counts, reason codes, and explicit v2 view rows. The output root is private: `feature-cache-extension/rank256_v2`. Each unit contains `arrays.npz`, `metadata.json`, and `complete.json`; publication is atomic and no-clobber. Existing complete units are reused only when metadata and file digests validate exactly.

The six approved Batch1 family Recipe IDs remain those frozen in `FEATURELAB_BATCH1_FULL_EXTRACTION_PLAN.md`; Rank256 does not alter them. For audit convenience they are: `eeg_tail=17f79dd79b79c4cc26ab0477b6c936e28cfb31161e9c0bb86ff44a21fa496054`, `eeg_temporal=5dc03f81224b71d166ec736f0517575e982debecab46f06c85ae2c0a1446f8bd`, `eeg_coherence_stage=c2c6949634746176d99c204792e5f860384530fee1a35ce54d5f4e41eb676a39`, `caisr_stage=6529a32282e29ec44ef5af0290379c2d11f3d70527504a513b5121e6069ed7f3`, `emg_stage=29cd008bf82a47a6fa823c400d8bca4b5e71f6a7c68a8b7a7eb7682da74ec91b`, and `resp_stage=3c1c051e7091cfeac73160a112f048208156cbdac1b5f4e33e24e4ce1825dd35`. The Rank256 metadata separately records the pool digest, ordered Feature-ID digest, formula, level count, rounding, protocol, fold counts, and view-manifest digest.

## Experiment column selections

- E0: columns `[0:807]`, 807 old independent features.
- E1: columns `[0:1191]`, old807 plus the first 384 new columns.
- E2: columns `[0:807] + [1191:1444]`, old807 plus the final 253 new columns.
- E3: columns `[0:1444]`, all candidate features.

Selections are explicit ordered index/Feature-ID lists. They are metadata only in this phase; no five-view training matrix is generated. View mapping is fixed as follows: E0 uses all old V1–V5; E1 uses v2 V1–V4 with the old V5; E2 uses old V1–V4 with v2 V5; E3 uses all v2 V1–V5.

## Compatibility and privacy

For every unit, the first 807 freshly ranked columns, source/rank masks, quantized levels, reference counts, and old view prefixes are compared exactly with the immutable G3-A unit. Public QC contains only site/protocol/role/feature aggregates; private arrays, support detail, row keys, and diagnostic values remain outside Git.
