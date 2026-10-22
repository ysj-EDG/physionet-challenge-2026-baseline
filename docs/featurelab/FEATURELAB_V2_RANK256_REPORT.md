# FeatureLab v2 Rank256 build report

This report records the authorized six-unit Rank256 build from the immutable `candidate_pool_v2_fix1`. It contains aggregate QC only; private arrays and row-level identities remain outside Git.

## Status

- `V2_RANK256_STATUS = COMPLETE` (six private units published and read back).
- `INPUT_POOL_VERSION = candidate_pool_v2_fix1`.
- `INPUT_POOL_IMMUTABILITY = PASS` (completion file summaries unchanged).
- `FOLD_PROTOCOL_UNITS = 6/6`.
- `OLD807_PERCENTILE_EXACT_MATCH = PASS`.
- `OLD807_LEVELS_EXACT_MATCH = PASS`.
- `OLD807_QUANTIZED_EXACT_MATCH = PASS`.
- `OLD807_SOURCE_RANK_MASK_MATCH = PASS`.
- `OLD_VIEW_EXACT_MATCH = PASS`.
- `TRAIN_PROTOCOL_EQUALITY = PASS` for all three holdouts; the shared S0001 train arrays for I0002 and I0006 are also exact.
- `EMPTY_SINGLETON_REFERENCE_TEST = PASS` in the 12-case synthetic suite; real empty-reference columns remain explicitly masked.
- `E0_E1_E2_E3_MAPPING_STATUS = PASS` (807/1191/1060/1444 ordered selections).
- `QUANTIZATION_QC_FLAGS = NONE` (half-up levels, -1 missing sentinel, finite q only on rank-valid positions; aggregate boundary/collision counts are in the CSV).
- `ARTIFACT_READBACK_STATUS = PASS` for all six units.
- `IMPUTATION = NOT_PERFORMED`; `FEATURE_SELECTION = NOT_PERFORMED`; `MODEL_TRAINING = NOT_STARTED`; `PREDICTION_AND_SCORING = NOT_STARTED`.
- `BATCH2 = AUDIT_ONLY`.
- Frozen Batch1 family Recipe IDs are unchanged: `eeg_tail=17f79dd79b79c4cc26ab0477b6c936e28cfb31161e9c0bb86ff44a21fa496054`, `eeg_temporal=5dc03f81224b71d166ec736f0517575e982debecab46f06c85ae2c0a1446f8bd`, `eeg_coherence_stage=c2c6949634746176d99c204792e5f860384530fee1a35ce54d5f4e41eb676a39`, `caisr_stage=6529a32282e29ec44ef5af0290379c2d11f3d70527504a513b5121e6069ed7f3`, `emg_stage=29cd008bf82a47a6fa823c400d8bca4b5e71f6a7c68a8b7a7eb7682da74ec91b`, `resp_stage=3c1c051e7091cfeac73160a112f048208cbdac1b5f4e33e24e4ce1825dd35`.

## Unit summary

| holdout | protocol | dimensions | train rows | test rows | source-valid train | rank-valid train | source-valid test | rank-valid test | cache bytes | readback | old807 regression |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| I0002 | transductive | 1444 | 5139 | 319 | 7122731 | 7122731 | 423903 | 423903 | 123496682 | PASS | PASS |
| I0002 | inductive | 1444 | 5139 | 319 | 7122731 | 7122731 | 423903 | 423903 | 154741382 | PASS | PASS |
| I0006 | transductive | 1444 | 5139 | 1142 | 7122731 | 7122731 | 1563164 | 1563164 | 141319372 | PASS | PASS |
| I0006 | inductive | 1444 | 5139 | 1142 | 7122731 | 7122731 | 1563164 | 1563164 | 168316673 | PASS | PASS |
| S0001 | transductive | 1444 | 1142 | 5139 | 1563164 | 1563164 | 7122731 | 7122731 | 141319372 | PASS | PASS |
| S0001 | inductive | 1444 | 1142 | 5139 | 1563164 | 1563164 | 7122731 | 7122731 | 102234250 | PASS | PASS |

## New-family aggregate QC

| source block | feature columns | total source-valid | total rank-valid |
|---|---:|---:|---:|
| eeg_tail | 288 | 9767216 | 9767216 |
| eeg_temporal | 96 | 3364552 | 3364552 |
| eeg_coherence_stage | 90 | 3002960 | 3002960 |
| caisr_stage | 35 | 1225292 | 1225292 |
| emg_stage | 72 | 2515248 | 2515248 |
| resp_stage | 56 | 1947412 | 1947412 |

## Selection and view contract

- E0/E1/E2/E3 counts are 807/1191/1060/1444 and are recorded as ordered Feature-ID lists in each private unit metadata and the public selection manifest.
- v2 view dimensions are V1/V2/V3/V4/V5 = 161/177/493/257/376; no training input matrix was produced. E0 uses old V1–V5, E1 uses v2 V1–V4 plus old V5, E2 uses old V1–V4 plus v2 V5, and E3 uses all v2 views.
- The public QC CSV contains 17,328 aggregate feature rows (6 units × 2 roles × 1,444 features). It has no patient identifiers, patient values, or per-patient support detail.

## Caveats and next gate

These are mathematical and artifact-level results on the frozen candidate pool, not model results and not evidence of clinical improvement. A downstream reviewer must separately approve any imputation, Rank256 feature selection, model training, or scoring.
