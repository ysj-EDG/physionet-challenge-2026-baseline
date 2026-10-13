# G2-A Independent Night-level Smoke Report

`G2A_STATUS = SUBMITTED_FOR_REVIEW`

## Environment

- Branch: `research/siren-a-stagepool-v1`
- Parent SHA: `9b99a0de7724093c30ca3135b3ef5bfae9bcce2d`
- Python: `3.9.18`; NumPy: `1.26.4`
- Input: `timegrid_v2.1.0` / `validity_v0.2`
- Manifest SHA256: `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a`
- Spec SHA256: `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a`
- Smoke execution script SHA256 recorded in cache: `69117c9d6a3626fd4adc6490ef9a536f720091e51f66644177798596b5bcc280`
- Current reviewed report module SHA256: `ad7e0cdee27107b7808a49f5521b735948ac75626f51abbd12bc77a399e77690`
- A post-run aggregate-report-only correction explains any difference between those script fingerprints; feature arrays and private caches were not rewritten.
- Git scope: eight approved G2-A files only.

## Sampling

- Smoke records: 18; I0002=6, I0006=6, S0001=6; fixed=9, seeded random=9, boundary additions=0.
- Output cap 30 respected; NPZ contents read: 18 distinct records.
- Real boundary coverage: BMI missing, EEG lead unavailable, N3<5, probability value 9, spatial intersection<5.
- NOT COVERED BY REAL SMOKE: Race unavailable, annotation missing, no HRV windows, probability incomplete.

## Output

- Private caches: 18; unique=807; V1/V2/V3/V4/V5=113/129/349/113/123; total view positions=827.
- Atomic write/read, source SHA preservation, and deterministic replay: PASS.

## Feature QC

| site | family | columns | valid | missing | valid ratio | fully invalid columns | zero values | support min/median/max |
|---|---|---:|---:|---:|---:|---:|---:|---|
| I0002 | CAISR20 | 20 | 114 | 6 | 0.9500 | 1 | 1 | 0/1.0/1 |
| I0002 | Demo5 | 5 | 25 | 5 | 0.8333 | 0 | 15 | 0/1.0/1 |
| I0002 | EEG54_stage_spectral | 648 | 3564 | 324 | 0.9167 | 0 | 3 | 0/302.5/1091 |
| I0002 | EMG24 | 48 | 288 | 0 | 1.0000 | 0 | 18 | 895/956.0/1097 |
| I0002 | HRV11 | 22 | 132 | 0 | 1.0000 | 0 | 0 | 886/947.0/1088 |
| I0002 | Resp14 | 28 | 168 | 0 | 1.0000 | 0 | 10 | 889/956.0/1097 |
| I0002 | spatial_ap_gradient_candidate | 12 | 56 | 16 | 0.7778 | 0 | 0 | 0/116.5/599 |
| I0002 | spatial_asymmetry_candidate | 24 | 128 | 16 | 0.8889 | 0 | 0 | 0/330.0/599 |
| I0006 | CAISR20 | 20 | 114 | 6 | 0.9500 | 1 | 9 | 0/1.0/1 |
| I0006 | Demo5 | 5 | 30 | 0 | 1.0000 | 0 | 15 | 1/1.0/1 |
| I0006 | EEG54_stage_spectral | 648 | 3456 | 432 | 0.8889 | 0 | 3 | 0/214.0/845 |
| I0006 | EMG24 | 48 | 288 | 0 | 1.0000 | 0 | 29 | 748/788.0/852 |
| I0006 | HRV11 | 22 | 132 | 0 | 1.0000 | 0 | 0 | 415/766.5/843 |
| I0006 | Resp14 | 28 | 168 | 0 | 1.0000 | 0 | 9 | 414/780.0/852 |
| I0006 | spatial_ap_gradient_candidate | 12 | 56 | 16 | 0.7778 | 0 | 0 | 0/101.5/543 |
| I0006 | spatial_asymmetry_candidate | 24 | 120 | 24 | 0.8333 | 0 | 0 | 0/163.0/543 |
| S0001 | CAISR20 | 20 | 114 | 6 | 0.9500 | 1 | 1 | 0/1.0/1 |
| S0001 | Demo5 | 5 | 24 | 6 | 0.8000 | 1 | 15 | 0/1.0/1 |
| S0001 | EEG54_stage_spectral | 648 | 3672 | 216 | 0.9444 | 0 | 2 | 0/314.0/1019 |
| S0001 | EMG24 | 48 | 288 | 0 | 1.0000 | 0 | 17 | 833/951.0/1037 |
| S0001 | HRV11 | 22 | 132 | 0 | 1.0000 | 0 | 0 | 824/942.0/1028 |
| S0001 | Resp14 | 28 | 168 | 0 | 1.0000 | 0 | 5 | 828/951.0/1037 |
| S0001 | spatial_ap_gradient_candidate | 12 | 64 | 8 | 0.8889 | 0 | 0 | 0/123.0/521 |
| S0001 | spatial_asymmetry_candidate | 24 | 144 | 0 | 1.0000 | 0 | 0 | 17/218.5/521 |

## Entropy QC

| site | records | value=9 | N1/N2/N3/REM/W value=9 | out of range | nonfinite | finite rows | in-range rows | zero-sum rows | 9@stage_valid | 9@stage_invalid | length mismatch | entropy valid | A/B/C |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| I0002 | 6 | 180 | 36/36/36/36/36 | 180 | 0 | 5833 | 5797 | 0 | 0 | 180 | 0 | 0 | 6/0/0 |
| I0006 | 6 | 180 | 36/36/36/36/36 | 208 | 0 | 4753 | 4689 | 0 | 0 | 180 | 0 | 0 | 6/0/0 |
| Overall | 18 | 540 | 108/108/108/108/108 | 568 | 0 | 16261 | 16125 | 0 | 0 | 540 | 0 | 0 | 18/0/0 |
| S0001 | 6 | 180 | 36/36/36/36/36 | 180 | 0 | 5675 | 5639 | 0 | 0 | 180 | 0 | 0 | 6/0/0 |

- Entropy coverage alert: `ENTROPY_COVERAGE_ALERT`.
- Values equal to 9 are never altered or used to recompute entropy; they invalidate only the stored entropy feature under the frozen rule.

## Validation

- `python test_g2a_features.py`: 8 tests passed.
- `python test_g2a_smoke.py`: 6 tests passed.
- Existing G1-A and G1-B tests: passed.
- Python syntax compilation: passed.
- Frozen Manifest row-by-row mapping and source indices: passed.
- No label read, record deletion, imputation, scaling, site mapping, or training path exists.

## Findings

- PASS: schema, 807/827 assembly, cache round-trip, source preservation, and deterministic replay.
- EXPECTED MISSINGNESS: per-column support below five and metadata-invalid static values remain NaN/false.
- COVERAGE WARNING: `ENTROPY_COVERAGE_ALERT`.
- BLOCKING SCHEMA OR CALCULATION ERROR: none.

No patient ID, filename, path, age, label, or patient-level vector is present in this report.

- G2A_SMOKE_CACHE: **GENERATED**
- G2B_FULL_CACHE: **NOT GENERATED**
- G3_PREPROCESSING: **NOT STARTED**
- MODEL_TRAINING: **NOT STARTED**
- SITE_RANK_NORMALIZATION: **NOT STARTED**
