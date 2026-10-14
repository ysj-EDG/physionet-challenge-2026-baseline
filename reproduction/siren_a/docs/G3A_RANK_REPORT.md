# G3-A LOSO Rank256 Audit Report

`G3A_STATUS = SUBMITTED_FOR_REVIEW`

## Inputs and splits

- Parent SHA: `bddf55e15033b7872588f7a630be4c0a178c06a4`
- G2-B1 cache fingerprint: `cf88b4e87f8b524258c685db696c278198dae1e6a10bd39fb63e893b2df031df`
- Records/features/views: 6,600 / 807 / 827.
- Sites: I0002=319, I0006=1,142, S0001=5,139.

| Holdout | Train | Train N | Test N | Unused |
|---|---|---:|---:|---|
| I0002 | S0001 | 5139 | 319 | I0006 |
| I0006 | S0001 | 5139 | 1142 | I0002 |
| S0001 | I0006 | 1142 | 5139 | I0002 |

## Six preprocessing units

| Holdout | Protocol | Role | Source valid | Rank valid | Reference-unavailable | Median occupied bins | Max quantization error |
|---|---|---|---:|---:|---:|---:|---:|
| I0002 | transductive | train | 4002096 | 4002096 | 0 | 256.0 | 0.00196078 |
| I0002 | transductive | test | 238476 | 238476 | 0 | 256.0 | 0.00196078 |
| I0002 | inductive | train | 4002096 | 4002096 | 0 | 256.0 | 0.00196078 |
| I0002 | inductive | test | 238476 | 238476 | 0 | 170.0 | 0.00196078 |
| I0006 | transductive | train | 4002096 | 4002096 | 0 | 256.0 | 0.00196078 |
| I0006 | transductive | test | 881160 | 881160 | 0 | 256.0 | 0.00196078 |
| I0006 | inductive | train | 4002096 | 4002096 | 0 | 256.0 | 0.00196078 |
| I0006 | inductive | test | 881160 | 881160 | 0 | 246.0 | 0.00196078 |
| S0001 | transductive | train | 881160 | 881160 | 0 | 256.0 | 0.00196078 |
| S0001 | transductive | test | 4002096 | 4002096 | 0 | 256.0 | 0.00196078 |
| S0001 | inductive | train | 881160 | 881160 | 0 | 256.0 | 0.00196078 |
| S0001 | inductive | test | 4002096 | 4002096 | 0 | 255.0 | 0.00196078 |

All 12 role/protocol blocks contain exactly 807 feature rows. No block had a source-valid value lose rank validity because its reference was unavailable. The independent verify-only pass returned 6/6 valid units and zero errors; a subsequent build pass returned `SKIP_VERIFIED` for all six units.

## Rank256 aggregate audit

| Holdout | Protocol | Role | Occupied bins min/median/P95/max | Added tie groups | Collapsed rank levels | Weighted mean abs. error |
|---|---|---|---|---:|---:|---:|
| I0002 | transductive | test | 0 / 256 / 256 / 256 | 29,958 | 29,958 | 0.000981929 |
| I0002 | inductive | test | 0 / 170 / 183 / 198 | 56,549 | 80,666 | 0.000972350 |
| I0006 | transductive | test | 1 / 256 / 256 / 256 | 192,559 | 624,654 | 0.000984880 |
| I0006 | inductive | test | 1 / 246 / 253 / 255 | 157,949 | 469,666 | 0.000977177 |
| S0001 | transductive | test | 2 / 256 / 256 / 256 | 193,059 | 3,502,664 | 0.000978088 |
| S0001 | inductive | test | 1 / 255 / 256 / 256 | 184,944 | 463,469 | 0.000978419 |

The zero occupied-bin minimum for I0002 is the entirely missing stage-entropy column. All valid quantized values were on the 256-level grid, remained monotone, and had maximum absolute error `0.001960784313725539`, within `1/510` plus floating tolerance. Added ties and collapsed levels are expected deterministic quantization loss; no noise or interpolation was used.

## Protocol comparison

The Transductive and Inductive training percentile matrices and rank masks are exactly equal in all three folds. The holdout comparison is:

| Holdout | Comparable positions | Different positions | Mean abs. difference | Median | P95 | Maximum | Below/above training range |
|---|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 238,476 | 238,475 | 0.039682 | 0.028381 | 0.113732 | 0.577061 | 351 / 537 |
| I0006 | 881,160 | 881,157 | 0.089602 | 0.056843 | 0.299427 | 0.622599 | 226 / 220 |
| S0001 | 4,002,096 | 4,002,095 | 0.089320 | 0.056632 | 0.299777 | 0.916667 | 9,473 / 19,747 |

These differences arise from the specified reference cohorts. Inductive references contain only the selected training site; Transductive holdout ranks contain only the complete unlabeled holdout cohort.

## S0001 to I0002 focused audit

- Train reference cohort: S0001=5,139; Transductive target reference cohort: I0002=319.
- Transductive and Inductive training matrices are exactly equal; the test protocols differ only by reference cohort.
- BMI has 501 valid S0001 references and 91 valid I0002 values. I0002 occupies 90 bins Transductively versus 69 Inductively; 3 values are below and 5 above the S0001 reference range. Source missingness is retained and no value is filled.
- Stage entropy has 6 valid S0001 references and 0 valid I0002 values. I0002 remains entirely NaN under both protocols and the column is retained.
- `demo_male` has 2 raw percentile levels and 2 quantized levels in both cohorts. Each race indicator is ranked independently, also retains 2 levels, and preserves the 301/319 valid-mask pattern in I0002.
- Across all I0002 test positions, protocol mean absolute difference is 0.039682. The largest feature-level inductive overflow proportion is BMI at 8/91 (8.79%); remaining leading overflow columns are EEG spectral features and are reported without changing the transform.
- All 807 columns were audited; no feature-specific ranking rule was introduced.

The other BMI reference/target counts were 501/946 for holdout I0006 and 946/501 for holdout S0001. Their Inductive occupied bins were 209 and 185, respectively. Entropy references were 6 for holdouts I0002/I0006 and 1 for holdout S0001; valid target counts were 0/1/6. Consequently, no fold had an unavailable training reference, although entropy coverage remains extremely sparse. The six EEG stage groups were programmatically audited across all channels/statistics; representative Wake, REM, N2, N3, NREM and All-stage columns all obeyed the same range, monotonicity and Rank256 checks.

## Validation

- Percentiles are finite on rank-valid positions, within [0,1], tie-consistent, monotone and order-independent.
- Rank256 levels are integers 0-255, missing sentinel is -1, and float outputs lie on k/255.
- Quantization error never exceeds 1/510 plus numerical tolerance.
- Source-valid and rank-valid masks remain separate; reference-unavailable outputs remain NaN.
- Frozen five-view mappings and repeated Demo5 values/masks were verified for every unit.
- G2-B1 coverage counts matched all three sites and all 807 columns.
- Summed unit construction elapsed: 800.12 seconds. The full build invocation wall time was 836.48 seconds, with peak resident memory 464,896 KiB and no swap. The earlier I0002 smoke invocation took 365.92 seconds with peak resident memory 434,200 KiB.
- Private completed output occupies 399 MiB. A failed pre-publication smoke attempt was preserved separately as immutable diagnostic evidence; it is not treated as a completed unit.

## Interpretation boundary

Occupied-bin counts and additional ties reflect deterministic quantization, not model performance. BMI site missingness and stage-entropy scarcity remain visible. No feature, sample, protocol, or formula was changed based on these findings.

- G3A_LOSO_SPLITS: **FROZEN**
- G3A_TRANSDUCTIVE: **VERIFIED**
- G3A_INDUCTIVE: **VERIFIED**
- G3A_256_LEVEL_GRID: **VERIFIED**
- G3A_FEATURE_AUDIT: **COMPLETE**
- G2B1_FULL_CACHE: **UNCHANGED**
- G3B_MODEL_INPUT_ADAPTER: **NOT STARTED**
- G4_MODEL_TRAINING: **NOT STARTED**
- G5_RANK_ENSEMBLE: **NOT STARTED**
