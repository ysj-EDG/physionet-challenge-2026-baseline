# G3-A-V Independent Verification Report

G3AV_STATUS = PASS

## Scope and method

This Gate independently reconstructed all six frozen G3-A Rank256 units from the validated G2-B1 night cache. The verifier did not call the existing G3-A ranking, quantization, view, audit, preparation, or validation functions as mathematical authority. It did not read source physiological NPZ/EDF files, age, labels, diagnoses, follow-up fields, or outcomes.

## Source verification

- G2-B1 cache files: 6600/6600 regular, non-symlink files.
- G2-B1 cache fingerprint: cf88b4e87f8b524258c685db696c278198dae1e6a10bd39fb63e893b2df031df.
- Inventory fingerprint: 1f914cdfe88857a5bc2b5a9a9773ab0c7e87fa9ca81354015bf7393d670732f3.
- Frozen Manifest SHA256: 7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a.
- Frozen Spec SHA256: f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a.
- Inventory and private index agreed at every ordered position on anonymous cache name, relative-source identity, and site; uniqueness and site closure were exact.
- G3-A run identity, data manifest, split manifest, and all six recorded source-code hashes matched the current frozen files.
- Source-valid closure: I0002=238,476; I0006=881,160; S0001=4,002,096; Overall=5,121,732.

## LOSO index verification

| Holdout | Train | Train N | Test N | Unused | Exact ordered match | Duplicates | Overlap |
|---|---|---:|---:|---|---|---:|---:|
| I0002 | S0001 | 5,139 | 319 | I0006 | PASS | 0 | 0 |
| I0006 | S0001 | 5,139 | 1,142 | I0002 | PASS | 0 | 0 |
| S0001 | I0006 | 1,142 | 5,139 | I0002 | PASS | 0 | 0 |

All saved train, test, and unused arrays matched independently rebuilt full arrays element by element; counts alone were not used as proof.

## Six-unit mathematical verification

| Holdout | Protocol | Train checked | Test checked | Train valid | Test valid | Ref match | Percentile mismatch | Level mismatch | View mismatch | Audit mismatch |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| I0002 | transductive | 4,147,173 | 257,433 | 4,002,096 | 238,476 | PASS | 0 | 0 | 0 | 0 |
| I0002 | inductive | 4,147,173 | 257,433 | 4,002,096 | 238,476 | PASS | 0 | 0 | 0 | 0 |
| I0006 | transductive | 4,147,173 | 921,594 | 4,002,096 | 881,160 | PASS | 0 | 0 | 0 | 0 |
| I0006 | inductive | 4,147,173 | 921,594 | 4,002,096 | 881,160 | PASS | 0 | 0 | 0 | 0 |
| S0001 | transductive | 921,594 | 4,147,173 | 881,160 | 4,002,096 | PASS | 0 | 0 | 0 | 0 |
| S0001 | inductive | 921,594 | 4,147,173 | 881,160 | 4,002,096 | PASS | 0 | 0 | 0 | 0 |

- Every saved reference array, offset, count, sorted value, and fingerprint exactly matched the independent reconstruction.
- Inductive test transformations used only the training-site reference; no holdout values entered reference fitting.
- Source-valid and rank-valid masks matched exactly. All valid percentiles matched bit-for-bit; all invalid outputs remained NaN.
- Transductive and Inductive training percentile, rank-mask, level, and quantized matrices were exactly equal within each fold.

## Rank256 verification

- Formula independently checked: k = floor(255*u + 0.5) and q = k/255.
- Integer levels and float64 grid values matched exactly at every position; mismatch, off-grid, and formula-error counts were all zero.
- Missing levels were exactly -1, missing float outputs were NaN, and no valid value was outside [0,1].
- Maximum observed quantization error: 0.0019607843137255387, within the frozen 1/510 plus tolerance bound.
- The u=0.5 boundary maps to 128/255; the injection test that distinguishes banker's rounding used a half-integer case at 255*u=76.5.

## Five-view verification

- Dimensions: V1=113, V2=129, V3=349, V4=113, V5=123; total positions=827.
- Every train/test view value, mask, row order, feature order, and NaN position matched the independent Manifest mapping.
- Repeated Demo5 positions mapped to identical unique-feature values and masks in all five views.

## Audit closure and focused checks

- G2-B1 public coverage rows: 3228; G3-A private/public rank-audit rows: 9684.
- All feature/family/fold/protocol/role keys and all integer and float audit statistics matched independent recomputation.
- I0002 holdout: 238,476 comparable test positions; Transductive versus Inductive mean absolute percentile difference 0.039681568334479854, maximum 0.577060676242447.
- I0002 test occupied-bin median: Transductive 256, Inductive 170.
- demo_bmi: S0001 train reference n=501; I0002 test valid n=91; occupied bins 90/69 (Transductive/Inductive); Inductive below/above range=3/5. Missingness was retained.
- caisr_sleep_mean_stage_entropy: S0001 reference n=6; I0002 source/rank valid n=0; the column remained present and entirely NaN for I0002.
- demo_male and each Race indicator were ranked independently with the universal tie formula; each had two occupied levels in the I0002 focus fold.
- Independently reproduced Inductive below/above counts: I0002=351/537, I0006=226/220, S0001=9,473/19,747.

## Integrity, recovery, and regression

- Every unit contained exactly seven payload files plus complete.json; hashes, schemas, dtypes, no-pickle loading, symlink absence, and absence of building directories were verified.
- Rehashed mutation tests for percentile, quantized, reference, views, and audit evidence were rejected by independent semantic checks.
- New G3-A-V tests: 30 passed. Historical main regression: 91 passed. G1-A direct regression: 15 test functions executed successfully. Python syntax checks passed.
- Full verification elapsed 356.28 s; peak RSS 935,804 KiB; swap 0. Smoke elapsed 257.26 s.
- Private verification evidence size: 6,487 bytes; completion marker and all file SHA256 values re-read successfully.
- Original G3-A artifacts: 52 files, 417,435,113 bytes; aggregate fingerprint 4a0f3956432030e4fa2fb8b51d15aa8fa5269f990d7795cca40d580970920a64 before final repository-only reporting.
- G2-B1 cache-file SHA aggregate: b1f79615d08fa64649683393837ca5e3acbc108004d70dea1f5d353bbb7fb447.

## Gate state

- G3AV_SOURCE_IDENTITY = PASS
- G3AV_LOSO_INDEX = PASS
- G3AV_REFERENCE = PASS
- G3AV_PERCENTILE = PASS
- G3AV_RANK256 = PASS
- G3AV_VIEWS = PASS
- G3AV_AUDIT = PASS
- G3A_ORIGINAL_ARTIFACTS = UNCHANGED
- G3B_MODEL_INPUT_ADAPTER = NOT STARTED
- G4_MODEL_TRAINING = NOT STARTED
- G5_RANK_ENSEMBLE = NOT STARTED

## Review boundary

No G3-A artifact was repaired, regenerated, or overwritten. No model adapter, training, scoring, or label connection was started. G3-A-V proves deterministic reconstruction and internal consistency only; it does not make a model-performance claim.
