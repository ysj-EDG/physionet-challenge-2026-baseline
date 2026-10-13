# G2-B1 Full 6600 Night-level Cache Report

`G2B1_STATUS = SUBMITTED_FOR_REVIEW`

## Environment and frozen inputs

- Branch: `research/siren-a-stagepool-v1`
- Parent Git SHA: `fa5e37bb09af428806a96a09f4f2e5fe69857c16`
- Cache schema: `g2b1_full_v1`
- Python: 3.9.18; NumPy: 1.26.4
- Input: 6,600 `timegrid_v2.1.0` / `validity_v0.2` NPZ files.
- Site closure: I0002=319, I0006=1,142, S0001=5,139.
- Input inventory fingerprint: `1f914cdfe88857a5bc2b5a9a9773ab0c7e87fa9ca81354015bf7393d670732f3`.
- Frozen Manifest SHA256: `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a`.
- Frozen Spec SHA256: `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a`.
- No symlink, duplicate record key, unexpected extension, or input/output overlap was found.

## Reference regression and build

- G2-A Golden: 18/18 exact values, NaN locations, masks, support counts, and all Views matched.
- Accepted G2-B0 Pilot: 2/2 exact values, NaN locations, masks, support counts, and all Views matched.
- Historical reference caches were hashed before and after replay and were unchanged.
- Initial build: computed=6,600, skipped=0, source SHA256 verified=6,600, complete=true.
- Build computation elapsed time: 1,685.38 s (28.09 min).
- Immutable checkpoints: 66, one for each completed block of 100 records.
- Completed Resume: computed=0, skipped=6,600, source SHA256 verified=6,600.
- A measured full Resume took 134.86 s wall time (129.70 s internal) with maximum RSS 72,076 KiB and no swap. Peak RSS for the initial feature-computing build was not separately instrumented; this is reported without inference.
- Private run-root disk use: 154 MiB; the 6,600 record caches contain 140,669,671 bytes, maximum single cache 22,141 bytes.

## Independent full verification

| Check | Result |
|---|---:|
| Cache files verified | 6,600 / 6,600 |
| Unique feature positions | 5,326,200 / 5,326,200 |
| Five-view positions | 5,458,200 / 5,458,200 |
| Unique features per record | 807 |
| View positions per record | 827 |
| V1 / V2 / V3 / V4 / V5 | 113 / 129 / 349 / 113 / 123 |
| Source SHA256 validations | 6,600 |
| Resume skips after completion | 6,600 |
| Invalid or overwritten caches | 0 |

All caches passed exact dtype, shape, finite/NaN, nonnegative support, family-specific support, provenance, and unique-to-View mapping checks. No record was removed, no missing value was filled, and no feature was standardized, clipped, ranked, or otherwise transformed.

## Full feature coverage

The public aggregate contains 3,228 rows: 807 unique feature IDs for each of I0002, I0006, S0001, and Overall. It contains no record identity, source path, patient-level value, age, or label.

| Family | Columns | Overall valid | Overall missing | Valid rate |
|---|---:|---:|---:|---:|
| EEG648 | 648 | 4,103,334 | 173,466 | 95.9440% |
| Spatial asymmetry | 24 | 151,144 | 7,256 | 95.4192% |
| Spatial AP gradient | 12 | 72,984 | 6,216 | 92.1515% |
| CAISR20 | 20 | 123,941 | 8,059 | 93.8947% |
| HRV22 | 22 | 144,496 | 704 | 99.5152% |
| EMG48 | 48 | 314,544 | 2,256 | 99.2879% |
| Resp28 | 28 | 183,906 | 894 | 99.5162% |
| Demo5 | 5 | 27,383 | 5,617 | 82.9788% |
| **All unique positions** | **807** | **5,121,732** | **204,468** | **96.1611%** |

### Site-level valid rates

| Site | EEG648 | Asymmetry24 | AP-gradient12 | CAISR20 | HRV22 | EMG48 | Resp28 | Demo5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 92.0324% | 87.4086% | 82.0272% | 94.4044% | 99.3730% | 99.3730% | 99.3730% | 82.3197% |
| I0006 | 95.0890% | 95.7677% | 90.5429% | 94.6060% | 99.9124% | 99.9416% | 99.7998% | 94.9912% |
| S0001 | 96.3768% | 95.8390% | 93.1374% | 93.7050% | 99.4357% | 99.1373% | 99.4621% | 80.3503% |
| Overall | 95.9440% | 95.4192% | 92.1515% | 93.8947% | 99.5152% | 99.2879% | 99.5162% | 82.9788% |

Per-feature/site valid and missing counts, support P05/P25/median/P75/P95, and lawful-zero counts and rates are in `g2b1_feature_coverage_aggregate.csv`. Missing values remain NaN with `valid_mask=false`; valid physiological zeros are retained.

## CAISR audit closure

- Annotation source missing: 70.
- Readable source with fewer than 20 aligned valid stages: 6.
- Stage source below 20 including missing sources: 76.
- All 70 missing-source records were retained; EEG648, Spatial36, and CAISR20 were invalid as frozen, while independent modalities followed their own validity metadata.
- All six readable short-stage records were retained and each stage-conditioned output used the frozen per-column minimum of five observations. No 20-epoch record exclusion was applied.
- No CAISR EDF was reopened and no expert annotation was used.

`caisr_sleep_mean_stage_entropy` was valid for 7/6,600 records: I0002=0/319, I0006=1/1,142, S0001=6/5,139. It was missing for 6,593 records. `ENTROPY_COVERAGE_ALERT = NOT_TRIGGERED_BY_ZERO_COVERAGE`, because coverage was not zero; the 0.1061% availability remains a prominent coverage warning for human review. The column remains in the frozen 807-feature contract.

## Safety, privacy, and validation

- Initial G2-B1 RED tests failed because the new modules did not exist; implementation then reached GREEN.
- G2-B1 contract and runner tests: 14/14 passed.
- Combined G1-B, G2-A, G2-B0, G2-B0-R, and G2-B1 unittest regression: 72/72 passed.
- G1-A direct regression checks: 15/15 passed.
- Python syntax compilation and `git diff --check`: passed.
- No label, diagnosis, follow-up field, model training, imputation, standardization, or site-rank operation is present in this workflow.
- Public CSV privacy scan found no patient path, source filename, record identity field, or patient identifier.
- Source NPZ files, G2-A Golden, G2-B0 Pilot, Frozen Manifest, and Frozen Spec were read-only and unchanged.

## Gate decision

No unexplained cache, schema, provenance, support, CAISR-closure, or privacy inconsistency remains. The extremely low entropy coverage and family/site missingness differences are reported evidence only and do not alter the frozen features.

- G2B1_FULL_CACHE: **GENERATED_AND_VERIFIED**
- G2A_GOLDEN_CACHE: **UNCHANGED**
- G2B0_PILOT_CACHE: **UNCHANGED**
- G1B_FROZEN_MANIFEST: **UNCHANGED**
- G3_PREPROCESSING: **NOT STARTED**
- SITE_RANK_NORMALIZATION: **NOT STARTED**
- MODEL_TRAINING: **NOT STARTED**

G2-B1 is submitted for human review. No subsequent Gate has been started.
