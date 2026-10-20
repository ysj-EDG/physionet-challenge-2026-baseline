# FeatureLab Batch 1 Cross-Site Pilot Report

This is a public aggregate report. It contains no patient identity, per-patient feature value, per-patient threshold, private mapping, or private file location. Detailed diagnostics remain under the private FeatureLab cache root and are not committed.

## Status

```text
CROSS_SITE_PILOT = COMPLETE
SOURCE_IMMUTABILITY = VERIFIED
CACHE_READBACK = VERIFIED
CHECKPOINT_COMPATIBILITY = VERIFIED_WITH_ZERO_EXECUTION_PROBE
BATCH1_637_RECIPE_STATUS = FROZEN
BATCH2_553_STATUS = AUDIT_ONLY
FULL_6600_STATUS = NOT_STARTED
MODEL_TRAINING = NOT_STARTED
LOSO_SCORING = NOT_STARTED
BRANCH_A = UNCHANGED
```

No EDF was read, no old extractor was invoked, no training matrix was created, and no formula, feature order, validity rule, or support threshold was changed.

## Deterministic selection

The path-only inventory contained 6,600 NPZ files across three sites. Building this inventory did not decode NPZ arrays. Candidate inspection then used the frozen source adapter and only schema, stage coverage, and modality availability; forbidden label, TTE, prediction, and score fields were neither requested nor exposed.

| measure | aggregate |
|---|---:|
| inventory files | 6,600 |
| I0002 inventory files | 319 |
| I0006 inventory files | 1,142 |
| S0001 inventory files | 5,139 |
| candidate-open hard limit | 80 |
| candidates opened | 80 |
| eligible candidates | 69 |
| rejected for support or modality coverage | 11 |
| selected files | 6 |
| selected per site | 2 |

The previously validated single-file Smoke cache was included by the deterministic preference rule and reused. The six selected sources received exactly one six-family batch request with `workers=1`.

## Per-site execution aggregates

| site | processed | COMPLETE | PARTIAL | BUILT families | SKIP_VERIFIED families | other family statuses |
|---|---:|---:|---:|---:|---:|---:|
| I0002 | 2 | 2 | 0 | 6 | 6 | 0 |
| I0006 | 2 | 2 | 0 | 12 | 0 | 0 |
| S0001 | 2 | 2 | 0 | 12 | 0 | 0 |
| **total** | **6** | **6** | **0** | **30** | **6** | **0** |

There were no `REQUIRES_INPUT_REVIEW`, `FAILED`, `STOP_INVALID`, or `STOP_RECORD` results. Legal per-feature missingness remained missingness and did not turn a record into `PARTIAL`.

## Six-family aggregates

Valid and missing distributions are per selected file. Totals cover all six selected files.

| family | dimension | BUILT / SKIP | valid min / median / max | missing min / median / max | valid total | missing total | aggregate missing reason | cache bytes | Recipe ID |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| eeg_tail | 288 | 5 / 1 | 192 / 288 / 288 | 0 / 0 / 96 | 1,632 | 96 | `CHANNEL_UNAVAILABLE` = 96 | 109,097 | `17f79dd79b79c4cc26ab0477b6c936e28cfb31161e9c0bb86ff44a21fa496054` |
| eeg_temporal | 96 | 5 / 1 | 64 / 96 / 96 | 0 / 0 / 32 | 544 | 32 | `CHANNEL_UNAVAILABLE` = 32 | 83,670 | `5dc03f81224b71d166ec736f0517575e982debecab46f06c85ae2c0a1446f8bd` |
| eeg_coherence_stage | 90 | 5 / 1 | 30 / 90 / 90 | 0 / 0 / 60 | 480 | 60 | `PAIR_UNAVAILABLE` = 60 | 75,690 | `c2c6949634746176d99c204792e5f860384530fee1a35ce54d5f4e41eb676a39` |
| caisr_stage | 35 | 5 / 1 | 35 / 35 / 35 | 0 / 0 / 0 | 210 | 0 | none | 67,087 | `6529a32282e29ec44ef5af0290379c2d11f3d70527504a513b5121e6069ed7f3` |
| emg_stage | 72 | 5 / 1 | 72 / 72 / 72 | 0 / 0 / 0 | 432 | 0 | none | 73,983 | `29cd008bf82a47a6fa823c400d8bca4b5e71f6a7c68a8b7a7eb7682da74ec91b` |
| resp_stage | 56 | 5 / 1 | 56 / 56 / 56 | 0 / 0 / 0 | 336 | 0 | none | 72,243 | `3c1c051e7091cfeac73160a112f048208156cbdac1b5f4e33e24e4ce1825dd35` |
| **total** | **637** | **30 / 6** | — | — | **3,634** | **188** | — | **481,770** | **six independent Recipes** |

All 36 successful family files passed Cache Schema, payload digest, Recipe, ordered Feature ID, source identity, validity, support, missing-reason, and strict readback checks. The CAISR aggregate contained zero `STATIC_BOUT_SEMANTIC_MISMATCH` results; its approved tolerance was not modified.

## Cache recovery and source protection

- The private checkpoint contains six completed entries, each with one attempt.
- A zero-execution compatibility probe reopened the same checkpoint and request contract without invoking `run_record()`.
- Synthetic interruption/resume tests prove that completed family caches become `SKIP_VERIFIED` and only absent family results build.
- The existing Smoke contribution returned six `SKIP_VERIFIED` decisions; its extractors were not called.
- SHA256 and stat snapshots compared equal before and after every Pilot request.
- No existing final cache was overwritten or deleted.

## Resource observations

| metric | observed aggregate |
|---|---:|
| one-worker batch elapsed | 13.612 s |
| per-file elapsed min / median / max | 1.003 / 2.476 / 2.713 s |
| newly built file elapsed min / median / max | 2.234 / 2.492 / 2.713 s |
| process peak RSS | 356,176 KiB (about 348 MiB) |
| newly published cache bytes | 402,279 |
| selected final cache bytes | 481,770 |
| new per-file cache bytes min / median / max | 80,354 / 80,467 / 80,561 |

The timing estimate is based on only five new builds plus one all-cache reuse. It is adequate for a conservative planning range, not a precise full-cohort benchmark.

## Risk conclusion

- No cross-site source-contract or family-input incompatibility appeared in the selected three-site Pilot.
- EEG and Coherence missingness was explained by frozen channel/pair availability rules, not by software failure or relaxed support thresholds.
- No low-support missing-reason category occurred in this selected set; this does not prove it will be absent in the full inventory.
- CAISR static/grid bout semantics agreed for all selected files.
- The safe recommendation remains one real worker for the first authorized full run. Real multi-worker resource scaling has not been measured; only synthetic publication competition is verified.
