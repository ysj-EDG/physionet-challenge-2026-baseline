# FeatureLab Batch 1 Full Extraction Aggregate Report

This public report contains aggregate outcomes only. Private NPZ files, checkpoints, logs, authorization files, record identities, feature values, and per-record diagnostics remain outside Git.

## Final status

```text
FULL_6600_STATUS = COMPLETE
FINAL_ACCEPTANCE = TRUE
WORKERS = 8
EDF_ACCESS = NONE
TRAINING = NOT_STARTED
LOSO_SCORING = NOT_STARTED
```

## Frozen request

- Inventory: 6,600 records.
- Families: `eeg_tail`, `eeg_temporal`, `eeg_coherence_stage`, `caisr_stage`, `emg_stage`, `resp_stage`.
- Ordered feature count: 637 per record.
- Execution used the frozen family order, defaults, Recipe IDs, source contract, and `workers=8`.
- Existing family caches were validated and reused; no final cache was deleted or overwritten.

## Aggregate outcome

| family | dimensions | BUILT | SKIP_VERIFIED | total |
|---|---:|---:|---:|---:|
| eeg_tail | 288 | 5,126 | 1,474 | 6,600 |
| eeg_temporal | 96 | 5,128 | 1,472 | 6,600 |
| eeg_coherence_stage | 90 | 5,128 | 1,472 | 6,600 |
| caisr_stage | 35 | 5,128 | 1,472 | 6,600 |
| emg_stage | 72 | 5,130 | 1,470 | 6,600 |
| resp_stage | 56 | 5,130 | 1,470 | 6,600 |
| **all families** | **637** | **30,770** | **8,830** | **39,600** |

All 6,600 records were recorded as `COMPLETE`. Every record had six family outcomes.

## Safety and integrity checks

- Source immutability: 6,600/6,600 verified.
- Strict cache readback: 6,600/6,600 verified.
- Active safety stop: none.
- Final cache files: 39,600 stable NPZ files.
- Final stable cache bytes: 529,905,501.
- Peak RSS: 686,140 KiB.
- Wall time: 22,223.26488 seconds.
- Full command exit summary reported `final_acceptance=true` and `safety_stop=null`.

## Scope boundaries

The run read only existing NPZ-derived inputs. It did not read EDF files, modify the legacy 807-feature cache, modify five-view manifests, train models, run LOSO, or implement Batch 2 features. This aggregate report does not authorize downstream model or view migration.
