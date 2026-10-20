# FeatureLab Batch 1 Aggregate Report

Public aggregate report for the six-family 637-column Batch 1 deliverable. This document contains aggregates and explicitly labeled synthetic numerical examples only. All real record values, thresholds, per-column supports, record identity, and the record mapping live exclusively in the private report under `/database/home/gaohaojie/workspace/feature-cache-extension/reports/` and are never committed or copied here.

## Status

```text
BATCH1_SMOKE = COMPLETE
SOURCE_IMMUTABILITY = VERIFIED
CACHE_RESUME = VERIFIED (6/6 SKIP_VERIFIED, 0 extractor calls)
EDF_ACCESS = NONE
REAL_RECORDS = 1 (single bounded smoke; no full extraction)
```

## Real one-record smoke (aggregates)

- Source type: `timegrid_v2.1.0` / `validity_v0.2` NPZ (no EDF read).
- Record status: `COMPLETE`; all six requested families built exactly once.
- Identical resume: six `SKIP_VERIFIED` decisions with zero extractor calls.
- Source NPZ SHA256/stat unchanged before and after the smoke.
- Total build elapsed: 3.423 s.

| family_id | planned dimension | implemented dimension | valid | missing | status | cache bytes |
|---|---|---|---|---|---|---|
| eeg_tail | 288 | 288 | 192 | 96 | BUILT | 17849 |
| eeg_temporal | 96 | 96 | 64 | 32 | BUILT | 13851 |
| eeg_coherence_stage | 90 | 90 | 30 | 60 | BUILT | 12341 |
| caisr_stage | 35 | 35 | 35 | 0 | BUILT | 11160 |
| emg_stage | 72 | 72 | 72 | 0 | BUILT | 12256 |
| resp_stage | 56 | 56 | 56 | 0 | BUILT | 12034 |
| **total** | **637** | **637** | **449** | **188** | — | **79491** |

Missing counts on this record come from per-column support and channel-availability rules already frozen in the family definitions; no formula or support threshold was changed because of them.

## Synthetic cache checks (all in isolated temporary directories)

| check | result |
|---|---|
| A: six-family build then identical run, six skips, zero extractor calls | PASS |
| A: single-family then six-family request order reuse | PASS |
| A: six-family then single-family request order reuse | PASS |
| B: EEG-tail parameter change invalidates only that recipe | PASS |
| C: removed family result in a disposable copy rebuilds only that family | PASS |
| D: family-input removal isolates `REQUIRES_INPUT_REVIEW` without cache publication | PASS |
| D: record-identity mismatch stops the record before any extractor runs | PASS |

## Labeled synthetic examples (no patient data)

Synthetic six-family build over a deterministic synthetic source (linear ramp columns, 120 epochs):

| example | value | support |
|---|---|---|
| featurelab_v2.eeg_tail.f3.delta.wake.q88 | 49908.56 | 20 |
| featurelab_v2.eeg_tail.f3.delta.wake.q95 | 53762.90 | 20 |
| featurelab_v2.eeg_tail.f3.delta.rem.q88 | 49425.56 | 20 |
| featurelab_v2.eeg_tail.f3.delta.rem.q95 | 53279.90 | 20 |

## Overlap and manifest

- `audit/featurelab_batch1_manifest.csv` holds exactly 637 rows and 637 globally unique feature IDs.
- Zero overlap with the 807 frozen unique IDs of the old manifest; the frozen manifest was not modified.

## Review flags

- Batch 2 (553 proposed columns) is audit-only: `BATCH2_IMPLEMENTATION = NOT_STARTED`. All six Batch 2 families are `REQUIRES_REVIEW` in `docs/featurelab/BATCH2_PREPROCESSING_REVIEW.md`.
- No `REQUIRES_INPUT_REVIEW`, `FAILED`, `STOP_INVALID`, or `STOP_RECORD` occurred during the real smoke.
