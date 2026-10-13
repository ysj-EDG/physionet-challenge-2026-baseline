# G2-B0-R CAISR EDF Audit Specification

## Scope

This gate audits the 6,600 data2 training records against the exact CAISR
`algorithmic_annotations` EDF path and hardens two G2-B0 safety contracts. It
does not create a night-level cache, alter the frozen 807/827 feature contract,
read expert annotations or labels, or filter patients.

## Frozen inputs

- Data root: `/database2/physionet2026_kaggle/data2`
- Identity source: `demographics.csv`, restricted to `SiteID`, `BidsFolder`, and
  `SessionID`
- EDF path: `algorithmic_annotations/{SiteID}/{BidsFolder}_ses-{SessionID}_caisr_annotations.edf`
- NPZ root: the frozen `timegrid_v2.1.0` / `validity_v0.2` 6,600-record cache
- Sites and counts: I0002=319, I0006=1142, S0001=5139

All source data are read-only. Exact paths are constructed from the three
identity columns; no glob fallback or expert/human annotation source is used.

## EDF status and stage validity

Each record has exactly one mutually exclusive status:

- `FILE_MISSING`
- `FILE_PRESENT_UNREADABLE`
- `FILE_PRESENT_NO_STAGE`
- `FILE_PRESENT_STAGE_READABLE`

The `stage_caisr` label is matched after `lower().strip()`. A raw value is valid
only when it is finite and exactly one of 1, 2, 3, 4, or 5. Values are never
rounded, imputed, interpolated, or resampled. The expected sample frequency is
1/30 Hz (one sample per 30-second epoch); deviations are review warnings.

## Raw and aligned counts

`raw_valid_count` is computed on the entire EDF stage vector. The aligned vector
has NPZ grid length T, starts as NaN, and receives the first
`min(T, raw_stage_length)` raw values. `aligned_valid_count` is computed using
the same exact validity definition. Raw and aligned 0, 1–4, 5–19, and >=20 bins
remain separate from missing, unreadable, and no-stage file states.

## NPZ cross-check

For every record, only version fields, `annotation_status`,
`stage_code_aligned`, `stage_valid`, and necessary grid-length metadata are
read with `allow_pickle=False`. The audit checks version, annotation status,
NaN-aware aligned stage equality, exact stage-valid mask equality, and count
equality. It does not access `y`, age, diagnosis/follow-up fields, `X_seq`, or
`X_ecg`.

## Private output and recovery

Detailed record-level results live only at
`/database/home/gaohaojie/workspace/python-example-2026-siren-a-caisr-audit-private/g2b0r_caisr6600_v1`.
Fixed sorted chunks are published atomically with a run identity containing
code/data fingerprints. A complete, verified chunk may be resumed; any
identity or integrity mismatch stops execution. Existing output is never
silently overwritten. The Git report contains aggregate site and overall
statistics only.

## Resume validator correction

Cache triplets require exact dtypes (`float64`, `bool`, `int64`) and identical
shapes before indexing. Malformed arrays return `STOP_INVALID` without an
uncaught validation exception. Pooled families require valid=>support>=5 and
invalid=>support<5; CAISR20 and Demo5 require valid=>support=1 and
invalid=>support=0. Value/mask NaN/finite semantics remain enforced.

## Annotation-missing correction

When `annotation_status=missing`, stage code is all NaN and stage validity is
false. Therefore EEG648, Spatial36, and CAISR20 are unavailable. HRV22, EMG48,
Resp28, and Demo5 remain independently governed by their own metadata, and the
record is retained. Expert staging is never used as fallback.

## Gate criteria

The census is complete only when all 6,600 identities close by site, every
record has one file state, all available NPZ metadata are cross-checked, and
private results pass recovery validation. Systematic source-snapshot or stage
semantics conflicts produce `NEEDS_REVIEW`; mapping/data-root/Git failures or
Golden regression failures produce `BLOCKED`.

