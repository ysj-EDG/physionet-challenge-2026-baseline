# H100 validity sample extraction

## Scope

- Code reference: `f608be10cf068e4f70f9b539a9bfdd0849cf9f04` (`timegrid_v2.1.0`, `validity_v0.1`).
- Actual cohort: 16 records. DATA2 uses 2 positive + 2 negative per I0002/I0006/S0001; DATA3 uses 2 random unlabeled records per I0004/I0007.
- DATA3 labels are unavailable. Runtime `y=0` is an operational placeholder and not scientific ground truth.
- Medex node: `yurui`; GPU: `GPU 0: NVIDIA H100 PCIe (UUID: GPU-d367af3b-ac4a-2be9-1a44-a7825fbe65d3)`; environment: `eeg_env`; workers: 4; BLAS threads per worker: 1.
- `.git` was excluded from synchronization. Provenance is the locally verified code reference plus per-file SHA256 and NPZ `source_sha256_json`.

## Extraction

- Result: 16 success, 0 failure.
- Schema/provenance: 16/16 passed shapes, versions, finite-core, mask alias, and source-hash checks.
- DATA2 regression: PASS for all 12 records; `X_seq`, `X_ecg`, `x_static`, `y`, and `mask` are exactly equal to frozen v2.0 cache.
- Frozen `npz_data2` count remained 6600.

## First look

| Site | N | Spectral valid | Coherence valid | Resp valid | Stage valid | Arousal valid | Resp-event valid | Limb valid | ECG aligned | HRV success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 4 | 0.900 | 0.763 | 0.998 | 0.993 | 1.000 | 1.000 | 1.000 | 0.988 | 0.954 |
| I0004 | 2 | 0.726 | 0.670 | 0.975 | 0.995 | 1.000 | 1.000 | 1.000 | 0.000 | N/A |
| I0006 | 4 | 0.967 | 0.934 | 0.998 | 0.993 | 1.000 | 1.000 | 1.000 | 0.988 | 0.990 |
| I0007 | 2 | 0.944 | 0.881 | 0.995 | 0.994 | 1.000 | 1.000 | 0.000 | 0.989 | 0.985 |
| S0001 | 4 | 0.985 | 0.909 | 0.875 | 0.993 | 1.000 | 1.000 | 1.000 | 0.989 | 1.000 |

External observations were not used to alter rules:
- Both I0004 records have no ECG windows/alignment; both also lack chin EMG.
- Both I0007 records have limb-event validity 0.
- One S0001 record lacks airflow, producing an approximately 0.5 Resp14 validity rate.
- All 16 records have consistent reported modality/final sequence lengths.

## Artifacts

- `selection/selected_records.csv` and ID lists
- `raw_subset/` (only selected raw EDF/annotation files)
- `workspace/medex-run.yaml`, `workspace/run_h100.py`, `workspace/source_files.sha256`
- `npz/` (16 extracted NPZ files)
- `logs/data2_manifest.jsonl`, `logs/data3_manifest.jsonl`
- `audit/core_regression.csv`
- `audit/npz_integrity.csv`
- `audit/sample_metadata_summary.csv` and `audit/site_metadata_summary.csv`
- `audit/zero_validity_summary.csv`

## Status

`V0_STATUS = REAL_SAMPLE_AUDIT_PENDING_REVIEW`

No validity rule was changed from these observations. No full extraction or model training was run.
