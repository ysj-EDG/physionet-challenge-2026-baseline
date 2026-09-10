# DATA2 timegrid_v2 cache start report

Status: **BLOCKED at the remote data mount, after implementation and successful 6-record acceptance.** No model training ran and no existing NPZ was overwritten.

## Scope and version

- Reference HEAD: `fc35c07f137cd1b26a277eaa2f70cfd8b0e8f4fc`; changes remain uncommitted.
- Explicit version: `timegrid_v2.0.0`. The inherited legacy API remains the default.
- New production entry: `per_epoch_features/timegrid_v2_extractor.py`; restartable batch entry: `extract_data2_timegrid_v2.py`.
- Dimensions remain X_seq 483, X_ecg 12, x_static 196. ECG offset and mask remain 10 epochs.
- No feature formula, threshold, label definition, source NPZ, or model/training code was changed.

## Header-only finding

The scan covered all 6,600 rows: I0002 319, I0006 1,142, S0001 5,139. All 6,600 physiological EDFs exist; 6,530 CAISR EDFs exist and 70 are absent. Every readable CAISR record has unavailable stages: 39,180 invalid epochs (38,886 code 9; 294 code 0). Stage frequency is 1/30 Hz. CAISR minus physiological duration ranges from -29 to 0 seconds (median -12). CAISR dates are anonymized and clocks are 00:00, so the header offset cannot be established reliably. V2 records `edf_time_offset_sec=NaN` and `alignment_status=official_relative_origin;header_offset_unavailable`; it uses the official relative record grid without stretching.

## Implementation and acceptance

V2 retains invalid stage positions, emits five zeros with `stage_valid=False`, uses actual EDF rates (arousal 2 Hz, respiratory/limb 1 Hz), and prevents bouts/transitions crossing unknown epochs. It saves raw/aligned stages, probabilities and validity, six-channel availability, real clean/total EEG subsegment counts, HRV success, modality lengths/status, identity, source hashes, dependency versions, and epoch coordinates. The old zero `pvalues` output is not used as quality.

Four finite tests passed: the synthetic N2/Unavailable/REM/N3 sequence stays four epochs and REM stays at 60-90 s; unknown splits bouts and is not a transition; complete grids match legacy outside the fix; metadata-on/off physiological features are identical and quality derives from artifact flags.

The fixed smoke list is in `data2_smoke_ids.txt`: two records per site selected by ID, duration and channel completeness only; record selection did not use outcome. One S0001 record intentionally covers missing CAISR. All 6 caches validate with `allow_pickle=False`; 0 failed. Five have six unavailable epochs preserved; the missing-CAISR record has 1,018 unknown epochs rather than fake Wake. Shapes are approximately 959-1,018 x 483 and 950-1,009 x 12; all core values are finite.

Single-record elapsed time was 523.5 s. The three-worker run completed five new records plus one validated skip in 16:13.7; new-record times were 247-672 s. Observed maximum RSS was 3.12 GiB in the single run and 2.33 GiB in the three-worker command report. Six caches occupy 8.5 MiB, projecting roughly 9-10 GiB for 6,600. Measured three-worker throughput was 18.5 new records/hour; ideal scaling to 16 workers is about 2.8 days, but I/O contention makes 3-5 days more realistic. The Medex runner caps at 16 workers, reserves 5 GiB/worker, and fixes all BLAS pools to one thread.

## Full launch status and recovery

An actual Medex submission was made at 2026-09-10 05:07 UTC with `eeg_env` on H100 GPU 0. Medex history entry 1 describes it as entry 1, but the client exposes no stable job ID. The strict remote preflight stopped before extraction because `/database2/physionet2026_kaggle/data2` was unavailable on the remote host. The only registered remote dataset is `physionet2026_data` (129 GiB; `training_set/` and `supplementary_set/`), not data2 (local data2 is 1.3 TiB). It was not substituted or uploaded. The complete evidence is `data2_medex_submit.log`.

After data2 is mounted at that exact path, resume with:

```bash
/opt/miniconda3/bin/medex-run run feat_input/medex_data2_timegrid_v2.yaml
```

The batch writer uses a same-directory temporary file, fsync, validation, and atomic rename. It skips only a valid matching version and logs each success, failure, or existing cache to `npz_data2/manifest.jsonl`, so the same command is the recovery command. Current local output contains only the six accepted smoke caches; full success counts are pending the required remote mount.
