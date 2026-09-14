# Extraction status before P6

Snapshot time: 2026-09-14 12:09 UTC. This was a read-only check. Neither extraction task was stopped, restarted, cleaned, or modified. P6 uses only the frozen legacy `npz_new` cache.

## data 1103 timegrid_v2

- Process: active on the local P6000 host in `sleepfm_env`, with the parent process and 12 workers present.
- Output: `/database/home/gaohaojie/workspace/python-example-2026/npz_1103_timegrid_v2/`.
- Manifest: `npz_1103_timegrid_v2/manifest.jsonl`.
- Log: `feat_input/data1103_timegrid_v2_extract.log`.
- Completed NPZ: 291 / 1103 (26.38%).
- Manifest status: 291 success, 0 existing, 0 failed.
- Latest completed record: `sub-I0006179014181_ses-1` at the snapshot.
- Liveness: the log was updated 47 seconds before the snapshot and all 12 workers were consuming CPU.
- Three newest NPZ files were sampled with `allow_pickle=False`. Each had `X_seq.shape[1] == 483`, `X_ecg.shape[1] == 12`, `x_static.shape == (196,)`, `extraction_version == timegrid_v2.0.0`, and finite core arrays.
- Current disk use: 390 MiB (407,196,515 bytes across NPZ files).

## data2 6600 timegrid_v2

- Process: active in Medex, run suffix `9755ed`, environment `eeg_env`.
- Remote output: `<Medex run>/python-example-2026/npz_data2/`.
- Configured rsync-back destination after normal completion: `/database/home/gaohaojie/workspace/python-example-2026/npz_data2/`.
- Completed: 3,781 / 6,600 (57.29%).
- Streamed status: 3,781 success, 0 failed.
- Latest completed record: `sub-S0001115789563_ses-1` at the snapshot.
- Liveness: the streamed log was updated about 2 seconds before the snapshot; the local Medex parent, SSH process, and remote registered task were active.
- Rsync back: not started. It is configured to run only after remote extraction exits normally.
- Local `npz_data2`: 6 smoke NPZ files, 8.5 MiB; these are not evidence of rsync-back completion.
- No traceback, abnormal exit, disk-full message, or SSH disconnect was present in the live log at the snapshot. Medex reported remote `/data` at 96% utilization, so disk capacity remains a condition to monitor, but it had not caused a reported failure.
