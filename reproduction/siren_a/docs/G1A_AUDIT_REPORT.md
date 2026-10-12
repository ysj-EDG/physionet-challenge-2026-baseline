# G1-A Audit Report

- Git/input scope: v2.1 NPZ only; repository HEAD at audit run: `048f19c341479e942d46004886dfc8b5ceaff93a`.
- NPZ root: `/database/home/gaohaojie/workspace/python-example-2026/validity_v0_2_full_6600/workspace/result_bundle/npz`; records scanned: **6600**.
- Versions: `{('timegrid_v2.1.0', 'validity_v0.2'): 6600}`; expected extraction `timegrid_v2.1.0`, validity `validity_v0.2`.
- Site counts: {'I0002': 319, 'I0006': 1142, 'S0001': 5139}.
- Stage consistency observations: stage_valid=true with invalid code = 0; stage_valid=false with valid code = 0.

## Stage and EEG support

`stage_code_aligned` and `stage_valid` were checked together. Invalid codes 0, 9, and nonfinite values were excluded from all stage groups. NREM is codes 1/2/3; All-stage is codes 1/2/3/4/5. EEG validity is `eeg_channel_available & eeg_clean_subsegment_count > 0` per lead and epoch. `eeg_total_subsegment_count` was retained for source verification but no clean-ratio threshold was added.

Per-record distributions, site summaries, threshold comparisons, six-lead counts, pair intersections, and F3/F4/O1/O2 intersections are in `audit/g1a_stage_support.csv`. All candidate thresholds 1, 2, 3, 5, 10, 20, 30, and 60 are descriptive only. No threshold was selected and no patient was excluded.
Raw stage-code counts are reported separately from effective stage counts; invalid codes remain excluded and stage_valid is not applied to the raw count.

### Candidate threshold record coverage

| threshold | Wake | REM | N1 | N2 | N3 | NREM | All-stage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6529/6600 (0.989) | 6370/6600 (0.965) | 6511/6600 (0.987) | 6520/6600 (0.988) | 6109/6600 (0.926) | 6521/6600 (0.988) | 6530/6600 (0.989) |
| 2 | 6528/6600 (0.989) | 6331/6600 (0.959) | 6505/6600 (0.986) | 6519/6600 (0.988) | 6037/6600 (0.915) | 6520/6600 (0.988) | 6530/6600 (0.989) |
| 3 | 6526/6600 (0.989) | 6294/6600 (0.954) | 6501/6600 (0.985) | 6519/6600 (0.988) | 5978/6600 (0.906) | 6519/6600 (0.988) | 6529/6600 (0.989) |
| 5 | 6525/6600 (0.989) | 6243/6600 (0.946) | 6490/6600 (0.983) | 6516/6600 (0.987) | 5901/6600 (0.894) | 6517/6600 (0.987) | 6529/6600 (0.989) |
| 10 | 6525/6600 (0.989) | 6122/6600 (0.928) | 6443/6600 (0.976) | 6511/6600 (0.987) | 5729/6600 (0.868) | 6513/6600 (0.987) | 6527/6600 (0.989) |
| 20 | 6517/6600 (0.987) | 5934/6600 (0.899) | 6232/6600 (0.944) | 6504/6600 (0.985) | 5435/6600 (0.823) | 6507/6600 (0.986) | 6524/6600 (0.988) |
| 30 | 6502/6600 (0.985) | 5743/6600 (0.870) | 5729/6600 (0.868) | 6502/6600 (0.985) | 5190/6600 (0.786) | 6505/6600 (0.986) | 6522/6600 (0.988) |
| 60 | 6358/6600 (0.963) | 4922/6600 (0.746) | 3288/6600 (0.498) | 6491/6600 (0.983) | 4210/6600 (0.638) | 6499/6600 (0.985) | 6517/6600 (0.987) |

The table reports count/total and proportion for every candidate threshold; it is descriptive and does not select a threshold.

### Site-stage median effective epochs

| site | Wake | REM | N1 | N2 | N3 | NREM | All-stage |
|---|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 246.0 | 92.0 | 53.0 | 410.0 | 84.0 | 557.0 | 911.0 |
| I0006 | 200.0 | 80.0 | 46.0 | 426.0 | 56.0 | 547.5 | 835.0 |
| S0001 | 197.0 | 108.0 | 64.0 | 411.0 | 87.0 | 581.0 | 916.0 |

### Demographic aggregate distributions

- I0002 (n=319): Sex {'Female': 156, 'Male': 163}; Race {'<MISSING>': 18, 'Asian': 19, 'Black': 42, 'Others': 12, 'White': 228}; BMI missing/nonfinite=228, zero=0, descriptive suspicious-range flags=0.
- I0006 (n=1142): Sex {'Female': 639, 'Male': 503}; Race {'<MISSING>': 30, 'Asian': 24, 'Black': 459, 'Others': 4, 'White': 625}; BMI missing/nonfinite=196, zero=0, descriptive suspicious-range flags=0.
- S0001 (n=5139): Sex {'Female': 2304, 'Male': 2835}; Race {'<MISSING>': 137, 'Asian': 122, 'Black': 308, 'Others': 258, 'White': 4314}; BMI missing/nonfinite=4638, zero=0, descriptive suspicious-range flags=0.

The audit uses exact same-epoch intersections for all spatial support counts; it does not substitute the minimum of marginal lead counts. The NPZ `mask` field was not used for EEG, EMG, Resp, or stage filtering; source code identifies it as ECG alignment only.

## Demographics audit

- Candidate source: `/database2/physionet2026_kaggle/data2/demographics.csv`; only SiteID, BidsFolder, SessionID, Sex, Race, BMI were read.
- CSV rows: 6600; unique primary keys: 6600; duplicate-row excess: 0.
- NPZ unique keys: 6600; exact key intersection: 6600; unmatched and encoding summaries are aggregate-only in `audit/g1a_join_consistency.csv`.
- Existing helper semantics were applied exactly for sex, race, and BMI comparison to x_static[1:4], x_static[4:9], and x_static[9]. Age was not included in Demo5.
- Raw category and BMI missing/zero/nonfinite/suspicious-range summaries are in `audit/g1a_demographics_categories.csv`; no patient-level identifiers are written.

## Draft Manifest

View row counts: {'V1': 113, 'V2': 129, 'V3': 349, 'V4': 113, 'V5': 123}; total rows: 827; unique feature IDs: 807.
The manifest has 827 rows, contiguous zero-based positions within each view, and the same five Demo5 unique IDs repeated across views. EEG648, candidate spatial36, and non-spectral118 are source-defined drafts. Demo5 and unresolved validity/small-n rules are `PENDING_G1B_REVIEW`.

## Findings and limits

- 1103 NPZ is unavailable and was not searched for or substituted.
- This run did not calculate patient-level features, create a Night-level Feature Cache, fit transforms, use labels, train models, or rerun Validity v0.2 QC.
- Any anomaly is preserved as evidence for review; this Gate does not repair source data or mappings.

See `docs/G1A_DECISION_REQUESTS.md` for the decisions intentionally left open.
