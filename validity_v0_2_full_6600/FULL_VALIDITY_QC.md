# DATA2 timegrid_v2.1.0 / validity_v0.2 full QC

## A. Dataset/schema integrity

- NPZ root: `/database/home/gaohaojie/workspace/python-example-2026/validity_v0_2_full_6600/workspace/result_bundle/npz`
- Records: **6,600 files / 6,600 unique record IDs**
- Extraction versions: `timegrid_v2.1.0`: 6,600
- Validity schema versions: `validity_v0.2`: 6,600
- `source_sha256_json` variants: **1**
- NPZ `code_git_sha`: `unknown`: 6,600
- H100 run-context source SHA: `5f33b48b4bfc4bc012dca1188255abfd8834c61f`
- Core shape/finite, metadata shape, and `mask == ecg_alignment_valid`: **PASS**

| Site | Records | Physical epochs | ECG windows |
|---|---:|---:|---:|
| I0002 | 319 | 290,169 | 287,309 |
| I0006 | 1,142 | 967,975 | 957,697 |
| S0001 | 5,139 | 4,671,110 | 4,612,664 |

## B. Site-level validity coverage

| Site | Spectral | Coherence | BSR | EMG epoch success | Resp14 | Stage | HRV success |
|---|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 94.1095% | 87.1480% | 95.1358% | 100.0000% | 99.9422% | 99.3421% | 99.7800% |
| I0006 | 96.3429% | 93.1235% | 100.0000% | 99.9494% | 99.3447% | 98.9342% | 98.3250% |
| S0001 | 97.5336% | 91.9549% | 99.8006% | 99.3214% | 99.5009% | 98.3246% | 99.1745% |

Rates are pooled object-level rates: record availability uses records; temporal validity uses actual epochs/windows; feature validity uses actual feature objects.

## C. EEG

| Site | F3 | F4 | C3 | C4 | O1 | O2 | Spectral valid | Mean clean ratio | Common-clean zero epochs | Coherence valid | BSR valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 100.0000% | 93.4169% | 99.6865% | 89.3417% | 99.6865% | 89.3417% | 94.1095% | 83.6202% | 5.0078% | 87.1480% | 95.1358% |
| I0006 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 96.3429% | 84.0491% | 6.8765% | 93.1235% | 100.0000% |
| S0001 | 99.8249% | 99.6108% | 99.9416% | 99.8054% | 99.8638% | 99.7081% | 97.5336% | 83.0739% | 7.7108% | 91.9549% | 99.8006% |

No clean-ratio threshold was introduced; `clean > 0` is used only for the frozen intrinsic spectral-valid rule.

## D. EMG

| Site | Channel | Available | Preprocessing success | Epoch success |
|---|---:|---:|---:|---:|
| I0002 | chin | 100.0000% | 100.0000% | 100.0000% |
| I0002 | lleg | 100.0000% | 100.0000% | 100.0000% |
| I0002 | rleg | 100.0000% | 100.0000% | 100.0000% |
| I0006 | chin | 100.0000% | 100.0000% | 100.0000% |
| I0006 | lleg | 99.9124% | 99.9124% | 99.9242% |
| I0006 | rleg | 99.9124% | 99.9124% | 99.9242% |
| S0001 | chin | 99.0270% | 99.0270% | 99.0273% |
| S0001 | lleg | 99.7665% | 99.7665% | 99.7818% |
| S0001 | rleg | 99.1438% | 99.1438% | 99.1551% |

## E. Resp

| Site | Airflow avail | Thorax avail | Abdomen avail | Airflow prep | Thorax prep | Abdomen prep | Resp14 | Airflow7 | Thorax2 | Abdomen2 | Joint3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 99.9422% | 99.8844% | 100.0000% | 100.0000% | 100.0000% |
| I0006 | 99.9124% | 99.9124% | 99.9124% | 99.9124% | 99.9124% | 99.9124% | 99.3447% | 99.3100% | 99.9242% | 99.9242% | 98.6530% |
| S0001 | 99.6887% | 99.7860% | 99.8054% | 99.6887% | 99.7860% | 99.8054% | 99.5009% | 99.2483% | 99.8017% | 99.8228% | 99.6751% |

Per-feature validity:

| Resp feature | I0002 | I0006 | S0001 |
|---|---:|---:|---:|
| airflow_rate_bpm | 99.6419% | 97.9054% | 98.2432% |
| airflow_cycle_cv | 99.5485% | 97.6441% | 98.0393% |
| airflow_amp_local_norm | 100.0000% | 99.9242% | 99.6911% |
| airflow_amp_iqr_over_median | 100.0000% | 99.9242% | 99.6911% |
| airflow_reduction30_fraction | 100.0000% | 99.9242% | 99.6911% |
| airflow_near_absent_fraction | 100.0000% | 99.9242% | 99.6911% |
| airflow_longest_reduction_sec | 100.0000% | 99.9242% | 99.6911% |
| thorax_amp_local_norm | 100.0000% | 99.9242% | 99.8017% |
| thorax_reduction30_fraction | 100.0000% | 99.9242% | 99.8017% |
| abdomen_amp_local_norm | 100.0000% | 99.9242% | 99.8228% |
| abdomen_reduction30_fraction | 100.0000% | 99.9242% | 99.8228% |
| thorax_abd_corr | 100.0000% | 98.6531% | 99.6752% |
| thorax_abd_lag_sec | 100.0000% | 98.6531% | 99.6752% |
| thorax_abd_paradox_fraction | 100.0000% | 98.6530% | 99.6750% |

## F. Stage/Event

| Site | Stage valid | Arousal valid | Arousal source-missing records | Resp-event valid | Resp source-missing records | Limb valid | Limb source-missing records |
|---|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 99.3421% | 99.9976% | 2 | 99.9976% | 2 | 99.9976% | 2 |
| I0006 | 98.9342% | 99.6396% | 4 | 99.2171% | 9 | 99.6396% | 4 |
| S0001 | 98.3246% | 98.9734% | 72 | 98.9538% | 73 | 98.2997% | 99 |

Invalid-stage code distributions:

| Site | Aligned code among invalid epochs | Raw code among invalid epochs |
|---|---:|---:|
| I0002 | `9`: 1,902, `NaN`: 7 | `9`: 1,902, `raw unavailable`: 7 (unmapped records: 2) |
| I0006 | `0`: 30, `9`: 6,798, `NaN`: 3,489 | `0`: 30, `9`: 6,798, `raw unavailable`: 3,489 (unmapped records: 4) |
| S0001 | `0`: 264, `9`: 30,186, `NaN`: 47,808 | `0`: 264, `9`: 30,186, `raw unavailable`: 47,808 (unmapped records: 64) |

## G. ECG/HRV

| Site | ECG unavailable records | Available but <300 s | X_ecg empty records | ECG alignment | HRV success | HRV feature valid | Circadian valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 0 | 2 | 2 | 98.9051% | 99.7800% | 99.7799% | 100.0000% |
| I0006 | 0 | 0 | 0 | 98.8202% | 98.3250% | 98.3199% | 100.0000% |
| S0001 | 14 | 12 | 26 | 98.6393% | 99.1745% | 99.1734% | 100.0000% |

HRV feature-level validity:

| HRV feature | I0002 | I0006 | S0001 |
|---|---:|---:|---:|
| MedianNN | 99.7800% | 98.3250% | 99.1745% |
| MCVNN | 99.7800% | 98.3250% | 99.1745% |
| CVNN | 99.7800% | 98.3250% | 99.1745% |
| CVSD | 99.7800% | 98.3250% | 99.1745% |
| pNN20 | 99.7800% | 98.3250% | 99.1745% |
| log_LF | 99.7793% | 98.2966% | 99.1732% |
| log_HF | 99.7800% | 98.3250% | 99.1745% |
| HF_over_LF_plus_HF | 99.7793% | 98.2966% | 99.1732% |
| SD1_over_SD2 | 99.7800% | 98.3250% | 99.1654% |
| Symbolic_0V | 99.7800% | 98.3250% | 99.1745% |
| Symbolic_2UV | 99.7800% | 98.3250% | 99.1745% |

## H. Edge-case counts

| Site | Edge case | Records |
|---|---:|---:|
| I0002 | ECG source absent | 0 |
| I0002 | ECG present but duration <300 s | 2 |
| I0002 | No 5-min ECG window | 2 |
| I0002 | Arousal source entirely invalid | 2 |
| I0002 | Resp-event source entirely invalid | 2 |
| I0002 | Limb source entirely invalid | 2 |
| I0006 | ECG source absent | 0 |
| I0006 | ECG present but duration <300 s | 0 |
| I0006 | No 5-min ECG window | 0 |
| I0006 | Arousal source entirely invalid | 4 |
| I0006 | Resp-event source entirely invalid | 9 |
| I0006 | Limb source entirely invalid | 4 |
| S0001 | ECG source absent | 14 |
| S0001 | ECG present but duration <300 s | 12 |
| S0001 | No 5-min ECG window | 26 |
| S0001 | Arousal source entirely invalid | 72 |
| S0001 | Resp-event source entirely invalid | 73 |
| S0001 | Limb source entirely invalid | 99 |

`ecg_channel_available` and `ecg_signal_duration_sec` distinguish source absence from a present ECG shorter than 300 s. Event-source missingness is represented by source-specific validity arrays rather than event values.

Zero × validity evidence (strict `value == 0`):

| Family | Zero + valid | Zero + invalid | Nonzero + valid | Nonzero + invalid | Zero rate | Valid rate |
|---|---:|---:|---:|---:|---:|---:|
| EEG spectral | 0 | 9,055,818 | 311,123,898 | 0 | 2.8284% | 97.1716% |
| EEG coherence | 457,735 | 172,673,880 | 1,961,399,825 | 0 | 8.1110% | 91.9105% |
| EEG BSR | 104,499,551 | 421,707 | 1,805,314 | 0 | 98.3085% | 99.6049% |
| EMG | 41,785,948 | 772,536 | 99,743,612 | 0 | 29.9071% | 99.4571% |
| Resp14 | 10,040,322 | 417,563 | 72,551,671 | 0 | 12.5984% | 99.4970% |
| Stage5 | 23,355,080 | 452,420 | 5,838,770 | 0 | 80.3052% | 98.4739% |
| Arousal | 4,349,474 | 51,449 | 1,528,331 | 0 | 74.2239% | 99.1323% |
| Resp event | 27,152,638 | 282,260 | 2,211,372 | 0 | 92.5408% | 99.0479% |
| Limb event | 8,684,968 | 165,840 | 3,007,700 | 0 | 74.6368% | 98.6015% |
| HRV11 | 156,260 | 603,337 | 63,674,773 | 0 | 1.1789% | 99.0636% |

- Valid physiological/algorithmic zeros observed: **YES**
- Invalid placeholder zeros observed: **YES**

## I. Unexpected findings

- No schema, shape, finite-core, identity, or mask-alias inconsistency was found.
- No new unexplained validity contradiction was found by the predefined consistency checks.
- NPZ `code_git_sha` is uniformly `unknown` because the staged Medex workspace excluded `.git`; provenance is instead anchored by run-context SHA `5f33b48b4bfc4bc012dca1188255abfd8834c61f` and one uniform, preflight-verified `source_sha256_json` set.

## J. Decision

**VALIDITY_V0_2_FULL_QC = PASS**

All 6,600 records have the expected schema/version/shapes, finite core arrays, an exact ECG mask alias, uniform source hashes, and no new internal validity contradiction. The observed missing-source and legitimate-zero patterns are represented without conflating zero with invalidity.
