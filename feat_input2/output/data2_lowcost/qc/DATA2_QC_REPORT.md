# Data2 QC report

- Source: `/database/home/gaohaojie/workspace/python-example-2026/npz_data2`.
- Exact coverage: 6600 unique manifest records and 6600 one-to-one local NPZ files; 6600 success and 0 failure.
- Full streaming validation passed for record/site/label, age, version, 483/12/196 shapes, metadata alignment, and finite core arrays.
- Extraction version: `timegrid_v2.0.0`; embedded extraction code SHA value(s): `unknown`.
- The remote extraction checkout did not contain Git metadata, so `code_git_sha=unknown`; exact extraction-source SHA256 values are retained in run metadata and each NPZ.
- Unique patients: 6600; multi-session patients: 0; maximum sessions per patient: 1.

## Site composition

| site | n | patients | positives | negatives | positive_prevalence | multi_session_patients |
|---|---|---|---|---|---|---|
| I0002 | 319 | 319 | 52 | 267 | 0.1630 | 0 |
| I0006 | 1142 | 1142 | 112 | 1030 | 0.0981 | 0 |
| S0001 | 5139 | 5139 | 334 | 4805 | 0.0650 | 0 |
| Total | 6600 | 6600 | 498 | 6102 | 0.0755 | 0 |

## Age-conditioned pair counts

| site | n | positives | negatives | eligible_ac_pairs_gap2 |
|---|---|---|---|---|
| I0002 | 319 | 52 | 267 | 1886 |
| I0006 | 1142 | 112 | 1030 | 14493 |
| S0001 | 5139 | 334 | 4805 | 182330 |

## Availability summary

| site | n | stage_valid_ratio_mean | stage_valid_ratio_median | unavailable_stage_ratio_mean | eeg_channels_available_mean | eeg_any_available_rate | eeg_all6_available_rate | eeg_clean_ratio_aggregate | hrv_success_window_rate | hrv_any_success_record_rate | ecg_available_record_rate | female_rate | male_rate | sex_other_unknown_rate | race_asian_rate | race_black_rate | race_others_rate | race_unavailable_rate | race_white_rate | bmi_missing_rate | bmi_median_available |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I0002 | 319 | 0.9864 | 0.9935 | 0.0136 | 5.7147 | 1.0000 | 0.8746 | 0.8362 | 0.9978 | 0.9937 | 0.9937 | 0.4890 | 0.5110 | 0.0000 | 0.0596 | 0.1317 | 0.0376 | 0.0564 | 0.7147 | 0.7147 | 29.6300 |
| I0006 | 1142 | 0.9894 | 0.9929 | 0.0106 | 6.0000 | 1.0000 | 1.0000 | 0.8405 | 0.9833 | 0.9991 | 1.0000 | 0.5595 | 0.4405 | 0.0000 | 0.0210 | 0.4019 | 0.0035 | 0.0263 | 0.5473 | 0.1716 | 32.2200 |
| S0001 | 5139 | 0.9802 | 0.9935 | 0.0198 | 5.9875 | 1.0000 | 0.9922 | 0.8307 | 0.9917 | 0.9947 | 0.9949 | 0.4483 | 0.5517 | 0.0000 | 0.0237 | 0.0599 | 0.0502 | 0.0267 | 0.8395 | 0.9025 | 28.8800 |
| Total | 6600 | 0.9821 | 0.9934 | 0.0179 | 5.9765 | 1.0000 | 0.9879 | 0.8326 | 0.9907 | 0.9955 | 0.9958 | 0.4695 | 0.5305 | 0.0000 | 0.0250 | 0.1226 | 0.0415 | 0.0280 | 0.7829 | 0.7670 | 30.8150 |

Age distributions split by site and label are in `age_summary.csv`. All availability statistics use metadata already stored in the NPZ; no raw EDF was read.
