# SIREN Branch A G1-B Frozen Feature Contract

## Scope and invariants

This contract freezes the deterministic inputs and aggregation rules for the 827-column, five-view SIREN-inspired Branch A representation. It applies only to the 6,600 v2.1 records already audited from `timegrid_v2.1.0` / `validity_v0.2` (I0002=319, I0006=1,142, S0001=5,139). Retain all 6,600 records: there is no record-level dropna, stage-based exclusion, or recreation of the paper's 6,582-record cohort. Missingness is per output column.

The source NPZ remains read-only. Every missing output is represented as raw `NaN` with `valid_mask=False`; valid numeric zero remains zero. No imputation, scaling, site mapping, label-derived transformation, feature cache, or model fitting belongs to this contract.

## Common aggregation rule

- Mean is the arithmetic mean over finite observations that satisfy the feature's metadata validity expression.
- IQR is `Q75 - Q25`, evaluated with `numpy.quantile(..., method="linear")`.
- EEG and spatial minimum support is five 30 s epochs. HRV minimum support is five valid 5 min sliding windows (30 s stride). EMG and Resp minimum support is five 30 s epochs. These counts are deliberately not interpreted as equal durations.
- If support is below the stated minimum, only that output column is missing. The patient is retained.

## Demo5

The five repeated columns in every view share the same unique IDs, values, and masks:

1. `demo_male = x_static[2]`: Male=1 and Female=0; `x_static[3]=1` (Other/Unknown) is missing.
2. `demo_bmi = x_static[9]`: finite nonzero BMI in kg/m² is retained; upstream zero is restored to missing.
3. `demo_race_white = x_static[8]`.
4. `demo_race_black = x_static[5]`.
5. `demo_race_asian = x_static[4]`.

The sex one-hot `x_static[1:4]` and race one-hot `x_static[4:9]` must be finite, binary, mutually exclusive, and have exactly one active entry. An illegal encoding is a hard contract error. Race Others (`x_static[6]=1`) produces three valid zeros; Race Unavailable (`x_static[7]=1`) produces three missing values. Age (`x_static[0]`) is evaluation metadata only and is absent from all model views.

## EEG648

Model stages are Wake=5, REM=4, N2=2, N3=1, NREM={1,2,3}, and All-stage={1,2,3,4,5}. N1 contributes to NREM but has no standalone output. Each of six leads and nine existing EEG54 source columns is pooled independently by stage with Mean and IQR.

An epoch is valid for a source column only when `stage_valid`, a valid in-group stage code, `eeg_channel_available[lead]`, `eeg_clean_subsegment_count[t,lead] > 0`, and a finite source value all hold. Finite values alone are insufficient because upstream invalid fills may be zero. At least five valid epochs are required per stage/lead/source column.

## Spatial36

These are project-defined SIREN-inspired formulas, not claimed reproductions of the paper's original spatial equations. Per-lead relative power is `logPbandPbeta + logPbeta - logP0` for Delta, Theta, Alpha, and Sigma.

- 24 asymmetry values: F3-F4, C3-C4, and O1-O2 in N2 and REM, computed per same epoch and then averaged.
- 12 anterior-posterior values: `(F3+F4)/2 - (O1+O2)/2` in N2, N3, and REM, computed per same epoch and then averaged. C3/C4 are not inputs.

All required leads, all listed finite source values, stage validity, and lead quality must intersect on the same epoch. Each output requires five such intersections.

## Non-spectral118

### CAISR20

The 20 static columns and order are copied exactly from `data2_caisr20_results/feature_manifest.csv`. Sleep features require available annotation and valid stage support. Arousal, respiratory, and limb features additionally require their respective `arousal_valid`, `resp_event_valid`, or `limb_event_valid` source support. The existing extractor emits an all-zero `x_static[10:196]` block when algorithmic annotations are absent; that documented whole-block pattern is missing. Individual zero-valued CAISR features remain legal whenever their required source metadata is valid, including genuine zero event burdens and safe-division results.

### HRV22

Use only `X_ecg[:,0:11]`; `X_ecg[:,11]` circadian cosine is excluded. A value is valid only under `hrv_success[j] & hrv_feature_valid[j,k] & isfinite(X_ecg[j,k])`. Each row is a 5 min window at 30 s stride; window row `j` aligns to epoch-grid position `j+10` through `ecg_alignment_valid`. The generic NPZ `mask` is not a global multimodal validity mask. Mean/IQR require five valid windows.

### EMG48 and Resp28

EMG uses `X_seq[:,432:456]`; Chin, left leg, and right leg are independent. Validity requires the lead's `emg_channel_available`, `emg_epoch_success[t,lead]`, and a finite source value. Resp uses `X_seq[:,456:470]` and per-feature `resp_feature_valid[t,k]` plus finiteness. Mean/IQR require five valid epochs. For both families, zero is retained when metadata says valid and excluded when it is an invalid fill.

## View layout

| View | Contents | Columns |
|---|---|---:|
| V1 | Wake EEG108 + Demo5 | 113 |
| V2 | REM EEG108 + REM Spatial16 + Demo5 | 129 |
| V3 | N2/N3/NREM EEG324 + N2/N3 Spatial20 + Demo5 | 349 |
| V4 | All-stage EEG108 + Demo5 | 113 |
| V5 | CAISR20 + HRV22 + EMG48 + Resp28 + Demo5 | 123 |

Total view rows are 827; shared Demo5 IDs make 807 unique features. Coherence360, BSR18, circadian cosine, age, new SpO2, embeddings, and all previously excluded families remain absent.

## Failure behavior and gate boundary

Invalid demographic one-hot encoding, incompatible NPZ shapes/versions, absent required validity metadata, or nonconforming source indices must raise an error rather than be silently repaired. Individual feature insufficiency produces per-column missingness and never drops a record. The night-level feature cache was not generated in G1-B; implementation and batch generation remain G2 work after review.
