# P0 feature clipping and scaling audit

## Scope and result

This read-only audit scanned 1103/1103 cached records under `/database/home/gaohaojie/workspace/python-example-2026/npz_new`.
It did not modify extractors, training/inference code, NPZ files, splits, or model weights, and it did not train a model.
Full finite/zero/clip counts and extrema are exact. Quantiles are reproducible approximations from uniformly spaced rows per record; observation-weighted and subject-equal summaries are both retained.

## Verified data layout

- Split counts: `{'train': 733, 'val': 158, 'test': 158, 'external': 54}`; site counts: `{'S0001': 857, 'I0006': 192, 'I0002': 54}`.
- Required array dimensions were verified as `{'X_seq': 483, 'X_ecg': 12, 'x_static': 196}`.
- Cache shape audit found 6 records with `X_ecg.shape[0] == 0`; these valid empty arrays are retained as a modality-availability finding, not a read failure.
- Read errors: 0; missing caches: 0; unexpected caches: 0.
- `mask` marks ECG alignment only. It must not filter EEG epochs and does not prove HRV computation success.
- No cache schema/extractor version field exists, so uniform extraction version cannot be proven from NPZ contents or timestamps.

## Production path and confirmed destructive preprocessing

- Extraction concatenates the 483-dimensional sequence and replaces NaN/Inf with zero in `per_epoch_features/per_epoch_extractor.py:278-284`.
- Training preserves raw age for the metric, then applies `nan_to_num` plus a uniform `[-50,50]` clip to all three branches in `train_lstm.py:109-125`.
- Inference validates and zero-fills nonfinite values in `team_code.py:175-198`, then applies the same uniform clip in `team_code.py:645-670`.
- Sigmoid clipping, probability clipping, source-level probability bounds, waveform QC, and gradient clipping are separate safeguards and were not treated as defects.

## Largest observed clipping damage

```text
  branch  index                                    name      family  flattened_by_old_clip_fraction_finite  record_clip_fraction_p95  min   max
   X_ecg      0                      model_HRV_MedianNN     ecg_hrv                               0.990631                         1    0 10315
x_static     10                     caisr_sleep_trt_sec caisr_sleep                               0.988214                         1    0 40890
x_static     15               caisr_sleep_wake_time_sec caisr_sleep                               0.987307                         1    0 25140
x_static     22                   caisr_sleep_dur_w_sec caisr_sleep                               0.987307                         1    0 25140
x_static     39          caisr_sleep_mean_bout_wake_sec caisr_sleep                               0.987307                         1    0 24960
x_static     49           caisr_sleep_max_bout_wake_sec caisr_sleep                               0.987307                         1    0 24960
x_static     64           caisr_sleep_p90_bout_wake_sec caisr_sleep                               0.987307                         1    0 24960
x_static     69           caisr_sleep_p95_bout_wake_sec caisr_sleep                               0.987307                         1    0 24960
x_static     11                     caisr_sleep_tst_sec caisr_sleep                               0.985494                         1    0 30120
x_static     24                  caisr_sleep_dur_n2_sec caisr_sleep                               0.985494                         1    0 27000
x_static     51             caisr_sleep_max_bout_n2_sec caisr_sleep                               0.985494                         1    0 26970
x_static     61             caisr_sleep_p75_bout_n2_sec caisr_sleep                               0.985494                         1    0 20235
x_static     66             caisr_sleep_p90_bout_n2_sec caisr_sleep                               0.985494                         1    0 24276
x_static     71             caisr_sleep_p95_bout_n2_sec caisr_sleep                               0.985494                         1    0 25623
x_static     82 caisr_sleep_longest_cont_sleep_bout_sec caisr_sleep                               0.985494                         1    0 27210
```

`clipping_damage_top.csv` contains all 691 columns in transparent descending order, not only this excerpt.

## Missingness interpretation

- Cached NaN/Inf counts describe the post-extraction cache, not raw-source availability. All-zero block heuristics were: `{'StageEvent_all_zero': 13, 'CAISR_static_all_zero': 13, 'BSR_all_zero': 20, 'Resp_all_zero': 1}`.
- HRV first-11 all-zero windows: 9170/978740. This is only a failure heuristic; circadian cosine does not validate HRV.
- A zero may be a valid no-event value, one-hot off state, true BSR zero, or an extraction/fallback placeholder. This audit does not convert zeros back to missing values.

## Candidate transforms (offline numeric preview only)

Candidates A-D were fitted only on the confirmed 733-record train manifest. Test/external values were not used to fit centers, scales, units, thresholds, or transform choices.
Categorical, fraction, probability, percentage and named ratio columns were left on their physical scale. Candidate D log1p was not run because feature-specific physical reference units have not been approved; eligible D rows fall back to C and are explicitly marked pending.
These previews do not establish a best transform and make no performance claim. Physical unit normalization (notably percent versus fraction) still requires manual schema approval before implementation.

## Formal repair design (not implemented)

1. Fit one shared column-aware transformer on unique inner-training records before balanced sampling.
2. Exclude padding and ECG non-alignment from fit; transform real values first, then create model padding zeros.
3. Preserve a separate raw-age copy for Age-conditioned AUROC; never transform it in place.
4. Save schema checksum, feature order, unit conversions, per-column transform, center/scale, missing policy, constant rules, and training-manifest checksum in the checkpoint.
5. Make cached and raw-extraction inference paths call the same transform exactly once; reject schema/checksum mismatches and never attach a new scaler to old weights.
6. Missing indicators or architecture changes belong to a later ablation and are not part of this P0 repair.

## Limitations and unresolved items

- Historical NPZ extraction-version uniformity is not provable because the archives have no version metadata.
- Source-level missingness and quality cannot generally be recovered after zero filling; no EDF re-extraction was performed.
- Several CAISR composite indices and coherence-derived quantities have definition-specific domains; schema confidence is explicitly marked and requires human review.
- Candidate D log1p was not run: physical reference units must be approved before formal code changes.

## Reproduction

```bash
conda run -n sleepfm_env python feat_input/test_audit_feature_scaling.py
conda run -n sleepfm_env python feat_input/audit_feature_scaling.py --cache-root npz_new --split-root split --sample-cap 64
```

Elapsed audit time: 212.5 seconds.
