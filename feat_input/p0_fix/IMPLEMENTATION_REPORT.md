# P0 typed_v1 input preprocessing implementation report

## Outcome

- Frozen scaler fitted from 733 unique train records only.
- Offline transformation covered 1103/1103 cached records with zero errors.
- All typed_v1 outputs were finite: 0 nonfinite values.
- Cache file size/mtime metadata remained unchanged: True.
- P0 engineering input repair passed mapping, path, finite-value, cache-integrity and test checks. No model training and no AUROC/AUPRC claim.

## Fixed interface

- One shared FeatureScaler implements typed_v1 and exact legacy_clip modes.
- Explicit per-column unit/nonlinear/type rules precede train-only robust parameters.
- Raw age is copied before model-input transformation; ECG offset and dimensions are unchanged.
- Padding, pre-offset ECG zeros, empty ECG and documented legacy sentinels remain zero.
- New checkpoints save preprocessing state; old checkpoints require LSTM_ALLOW_LEGACY_INPUT=1.

## Data accounting

- Raw ECG windows: 978740; consumed: 971103; excluded tail: 7637.
- Legacy all-zero HRV sentinel windows: 9072.
- Legacy all-zero CAISR sentinel records: 13.
- Scale methods: {'iqr': 390, 'identity': 299, 'p95_p05': 2}.
- typed_v1 abs(value)>50 count across global feature rows: 1088.
- legacy_clip abs(value)>50 count: 0.

## Remaining large transformed values

```text
  branch  index                                     name  abs_gt_50_count  abs_gt_50_fraction_finite      min      max
   X_seq    441               emg_lleg_envelope_iqr_norm              584                0.000593624 -1.03840 110.7811
   X_seq    433               emg_chin_envelope_iqr_norm              266                0.000270381 -1.57880 163.0198
   X_seq    449               emg_rleg_envelope_iqr_norm              236                0.000239887 -1.10209 125.2871
x_static    106 caisr_arousal_min_inter_arousal_interval                1                0.000906619 -1.20843  56.0730
x_static    189              caisr_limb_delta_limb_index                1                0.000906619 -193.262  32.9364
```

These values are finite and rare but remain explicit review items; no automatic z clipping was added.


## Key feature comparison

           feature     variant      n       min        p1         p50      p99         max
               age         raw   1103        50        50          61       83          88
               age legacy_clip   1103        50        50          50       50          50
               age    typed_v1   1103 -0.916667 -0.916667           0  1.83333        2.25
               BMI         raw   1103         0         0           0  49.1632       63.08
               BMI legacy_clip   1103         0         0           0  49.1632          50
               BMI    typed_v1   1103   -1.5088 -0.953474           0  2.08547      3.7189
          MedianNN         raw 971103         0       470         930     1580       10315
          MedianNN legacy_clip 971103         0        50          50       50          50
          MedianNN    typed_v1 971103  -2.93902  -1.59756           0  3.15854     45.7683
             pNN20         raw 971103         0         0     35.0877  94.4444     99.8413
             pNN20 legacy_clip 971103         0         0     35.0877       50          50
             pNN20    typed_v1 971103         0         0    0.350877 0.944444    0.998413
               TRT         raw   1103         0         0       27060  32578.8       40890
               TRT legacy_clip   1103         0         0          50       50          50
               TRT    typed_v1   1103  -6.85605  -3.93566  -0.0153551   1.3663     3.49328
               TST         raw   1103         0         0       20670  27538.8       30120
               TST legacy_clip   1103         0         0          50       50          50
               TST    typed_v1   1103  -3.79235  -3.22885  -0.0163934  1.22383     1.69399
              WASO         raw   1103         0         0        4770  16287.6       22770
              WASO legacy_clip   1103         0         0          50       50          50
              WASO    typed_v1   1103  -1.22137  -1.06855           0  2.92305     4.57252
               BSR         raw 983796         0         0           0        0         100
               BSR legacy_clip 983796         0         0           0        0          50
               BSR    typed_v1 983796         0         0           0        0           1
EMG_envelope_ratio         raw 983796         0         0    0.397764  2.41168 1.63317e+15
EMG_envelope_ratio legacy_clip 983796         0         0    0.397764  2.41168          50
EMG_envelope_ratio    typed_v1 983796   -1.5788   -1.5788 -0.00527125  4.18769      163.02
       N3_N1_ratio         raw   1103         0         0        1.24      9.9        51.5
       N3_N1_ratio legacy_clip   1103         0         0        1.24      9.9          50
       N3_N1_ratio    typed_v1   1103 -0.983376 -0.983376           0  1.90436     3.80515

## Validation

- test_feature_scaling.py: 11/11 passed.
- test_feature_scaling_fit_and_compat.py: 4/4 passed.
- CPU forward/backward finite gradients and train/inference logit agreement at 1e-6.
- Two real train caches also passed shared-transform equality and a finite CPU backward pass: X_seq=(2,999,483), X_ecg=(2,999,12), x_static=(2,196).
- Full offline transformation uses no labels; test/external y=-1 is not scored.

## Reproduction

conda run -n sleepfm_env python feat_input/feat_mody/build_feature_rules_v1.py
conda run -n sleepfm_env python feat_input/feat_mody/test_feature_scaling.py
conda run -n sleepfm_env python feat_input/feat_mody/test_feature_scaling_fit_and_compat.py
conda run -n sleepfm_env python feat_input/feat_mody/smoke_real_cache.py
conda run -n sleepfm_env python feat_input/feat_mody/offline_validate_typed_v1.py --cache-root npz_new --split-root split

## Training command (not executed)

LSTM_INPUT_PREPROCESSING=typed_v1 LSTM_CACHE_DIR=npz_new LSTM_SPLITS_DIR=split LSTM_MODEL_DIR=<new_model_dir> LSTM_SEED=7 conda run -n sleepfm_env python train_lstm.py

## Boundary of evidence

- Code proves shared deterministic behavior, strict state validation and explicit legacy compatibility.
- Cache measurement proves finite transformed outputs for this 1103-record cache.
- Historical provenance remains assumed because NPZ has no embedded feature names/version.
- Performance and generalization require controlled legacy_clip versus typed_v1 retraining.

Offline validation elapsed: 499.8 seconds.
