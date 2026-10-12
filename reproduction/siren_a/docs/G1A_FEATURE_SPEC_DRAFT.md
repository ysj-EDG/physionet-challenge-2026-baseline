# G1-A Feature Specification Draft

This is a draft for review, not a frozen implementation specification. It is derived from the committed extractor and `feature_rules_v1.json`.

## View sizes

| View | Rows |
|---|---:|
| V1 Wake | 113 |
| V2 REM | 129 |
| V3 N2/N3/NREM | 349 |
| V4 All-stage | 113 |
| V5 Non-spectral | 123 |
| Total | 827 |

The draft has 807 unique feature IDs and repeats the same five pending Demo5 IDs in each view.

## Fixed source boundaries

- EEG54 uses the six committed stage groups, six leads, nine source features, and Mean/IQR.
- Spatial rows are the 36 explicitly listed SIREN-inspired candidates: 24 interhemispheric asymmetries and 12 anterior–posterior gradients. They are not claimed to reproduce an original-paper formula.
- Non-spectral rows are CAISR20, HRV11 Mean/IQR, EMG24 Mean/IQR, and Resp14 Mean/IQR. X_ecg index 11 (`hrv_circadian_cos`) is excluded.
- Demo5 has five repeated pending positions but no frozen encoding. Age is not a model input.

## Open status

`MIN_VALID_EPOCHS`, small-n IQR behavior, spatial support threshold, Demo5 coding, and any unresolved validity semantics remain `PENDING_G1B_REVIEW`.
