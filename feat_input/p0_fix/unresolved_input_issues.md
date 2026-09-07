# Unresolved input issues

- Historical NPZ files have dimensions but no embedded feature names or extractor version; current-order compatibility is an explicit assumption.
- Upstream extraction already converted some missing or failed measurements to zero. Except for approved all-11 HRV and all-186 CAISR sentinels, those origins cannot be recovered.
- Extreme EMG envelope ratios can reflect denominator or quality failures. typed_v1 makes them finite with log1p but does not certify physiological validity.
- All-zero StageEvent and modality blocks remain ambiguous and are not used to delete EEG records.
- Hard-range violations are counted and treated as invalid; they are not silently clipped.
- No standardized z clipping is enabled. Remaining abs(z)>50 values occur in three EMG envelope ratios (1086 observations total), one arousal interval and one signed limb-index difference; these remain column-level review items.
- typed_v1 combines scaling, BMI missing handling and legacy sentinels; individual effects require later ablations.
- No training, scoring, label recovery, feature re-extraction or architecture change was performed.
