# Data2 low-cost baseline reuse audit

## Repository state before execution

- Branch: `official-submission`.
- HEAD: `9eb1b8dc93e2d5905ae544a462f2dce71b1e1fe1`.
- No checkout or reset was performed. Existing untracked extraction outputs and user files were left intact.
- The detached 1103 timegrid_v2 extraction process was confirmed active. Its incomplete NPZ output is excluded from this experiment.

## Reused implementation

- `pooled_logistic_v2.py`: frozen `COMPACT_INDICES`, `assemble_features`, and global EEG pooling definitions.
- `feat_input/feat_mody/run_p3_v2.py`: frozen elastic-net logistic configuration, fold-local linear imputation/scaling, model fitting, and feature-name generation.
- `feat_input/feat_mody/run_loso_1103.py`: eligible age-pair counting and metric bundle, which calls the official age-conditioned and age-weighted AUROC functions.
- `feat_input/feat_mody/run_p6_site_audit.py`: patient-level continuous median/IQR summary, stage/event mean summary, fold-local site-classifier preprocessing, and fixed site logistic configuration.
- `feature_scaling.py`: the reviewed 691-column rules and its exact value preparation, deterministic epoch sampling, and weighted-quantile helpers.

## Data2 adaptation

- A new thin runner, `feat_input2/run_data2_lowcost.py`, adapts the local `npz_data2/{site}/{record}.npz` layout and embedded timegrid_v2 metadata.
- The adapter fits only the P3-selected static and EEG columns needed by demo10, compact30, and global59. It calls the existing feature rules and scaling primitives and validates the selective result against the complete shared `FeatureScaler` on a smoke subset.
- The timegrid_v2 fields `stage_code_aligned`, `stage_valid`, and `eeg_channel_available` directly supply the existing P3 global59 builder. No sidecar, raw EDF, or EEG re-extraction is needed.
- Site-decodability calls the P6-A summary and fold-processing functions rather than defining a second classifier protocol.
- LOSO site membership is taken directly from the frozen three-site definition; patient IDs embedded in NPZ are retained and checked for cross-site leakage.

No second demo10, compact30, global59, official metric, or site-classifier feature definition was created.
