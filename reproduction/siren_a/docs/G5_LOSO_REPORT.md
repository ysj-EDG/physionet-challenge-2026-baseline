# G5 Branch A Rank Ensemble and LOSO Scoring Report

`G5_STATUS = SUBMITTED_FOR_REVIEW`

All results use the fixed equal-weight rank mean over 5 views × 3 model families × 15 seeds (225 models). No model, seed, or weight was selected from holdout performance.

| Holdout | Train site | Protocol | N | Positive | Negative | AC pairs | Models | AC-AUROC | AUROC | AUPRC |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| I0002 | S0001 | transductive | 319 | 52 | 267 | 1886 | 225 | 0.679745 | 0.715968 | 0.374518 |
| I0002 | S0001 | inductive | 319 | 52 | 267 | 1886 | 225 | 0.679215 | 0.722270 | 0.371968 |
| I0006 | S0001 | transductive | 1142 | 112 | 1030 | 14493 | 225 | 0.654040 | 0.723266 | 0.298842 |
| I0006 | S0001 | inductive | 1142 | 112 | 1030 | 14493 | 225 | 0.665080 | 0.734804 | 0.301191 |
| S0001 | I0006 | transductive | 5139 | 334 | 4805 | 182330 | 225 | 0.653392 | 0.724449 | 0.197909 |
| S0001 | I0006 | inductive | 5139 | 334 | 4805 | 182330 | 225 | 0.668639 | 0.743788 | 0.212919 |

| Protocol | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site AC | Small-site mean AC |
|---|---:|---:|---:|---:|---:|---:|
| transductive | 0.679745 | 0.654040 | 0.653392 | 0.662393 | 0.653392 | 0.666893 |
| inductive | 0.679215 | 0.665080 | 0.668639 | 0.670978 | 0.665080 | 0.672147 |

## Interpretation and scope

This is the first fixed SIREN-inspired Branch A baseline score and does not include Branch B. Each source score vector was ranked over its complete holdout cohort with average ties, then the 225 rank vectors were averaged with weight 1/225. Raw Rank256 inputs, models, and predictions were not recomputed.

The inductive-feature comparison means that its feature reference distribution came from the training site. Its final rank ensemble remains cohort-based and is not an end-to-end strictly inductive deployment protocol.

No holdout result was used for model selection, seed selection, sign reversal, calibration, thresholding, or weight optimization. Repeated future tuning against these LOSO results would introduce model-selection bias and must be disclosed separately.

`G5_PREDICTION_INPUTS = VERIFIED`

`G5_FIXED_RANK_ENSEMBLE = COMPLETE`

`G5_OFFICIAL_LOSO_SCORES = COMPLETE`

`G4_MODELS_REFIT = 0`

`G4_MODEL_REPLAY = 0`

`G1_G2_G3_G4_ARTIFACTS = UNCHANGED`

`HYPERPARAMETER_SEARCH = NOT_STARTED`

`ADDITIONAL_ABLATIONS = NOT_STARTED`
