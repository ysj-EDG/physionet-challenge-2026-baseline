# G5-C View × Model Component Decomposition

`G5C_STATUS = SUBMITTED_FOR_REVIEW`

This post-hoc diagnostic averages only the frozen per-model cohort-rank columns saved by G5. It performs no training, inference, raw-prediction reload, reranking, weight search, or component selection.

## Protocol-level Macro AC-AUROC matrices

### Transductive Macro AC-AUROC

| View | LR | HGB | LambdaRank | Three-model fusion |
|---|---:|---:|---:|---:|
| V1 | 0.641824 | 0.606873 | 0.604586 | 0.629218 |
| V2 | 0.630346 | 0.637622 | 0.633673 | 0.644282 |
| V3 | 0.616335 | 0.638651 | 0.629826 | 0.638150 |
| V4 | 0.639983 | 0.622802 | 0.613755 | 0.634842 |
| V5 | 0.580339 | 0.605616 | 0.604415 | 0.606453 |
| Five-view fusion | 0.664048 | 0.659301 | 0.649976 | 0.662393 |

### Inductive Macro AC-AUROC

| View | LR | HGB | LambdaRank | Three-model fusion |
|---|---:|---:|---:|---:|
| V1 | 0.652059 | 0.626790 | 0.615828 | 0.642454 |
| V2 | 0.619914 | 0.653312 | 0.640163 | 0.650150 |
| V3 | 0.617591 | 0.652100 | 0.627217 | 0.641791 |
| V4 | 0.649228 | 0.635628 | 0.619589 | 0.646062 |
| V5 | 0.583503 | 0.598691 | 0.593219 | 0.606666 |
| Five-view fusion | 0.672476 | 0.669200 | 0.648197 | 0.670978 |

## Holdout-level AC-AUROC matrices

#### I0002 / transductive

| View | LR | HGB | LambdaRank | Three-model fusion |
|---|---:|---:|---:|---:|
| V1 | 0.675504 | 0.628314 | 0.613998 | 0.653765 |
| V2 | 0.682131 | 0.673913 | 0.687434 | 0.692736 |
| V3 | 0.628844 | 0.678685 | 0.667285 | 0.673913 |
| V4 | 0.657476 | 0.637858 | 0.639183 | 0.656946 |
| V5 | 0.590668 | 0.617709 | 0.608431 | 0.620891 |
| Five-view fusion | 0.683457 | 0.670201 | 0.664899 | 0.679745 |

#### I0006 / transductive

| View | LR | HGB | LambdaRank | Three-model fusion |
|---|---:|---:|---:|---:|
| V1 | 0.650624 | 0.582178 | 0.594011 | 0.621679 |
| V2 | 0.643518 | 0.614642 | 0.611123 | 0.631443 |
| V3 | 0.638929 | 0.614435 | 0.610364 | 0.626578 |
| V4 | 0.658456 | 0.602739 | 0.600773 | 0.626992 |
| V5 | 0.585800 | 0.611847 | 0.608121 | 0.609536 |
| Five-view fusion | 0.686814 | 0.639412 | 0.636721 | 0.654040 |

#### S0001 / transductive

| View | LR | HGB | LambdaRank | Three-model fusion |
|---|---:|---:|---:|---:|
| V1 | 0.599345 | 0.610127 | 0.605748 | 0.612209 |
| V2 | 0.565390 | 0.624310 | 0.602463 | 0.608668 |
| V3 | 0.581232 | 0.622832 | 0.611830 | 0.613958 |
| V4 | 0.604017 | 0.627809 | 0.601308 | 0.620586 |
| V5 | 0.564548 | 0.587292 | 0.596693 | 0.588932 |
| Five-view fusion | 0.621872 | 0.668288 | 0.648308 | 0.653392 |

#### I0002 / inductive

| View | LR | HGB | LambdaRank | Three-model fusion |
|---|---:|---:|---:|---:|
| V1 | 0.669141 | 0.665960 | 0.647932 | 0.669141 |
| V2 | 0.662778 | 0.666225 | 0.678950 | 0.691145 |
| V3 | 0.632556 | 0.693531 | 0.673118 | 0.674973 |
| V4 | 0.649523 | 0.640509 | 0.639449 | 0.651644 |
| V5 | 0.591729 | 0.618240 | 0.602863 | 0.622481 |
| Five-view fusion | 0.684517 | 0.676034 | 0.664899 | 0.679215 |

#### I0006 / inductive

| View | LR | HGB | LambdaRank | Three-model fusion |
|---|---:|---:|---:|---:|
| V1 | 0.664390 | 0.589250 | 0.612192 | 0.637756 |
| V2 | 0.600980 | 0.641172 | 0.618574 | 0.624612 |
| V3 | 0.624577 | 0.638515 | 0.616194 | 0.638446 |
| V4 | 0.668426 | 0.628579 | 0.627199 | 0.655834 |
| V5 | 0.568481 | 0.578486 | 0.592182 | 0.596150 |
| Five-view fusion | 0.677154 | 0.645174 | 0.640033 | 0.665080 |

#### S0001 / inductive

| View | LR | HGB | LambdaRank | Three-model fusion |
|---|---:|---:|---:|---:|
| V1 | 0.622646 | 0.625160 | 0.587361 | 0.620466 |
| V2 | 0.595985 | 0.652539 | 0.622964 | 0.634693 |
| V3 | 0.595640 | 0.624253 | 0.592338 | 0.611954 |
| V4 | 0.629735 | 0.637797 | 0.592119 | 0.630708 |
| V5 | 0.590298 | 0.599347 | 0.584610 | 0.601366 |
| Five-view fusion | 0.655756 | 0.686393 | 0.639659 | 0.668639 |

## View fusions

| Protocol | Component | I0002 | I0006 | S0001 | Macro | Worst | Small-site mean | ΔMacro vs full | ΔWorst vs full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transductive | V1_all | 0.653765 | 0.621679 | 0.612209 | 0.629218 | 0.612209 | 0.637722 | -0.033175 | -0.041184 |
| transductive | V2_all | 0.692736 | 0.631443 | 0.608668 | 0.644282 | 0.608668 | 0.662089 | -0.018110 | -0.044724 |
| transductive | V3_all | 0.673913 | 0.626578 | 0.613958 | 0.638150 | 0.613958 | 0.650246 | -0.024243 | -0.039434 |
| transductive | V4_all | 0.656946 | 0.626992 | 0.620586 | 0.634842 | 0.620586 | 0.641969 | -0.027551 | -0.032806 |
| transductive | V5_all | 0.620891 | 0.609536 | 0.588932 | 0.606453 | 0.588932 | 0.615213 | -0.055940 | -0.064460 |
| inductive | V1_all | 0.669141 | 0.637756 | 0.620466 | 0.642454 | 0.620466 | 0.653449 | -0.028524 | -0.044614 |
| inductive | V2_all | 0.691145 | 0.624612 | 0.634693 | 0.650150 | 0.624612 | 0.657879 | -0.020828 | -0.040468 |
| inductive | V3_all | 0.674973 | 0.638446 | 0.611954 | 0.641791 | 0.611954 | 0.656710 | -0.029187 | -0.053126 |
| inductive | V4_all | 0.651644 | 0.655834 | 0.630708 | 0.646062 | 0.630708 | 0.653739 | -0.024916 | -0.034372 |
| inductive | V5_all | 0.622481 | 0.596150 | 0.601366 | 0.606666 | 0.596150 | 0.609316 | -0.064312 | -0.068930 |

## Model-family fusions

| Protocol | Component | I0002 | I0006 | S0001 | Macro | Worst | Small-site mean | ΔMacro vs full | ΔWorst vs full |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| transductive | all_lr | 0.683457 | 0.686814 | 0.621872 | 0.664048 | 0.621872 | 0.685136 | 0.001655 | -0.031520 |
| transductive | all_hgb | 0.670201 | 0.639412 | 0.668288 | 0.659301 | 0.639412 | 0.654807 | -0.003092 | -0.013980 |
| transductive | all_lambdarank | 0.664899 | 0.636721 | 0.648308 | 0.649976 | 0.636721 | 0.650810 | -0.012416 | -0.016671 |
| inductive | all_lr | 0.684517 | 0.677154 | 0.655756 | 0.672476 | 0.655756 | 0.680836 | 0.001498 | -0.009324 |
| inductive | all_hgb | 0.676034 | 0.645174 | 0.686393 | 0.669200 | 0.645174 | 0.660604 | -0.001778 | -0.019906 |
| inductive | all_lambdarank | 0.664899 | 0.640033 | 0.639659 | 0.648197 | 0.639659 | 0.652466 | -0.022781 | -0.025421 |

## Protocol differences

| Component | Inductive Macro − Transductive Macro | Inductive Worst − Transductive Worst |
|---|---:|---:|
| V1_lr | 0.010235 | 0.023301 |
| V1_hgb | 0.019917 | 0.007072 |
| V1_lambdarank | 0.011243 | -0.006650 |
| V2_lr | -0.010432 | 0.030596 |
| V2_hgb | 0.015690 | 0.026530 |
| V2_lambdarank | 0.006490 | 0.016112 |
| V3_lr | 0.001256 | 0.014408 |
| V3_hgb | 0.013449 | 0.009818 |
| V3_lambdarank | -0.002610 | -0.018026 |
| V4_lr | 0.009245 | 0.025717 |
| V4_hgb | 0.012826 | 0.025840 |
| V4_lambdarank | 0.005834 | -0.008654 |
| V5_lr | 0.003164 | 0.003934 |
| V5_hgb | -0.006925 | -0.008806 |
| V5_lambdarank | -0.011196 | -0.012082 |
| V1_all | 0.013237 | 0.008257 |
| V2_all | 0.005868 | 0.015944 |
| V3_all | 0.003641 | -0.002005 |
| V4_all | 0.011220 | 0.010122 |
| V5_all | 0.000213 | 0.007218 |
| all_lr | 0.008428 | 0.033884 |
| all_hgb | 0.009899 | 0.005761 |
| all_lambdarank | -0.001779 | 0.002938 |
| all_all | 0.008586 | 0.011687 |

## Observations

- V5 was not the strongest view in this fixed decomposition. Its three-model Macro AC was 0.606453 (transductive) and 0.606666 (inductive), below every EEG view. V2 was the strongest single-view fusion in both protocols: 0.644282 and 0.650150. Thus the EEG views retain substantial ranking ability in these predictions.
- Across all five views, LR had the highest model-family Macro AC in both protocols: 0.664048 and 0.672476. HGB was strongest on S0001 (0.668288 and 0.686393), while LR was stronger on the two smaller sites. LambdaRank had the lowest five-view family Macro in both protocols.
- The LR-only five-view fusion exceeded the full 225-model Macro by 0.001655 (transductive) and 0.001498 (inductive), but its Worst-site AC was lower by 0.031520 and 0.009324. No single-view three-model fusion exceeded the corresponding full Macro. These post-hoc differences do not establish that removing components would generalize.
- The full inductive-feature ensemble improved Macro AC over transductive by 0.008586. The largest positive cell-level Macro differences were V1-HGB (+0.019917), V2-HGB (+0.015690), V3-HGB (+0.013449), and V4-HGB (+0.012826). V5 fusion changed by only +0.000213; V5-HGB and V5-LambdaRank decreased by 0.006925 and 0.011196.
- A sensible next step, if separately approved and frozen, would be a controlled confirmation of model-family balance and protocol sensitivity. These observed LOSO results must not themselves be used to silently redefine the primary G5 ensemble.

## Interpretation boundaries

These are prediction-layer component diagnostics after observing LOSO outcomes, not causal input-feature contributions or a new selected primary model. A component exceeding the full ensemble does not establish that removing other components will generalize.

V5 combines CAISR20, HRV22, EMG48, Resp28 and Demo5; it is not a pure CAISR view. Every view contains Demo5, EEG views overlap in information, and equal model weights are not physiological contribution weights. No statistical-significance claim is made.

Any follow-up prompted by these observations must be a separately frozen controlled experiment; this gate does not alter the G5 baseline.

`G5C_FIXED_COMPONENTS = 24_PER_GROUP`

`G5C_SCORE_ROWS = 144`

`G5C_FULL_BASELINE_REPRODUCTION = PASS`

`MODEL_REFIT = 0`

`MODEL_REPLAY = 0`

`RAW_PREDICTION_RELOAD = 0`

`MODEL_RANK_RECOMPUTATION = 0`

`G1_G2_G3_G4_G5_ARTIFACTS = UNCHANGED`

`WEIGHT_SEARCH = NOT_STARTED`

`INPUT_FEATURE_ABLATION = NOT_STARTED`
