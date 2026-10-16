# G5-D Fixed Leave-One-View-Out Analysis

`G5D_STATUS = SUBMITTED_FOR_REVIEW`

This prediction-layer diagnostic directly averages the frozen G5 per-model cohort ranks after removing all 45 columns trained on one view. It performs no training, model loading, inference, raw-prediction reload, reranking, weight search or input-feature change.

## Transductive

| Configuration | Models | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site AC | Small-site mean AC | ΔMacro vs full | ΔWorst vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 225 | 0.679745 | 0.654040 | 0.653392 | 0.662393 | 0.653392 | 0.666893 | 0.000000 | 0.000000 |
| drop_V1 | 180 | 0.684517 | 0.648727 | 0.652081 | 0.661775 | 0.648727 | 0.666622 | -0.000617 | -0.004665 |
| drop_V2 | 180 | 0.668081 | 0.648244 | 0.644617 | 0.653647 | 0.644617 | 0.658162 | -0.008745 | -0.008775 |
| drop_V3 | 180 | 0.672587 | 0.653005 | 0.651966 | 0.659186 | 0.651966 | 0.662796 | -0.003206 | -0.001426 |
| drop_V4 | 180 | 0.686108 | 0.659629 | 0.657237 | 0.667658 | 0.657237 | 0.672868 | 0.005265 | 0.003845 |
| drop_V5 | 180 | 0.681336 | 0.653281 | 0.642730 | 0.659116 | 0.642730 | 0.667309 | -0.003277 | -0.010662 |

## Inductive-feature

| Configuration | Models | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site AC | Small-site mean AC | ΔMacro vs full | ΔWorst vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full | 225 | 0.679215 | 0.665080 | 0.668639 | 0.670978 | 0.665080 | 0.672147 | 0.000000 | 0.000000 |
| drop_V1 | 180 | 0.681336 | 0.656938 | 0.667224 | 0.668499 | 0.656938 | 0.669137 | -0.002479 | -0.008142 |
| drop_V2 | 180 | 0.675504 | 0.656800 | 0.655353 | 0.662552 | 0.655353 | 0.666152 | -0.008426 | -0.009727 |
| drop_V3 | 180 | 0.679745 | 0.660043 | 0.674650 | 0.671480 | 0.660043 | 0.669894 | 0.000501 | -0.005037 |
| drop_V4 | 180 | 0.683457 | 0.660457 | 0.670674 | 0.671529 | 0.660457 | 0.671957 | 0.000551 | -0.004623 |
| drop_V5 | 180 | 0.682927 | 0.671980 | 0.652841 | 0.669249 | 0.652841 | 0.677453 | -0.001729 | -0.012239 |

## Interpretation

- In transductive, the largest Macro change was drop_V4 (+0.005265); the most negative was drop_V2 (-0.008745). Macro and Worst-site changes should be considered separately because a removal can redistribute performance across sites.
- In inductive, the largest Macro change was drop_V4 (+0.000551); the most negative was drop_V2 (-0.008426). Macro and Worst-site changes should be considered separately because a removal can redistribute performance across sites.

A positive delta only means the fixed 180-rank ensemble scored higher on these already observed LOSO cohorts; it does not prove that the removed view lacks physiological value or that removal will generalize. A negative delta indicates marginal complementarity within this fixed ensemble, not causal attribution.

`drop_V5` removes the complete V5 model block (CAISR20, HRV22, EMG48, Resp28 and Demo5), not CAISR alone. Every remaining view still contains Demo5, and the retained views each change weight from 20% to 25%. The EEG views overlap in information, so ensemble weight is not physiological contribution.

The inductive-feature route still uses test-cohort output ranking and is not end-to-end inductive deployment. This is post-hoc exploratory diagnosis after observing LOSO outcomes; no significance claim, best-subset selection or mainline replacement is made. The five-view primary baseline remains unchanged.

All 36 formal site/protocol/configuration rows, including AUROC and AUPRC, are in `reproduction/siren_a/audit/g5d_leave_one_view_scores.csv`.

`G5D_FIXED_REMOVALS = 5_PER_GROUP`
`G5D_NEW_SCORE_ROWS = 30`
`G5D_BASELINE_ROWS_REUSED = 6`
`G5D_TOTAL_SCORE_ROWS = 36`
`MODEL_REFIT = 0`
`MODEL_LOAD = 0`
`MODEL_REPLAY = 0`
`RAW_PREDICTION_RELOAD = 0`
`MODEL_RANK_RECOMPUTATION = 0`
`G1_G2_G3_G4_G5_G5C_ARTIFACTS = UNCHANGED`
`FIVE_VIEW_MAINLINE = UNCHANGED`
`NEW_FEATURE_DESIGN = NOT_STARTED`
`FEATURE_EXTRACTION_CHANGE = NOT_STARTED`
`HYPERPARAMETER_SEARCH = NOT_STARTED`
