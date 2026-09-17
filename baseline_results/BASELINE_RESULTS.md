# Standalone DATA2 CAISR20 baseline

## 1. Objective

Freeze a reproducible low-capacity CAISR20-only elastic-net logistic-regression baseline for three-site DATA2 LOSO. The primary metric is official age-conditioned AUROC at gap 2.

## 2. Dataset and LOSO protocol

Directly scanned 6600 `timegrid_v2.0.0` NPZ files from `/database/home/gaohaojie/workspace/python-example-2026/npz_data2`. Each held-out site was invisible to transform, imputation, standardization, and model fitting. Age was used only for evaluation.

CAISR block missingness (finite all-zero algorithmic186 sentinel):

| Site | Missing | N | Rate |
|---|---|---|---|
| I0002 | 0 | 319 | 0.000% |
| I0006 | 0 | 1142 | 0.000% |
| S0001 | 0 | 5139 | 0.000% |

## 3. CAISR20 feature definition

1. `caisr_sleep_tst_sec` — x_static[11], sleep_architecture
2. `caisr_sleep_se` — x_static[12], sleep_architecture
3. `caisr_sleep_sol_sec` — x_static[13], sleep_architecture
4. `caisr_sleep_rem_latency_sec` — x_static[14], sleep_architecture
5. `caisr_sleep_waso_sec` — x_static[16], sleep_architecture
6. `caisr_sleep_n1_pct` — x_static[17], sleep_architecture
7. `caisr_sleep_n2_pct` — x_static[18], sleep_architecture
8. `caisr_sleep_n3_pct` — x_static[19], sleep_architecture
9. `caisr_sleep_rem_pct` — x_static[20], sleep_architecture
10. `caisr_sleep_transition_rate` — x_static[32], fragmentation_uncertainty
11. `caisr_sleep_short_bout_ratio` — x_static[34], fragmentation_uncertainty
12. `caisr_sleep_mean_stage_entropy` — x_static[87], fragmentation_uncertainty
13. `caisr_arousal_arousal_index` — x_static[96], arousal
14. `caisr_arousal_arousal_burden_ratio` — x_static[97], arousal
15. `caisr_arousal_mean_arousal_duration` — x_static[98], arousal
16. `caisr_respiratory_respiratory_event_index` — x_static[126], respiratory
17. `caisr_respiratory_respiratory_burden_ratio` — x_static[127], respiratory
18. `caisr_respiratory_mean_resp_duration` — x_static[141], respiratory
19. `caisr_limb_limb_movement_index` — x_static[168], limb
20. `caisr_limb_plmi` — x_static[173], limb

All inputs are automated CAISR algorithmic annotation summaries; demographics are excluded.

## 4. Preprocessing

For each outer fold and column, typed_v1 rules were applied and robust center/scale parameters were fitted only on the two training sites. The all-zero algorithmic186 sentinel was invalid. P3-equivalent median imputation and population mean/std standardization were then fitted only on training data. No cache or transformed NPZ was created.

## 5. Classifier

Frozen elastic-net logistic regression: `{"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.4, "C": 0.03, "class_weight": "balanced", "fit_intercept": true, "max_iter": 100000, "tol": 0.0001, "random_state": 7}`. No tuning, calibration, ranking loss, or threshold selection.

## 6. Primary results

| Site | N | Positive | Negative | AC-AUROC | AUROC | AUPRC |
|---|---|---|---|---|---|---|
| I0002 | 319 | 52 | 267 | 0.746023 | 0.764117 | 0.512930 |
| I0006 | 1142 | 112 | 1030 | 0.677706 | 0.731493 | 0.202933 |
| S0001 | 5139 | 334 | 4805 | 0.616662 | 0.699122 | 0.131674 |

- Macro AC: **0.680131**
- Worst-site AC: **0.616662**

## 7. Historical regression validation

Regression against commit `79adc97e180e9d38f3743fbf8d10f18e6b5b11cc`: **PASS**. Maximum aligned score difference: 4.44e-16; all metric differences were required to be <=1e-12.

## 8. Solver seed stability

| Scope | Mean AC | SD | Min | Max |
|---|---|---|---|---|
| I0002 | 0.746023 | 0 | 0.746023 | 0.746023 |
| I0006 | 0.677624 | 5.77286e-05 | 0.677568 | 0.677706 |
| S0001 | 0.616677 | 1.41964e-05 | 0.616662 | 0.616695 |
| Macro | 0.680108 | 1.8039e-05 | 0.680086 | 0.680131 |
| Worst | 0.616677 | 1.41964e-05 | 0.616662 | 0.616695 |

The canonical paper point estimate remains seed 7; the five-seed mean is not substituted for it.

## 9. Bootstrap uncertainty

| Scope | Point | Bootstrap mean | SD | 95% CI | Valid |
|---|---|---|---|---|---|
| I0002 | 0.746023 | 0.746196 | 0.045989 | [0.651223, 0.831530] | 1000 |
| I0006 | 0.677706 | 0.676230 | 0.029670 | [0.616852, 0.733406] | 1000 |
| S0001 | 0.616662 | 0.616356 | 0.018204 | [0.582224, 0.651874] | 1000 |
| Macro | 0.680131 | 0.679594 | 0.019131 | [0.642856, 0.716512] | 1000 |
| Worst | 0.616662 | 0.615628 | 0.018188 | [0.581679, 0.649611] | 1000 |

Patient-level bootstrap sampled positives and negatives separately within site (1000 replicates; seed 20260917).

## 10. Interpretability

Most sign-stable standardized coefficients:

| Feature | Group | Median coef | Median abs(coef) | Sign consistency | Zero fraction |
|---|---|---|---|---|---|
| caisr_sleep_mean_stage_entropy | fragmentation_uncertainty | 0.5559 | 0.5559 | 1.000 | 0.000 |
| caisr_limb_limb_movement_index | limb | 0.3483 | 0.3483 | 1.000 | 0.000 |
| caisr_sleep_n1_pct | sleep_architecture | -0.2928 | 0.2928 | 1.000 | 0.000 |
| caisr_sleep_waso_sec | sleep_architecture | 0.2496 | 0.2496 | 1.000 | 0.000 |
| caisr_respiratory_mean_resp_duration | respiratory | 0.2322 | 0.2322 | 1.000 | 0.000 |
| caisr_arousal_arousal_burden_ratio | arousal | -0.0730 | 0.0730 | 1.000 | 0.000 |
| caisr_sleep_se | sleep_architecture | -0.2328 | 0.2328 | 0.667 | 0.333 |
| caisr_arousal_arousal_index | arousal | -0.2079 | 0.2079 | 0.667 | 0.333 |

Leave-one-group-out results (delta is ablation minus full CAISR20):

| Removed group | Remaining dim | Macro AC | Worst AC | Delta Macro | Delta Worst |
|---|---|---|---|---|---|
| sleep_architecture | 11 | 0.679957 | 0.608079 | -0.000174 | -0.008583 |
| fragmentation_uncertainty | 17 | 0.639605 | 0.578051 | -0.040525 | -0.038611 |
| arousal | 17 | 0.670098 | 0.626820 | -0.010033 | +0.010157 |
| respiratory | 17 | 0.687472 | 0.633982 | +0.007341 | +0.017320 |
| limb | 18 | 0.650250 | 0.594762 | -0.029881 | -0.021900 |

Coefficients describe associations in fold-standardized linear models, not causal importance. Correlated features can share or substitute coefficients. Site distributions and missingness are reported descriptively and were not used for feature selection.

## 11. Limitations

CAISR20 is derived from automated CAISR annotations, not human expert annotations. `mean_stage_entropy` uses CAISR stage-posterior uncertainty. Human annotations exist in the broader training data, but this baseline does not use them. A matched human-vs-CAISR annotation-source comparison remains a separate future experiment and no result is implied here.

No LSTM, temporal model, demographics fusion, ranking objective, human-annotation baseline, SHAP, feature search, or extractor change was performed.
