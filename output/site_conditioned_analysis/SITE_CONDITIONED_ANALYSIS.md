# Site-conditioned analysis of frozen P3/P4 outer-CV logits

## Scope and provenance

- Analysis commit before this uncommitted analysis: `736c2dc9f2982c4699886624eb1971eab2f2adfc`.
- No model training, inference, NPZ generation, feature transformation, or checkpoint modification was performed.
- P3 inputs: `output/p3_v2/cv/seed{7,17,29}/fold{0,1,2}/P3_*_holdout.csv`.
- P4 inputs: `output/p4_legacy_lstm_cv/seed7_model/outer_seed*_fold*/*_outer_holdout_logits.csv`.
- Frozen fold manifest: `output/p3_v2/cv_fold_manifest.csv`, SHA256 `d3642ebc22b932ca8c17daba9d94cffbd8444580e80d64f3cd7f7afa83076426`.
- Combined SHA256 over the 63 input prediction CSV paths and contents: `77883cc8808e4cde6acae2a176274a10fa8cbe60fa2227c59fb549f4d564b034`.
- All prediction record IDs, labels, and raw ages matched the frozen holdout manifest exactly.
- The 733-record outer-CV population contains I0006 and S0001 only; I0002 has zero records and is therefore reported as N/A rather than imputed.

## Metric definition

AC-AUROC uses the official gap=2 pairwise comparison with tie credit 0.5. `same_site` adds equal-site filtering; `cross_site` adds unequal-site filtering. Per-site values restrict the patient pool to that site. Aggregation is fold metric -> mean of three folds per repeat -> mean of three repeat means. Nine-fold logits are never pooled for the primary result.

`Pos/neg` in the fold CSV is the full class count in the relevant patient pool. `participating_*` additionally reports unique patients that occur in at least one eligible age/site pair.

## Final hierarchical summary

| Model | Overall AC | Same-site AC | Cross-site AC | I0002 AC | I0006 AC | S0001 AC | Same-overall | Cross-same |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p4_legacy_best_val_ac | 0.701 | 0.715 | 0.671 | N/A | 0.543 | 0.726 | +0.015 | -0.044 |
| p4_legacy_fixed_epoch6 | 0.682 | 0.688 | 0.663 | N/A | 0.578 | 0.693 | +0.006 | -0.026 |
| p3_demo10 | 0.567 | 0.588 | 0.526 | N/A | 0.506 | 0.589 | +0.021 | -0.062 |
| p3_compact30 | 0.581 | 0.535 | 0.665 | N/A | 0.691 | 0.523 | -0.046 | +0.129 |
| p3_compact35 | 0.573 | 0.526 | 0.662 | N/A | 0.692 | 0.513 | -0.048 | +0.136 |
| p3_global59 | 0.590 | 0.562 | 0.630 | N/A | 0.632 | 0.555 | -0.029 | +0.069 |
| p3_stage155 | 0.494 | 0.466 | 0.546 | N/A | 0.551 | 0.460 | -0.028 | +0.080 |

## Repeat-level AC-AUROC for key comparisons

| Model | Condition | Seed 7 | Seed 17 | Seed 29 | Final | Repeat min-max | SD |
|---|---|---:|---:|---:|---:|---:|---:|
| p4_legacy_best_val_ac | overall | 0.735 | 0.687 | 0.681 | 0.701 | 0.681-0.735 | 0.030 |
| p4_legacy_best_val_ac | same_site | 0.753 | 0.693 | 0.701 | 0.715 | 0.693-0.753 | 0.032 |
| p4_legacy_best_val_ac | cross_site | 0.706 | 0.678 | 0.630 | 0.671 | 0.630-0.706 | 0.038 |
| p4_legacy_best_val_ac | site_I0002 | N/A | N/A | N/A | N/A | N/A | N/A |
| p4_legacy_best_val_ac | site_I0006 | 0.621 | 0.593 | 0.414 | 0.543 | 0.414-0.621 | 0.112 |
| p4_legacy_best_val_ac | site_S0001 | 0.765 | 0.697 | 0.716 | 0.726 | 0.697-0.765 | 0.035 |
| p4_legacy_fixed_epoch6 | overall | 0.700 | 0.663 | 0.684 | 0.682 | 0.663-0.700 | 0.019 |
| p4_legacy_fixed_epoch6 | same_site | 0.703 | 0.664 | 0.699 | 0.688 | 0.664-0.703 | 0.021 |
| p4_legacy_fixed_epoch6 | cross_site | 0.686 | 0.665 | 0.637 | 0.663 | 0.637-0.686 | 0.025 |
| p4_legacy_fixed_epoch6 | site_I0002 | N/A | N/A | N/A | N/A | N/A | N/A |
| p4_legacy_fixed_epoch6 | site_I0006 | 0.537 | 0.773 | 0.424 | 0.578 | 0.424-0.773 | 0.178 |
| p4_legacy_fixed_epoch6 | site_S0001 | 0.709 | 0.659 | 0.711 | 0.693 | 0.659-0.711 | 0.029 |
| p3_compact30 | overall | 0.544 | 0.562 | 0.637 | 0.581 | 0.544-0.637 | 0.049 |
| p3_compact30 | same_site | 0.503 | 0.509 | 0.594 | 0.535 | 0.503-0.594 | 0.051 |
| p3_compact30 | cross_site | 0.603 | 0.671 | 0.719 | 0.665 | 0.603-0.719 | 0.059 |
| p3_compact30 | site_I0002 | N/A | N/A | N/A | N/A | N/A | N/A |
| p3_compact30 | site_I0006 | 0.557 | 0.723 | 0.794 | 0.691 | 0.557-0.794 | 0.121 |
| p3_compact30 | site_S0001 | 0.492 | 0.495 | 0.582 | 0.523 | 0.492-0.582 | 0.051 |
| p3_global59 | overall | 0.537 | 0.575 | 0.658 | 0.590 | 0.537-0.658 | 0.062 |
| p3_global59 | same_site | 0.496 | 0.558 | 0.631 | 0.562 | 0.496-0.631 | 0.067 |
| p3_global59 | cross_site | 0.571 | 0.614 | 0.707 | 0.630 | 0.571-0.707 | 0.070 |
| p3_global59 | site_I0002 | N/A | N/A | N/A | N/A | N/A | N/A |
| p3_global59 | site_I0006 | 0.529 | 0.624 | 0.744 | 0.632 | 0.529-0.744 | 0.108 |
| p3_global59 | site_S0001 | 0.486 | 0.555 | 0.624 | 0.555 | 0.486-0.624 | 0.069 |

## Evidence volume

The table below gives the range across the nine frozen folds. Exact fold/site counts are in `site_composition.csv`.

| Quantity | Minimum per fold | Maximum per fold |
|---|---:|---:|
| Eligible same-site age pairs | 192 | 422 |
| Eligible cross-site age pairs | 72 | 202 |
| I0002 samples (positives) | 0 (0) | 0 (0) |
| I0002 eligible within-site pairs | 0 | 0 |
| I0006 samples (positives) | 39 (1) | 41 (7) |
| I0006 eligible within-site pairs | 7 | 30 |
| S0001 samples (positives) | 204 (8) | 205 (17) |
| S0001 eligible within-site pairs | 178 | 412 |

## Answers to the six questions

1. **Legacy same-site signal:** yes. Best-val legacy same-site AC is **0.715**, with repeat means 0.753/0.693/0.701; fixed-epoch6 is **0.688**. Both remain clearly above 0.5.
2. **Source of the legacy advantage:** it is stronger on same-site pairs, not cross-site pairs. Best-val legacy is **0.715** same-site versus **0.671** cross-site (cross-minus-same -0.044); restricting to same-site also raises AC by +0.015 over overall. Cross-site signal is still meaningful, but site separation does not explain the 0.70 headline result.
3. **Within-site consistency:** no evidence supports efficacy at all three sites. I0002 is absent. S0001 is strong and stable (0.726; repeat 0.697-0.765), while I0006 is only 0.543 and unstable (0.414-0.621). I0006 contributes just 7-30 eligible within-site pairs per fold; about 95% of eligible same-site pairs come from S0001, so the same-site legacy result is predominantly supported by S0001.
4. **P3 site dependence:** yes, especially compact30. compact30 is **0.535** same-site versus **0.665** cross-site (difference +0.129); global59 is **0.562** versus **0.630** (difference +0.069). Their overall AC is therefore more site-assisted than legacy's.
5. **Legacy advantage after site control:** legacy best-val minus global59 is **+0.154 AC** on same-site pairs, compared with approximately +0.110 in the original overall P3/P4 reports.
6. **Next-step implication:** prioritize LOSO/site-domain validation before further temporal/static branch ablation. The legacy advantage survives same-site control, so branch ablation remains scientifically useful, but current evidence is dominated by S0001, weak/unstable for I0006, and contains no I0002. A branch ablation on the same folds would explain the S0001-heavy signal without first establishing multicenter robustness.

## Interpretation

The central finding is that P4 legacy's 0.70 AC is not created by easier cross-site comparisons: its same-site AC is higher than both overall and cross-site AC. However, `same-site` is not synonymous with `multicenter-stable`: S0001 supplies about 95% of eligible same-site pairs, I0006 is underpowered and unstable, and I0002 is absent. Thus the model has credible within-hospital ranking signal, but the present CV cannot establish balanced cross-center generalization.
