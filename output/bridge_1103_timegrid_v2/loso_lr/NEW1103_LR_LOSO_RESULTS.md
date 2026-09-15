# NEW1103 timegrid_v2 LR LOSO results

All models use the frozen P3 definitions and LR configuration. Typed selected-column scaling and P3 linear imputation/scaling were fitted only on the two outer training sites. No threshold, calibration, model selection, or holdout tuning was performed.

| Model | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst AC | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| demo10 | 0.542 | 0.382 | 0.513 | 0.479 | 0.382 | 0.707 | 0.233 |
| compact30 | 0.625 | 0.531 | 0.526 | 0.560 | 0.526 | 0.734 | 0.260 |
| global59 | 0.646 | 0.494 | 0.491 | 0.544 | 0.491 | 0.729 | 0.270 |

## Descriptive old1103 comparison

| Model | Old I0002/I0006/S0001 | Old Macro/Worst | NEW1103 timegrid_v2 Macro/Worst |
|---|---:|---:|---:|
| demo10 | 0.542 / 0.382 / 0.513 | 0.479 / 0.382 | 0.479 / 0.382 |
| compact30 | 0.583 / 0.533 / 0.522 | 0.546 / 0.522 | 0.560 / 0.526 |
| global59 | 0.646 / 0.494 / 0.491 | 0.544 / 0.491 | 0.544 / 0.491 |

OLD1103 and NEW1103 use the same patients, labels, sites, folds, feature definitions, and LR protocol. The principal systematic change is extraction version; the observed differences remain descriptive rather than strict causal proof.

Convergence warning retained without hyperparameter changes or reruns: demo10/I0002 reached max_iter=100000 before convergence.
