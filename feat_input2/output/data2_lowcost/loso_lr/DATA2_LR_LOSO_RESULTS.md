# Data2 LR LOSO results

All models use the frozen P3 definitions and LR configuration. Typed selected-column scaling and P3 linear imputation/scaling were fitted only on the two outer training sites. No threshold, calibration, model selection, or holdout tuning was performed.

| Model | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst AC | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| demo10 | 0.605 | 0.490 | 0.577 | 0.557 | 0.490 | 0.753 | 0.275 |
| compact30 | 0.748 | 0.681 | 0.654 | 0.694 | 0.654 | 0.793 | 0.337 |
| global59 | 0.748 | 0.686 | 0.614 | 0.683 | 0.614 | 0.792 | 0.364 |

## Descriptive old1103 comparison

| Model | Old I0002/I0006/S0001 | Old Macro/Worst | Data2 Macro/Worst |
|---|---:|---:|---:|
| demo10 | 0.542 / 0.382 / 0.513 | 0.479 / 0.382 | 0.557 / 0.490 |
| compact30 | 0.583 / 0.533 / 0.522 | 0.546 / 0.522 | 0.694 / 0.654 |
| global59 | 0.646 / 0.494 / 0.491 | 0.544 / 0.491 | 0.683 / 0.614 |

Old1103 to data2 simultaneously changes extractor, sample size, and site composition. These differences are descriptive and cannot be attributed until the new1103 bridge is complete.
