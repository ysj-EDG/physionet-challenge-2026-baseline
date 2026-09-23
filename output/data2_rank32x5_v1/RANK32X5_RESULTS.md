# DATA2 compact30 Rank32x5 results

Frozen baseline Macro/Worst: 0.693989808495 / 0.653819996709

| Site | Rank AC mean ± SD | Baseline AC | Delta |
|---|---:|---:|---:|
| I0002 | 0.745069 ± 0.004080 | 0.747614 | -0.002545 |
| I0006 | 0.682219 ± 0.011305 | 0.680535 | +0.001684 |
| S0001 | 0.614089 ± 0.001791 | 0.653820 | -0.039731 |

Macro AC: 0.680459 ± 0.002500; delta -0.013531
Worst AC: 0.614089 ± 0.001791; delta -0.039731
Pre-specified classification: **Negative**.

Ranker: l2 LogisticRegression, C=0.03, no intercept; five age-windowed within-site subsamples; symmetric differences; equal total site weights; raw-score mean ensemble.
