# P6-B: site-balanced LSTM LOSO results

## Frozen protocol

- Source commit before P6 changes: `69e2aecef497f879b6a9e9c1953474191b8e6d91`.
- Input is only the frozen P5 1103-record LOSO manifest and legacy `npz_new` cache.
- Both P6 arms reuse P5's 2-layer LSTM, `legacy_clip`, seed 7, Adam 1e-3, batch size 8, gradient clip 1.0, and exactly six epochs.
- The only changes are the requested sampler/positive-loss weighting. Holdout sites were constructed and evaluated only after epoch-6 checkpoints were frozen.
- Main results use raw logits and official Age-conditioned AUROC (gap=2); no holdout threshold, calibration, epoch selection, or tuning was used.

## Main comparison

| Protocol | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site AC |
|---|---:|---:|---:|---:|---:|
| P5 balanced class sampler + empirical pos_weight | 0.646 | 0.474 | 0.505 | 0.541 | 0.474 |
| P6-B1 global class sampler + pos_weight=1 | 0.667 | 0.532 | 0.563 | 0.587 | 0.532 |
| P6-B2 site+class sampler + pos_weight=1 | 0.646 | 0.562 | 0.617 | 0.608 | 0.562 |

## Delta from P5

| Protocol | I0002 | I0006 | S0001 | Macro | Worst-site |
|---|---:|---:|---:|---:|---:|
| P6-B1 global class sampler + pos_weight=1 | +0.021 | +0.058 | +0.059 | +0.046 | +0.058 |
| P6-B2 site+class sampler + pos_weight=1 | +0.000 | +0.088 | +0.112 | +0.067 | +0.088 |

## Fold diagnostics

| Protocol | Holdout | n (+/-) | Eligible pairs | Train AC | Holdout AC | Age-weighted | AUROC | AUPRC | Train-holdout gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1_global_class_pos1 | I0002 | 54 (8/46) | 48 | 0.727 | 0.667 | 0.661 | 0.677 | 0.258 | +0.060 |
| B1_global_class_pos1 | I0006 | 192 (20/172) | 441 | 0.527 | 0.532 | 0.532 | 0.544 | 0.126 | -0.004 |
| B1_global_class_pos1 | S0001 | 857 (56/801) | 5124 | 0.832 | 0.563 | 0.546 | 0.557 | 0.091 | +0.269 |
| B2_site_class_pos1 | I0002 | 54 (8/46) | 48 | 0.681 | 0.646 | 0.643 | 0.576 | 0.249 | +0.035 |
| B2_site_class_pos1 | I0006 | 192 (20/172) | 441 | 0.794 | 0.562 | 0.561 | 0.607 | 0.170 | +0.231 |
| B2_site_class_pos1 | S0001 | 857 (56/801) | 5124 | 0.729 | 0.617 | 0.651 | 0.639 | 0.141 | +0.113 |

## Interpretation

- `B1_global_class_pos1` changed macro AC by +0.046 and worst-site AC by +0.058; 3/3 sites improved. Under the predeclared criteria this is **supportive** of carrying the strategy forward.
- `B2_site_class_pos1` changed macro AC by +0.067 and worst-site AC by +0.088; 2/3 sites improved. Under the predeclared criteria this is **supportive** of carrying the strategy forward.
