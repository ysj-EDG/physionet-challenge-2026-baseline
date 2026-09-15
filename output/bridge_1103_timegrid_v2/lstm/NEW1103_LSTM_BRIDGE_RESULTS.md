# NEW1103 timegrid_v2 LSTM bridge results

## Frozen protocol

- Source commit before bridge changes: `d6bb0606b3e11b1694ce5d82e1b75cf0b37a912c`.
- Input is only the frozen P5 1103-record LOSO manifest and NEW `npz_1103_timegrid_v2` cache.
- All three bridge arms reuse P5's 2-layer LSTM, `legacy_clip`, seed 7, Adam 1e-3, batch size 8, gradient clip 1.0, and exactly six epochs.
- Across OLD/NEW comparison the model protocols remain frozen; within NEW, L0/L1/L2 differ only by the requested sampler/positive-loss weighting.
- Main results use raw logits and official Age-conditioned AUROC (gap=2); no holdout threshold, calibration, epoch selection, or tuning was used.

## Main comparison

| Protocol | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site AC |
|---|---:|---:|---:|---:|---:|
| OLD P5 balanced class sampler + empirical pos_weight | 0.646 | 0.474 | 0.505 | 0.541 | 0.474 |
| NEW-L0 global class sampler + empirical pos_weight | 0.729 | 0.537 | 0.600 | 0.622 | 0.537 |
| NEW-L1 global class sampler + pos_weight=1 | 0.708 | 0.545 | 0.531 | 0.595 | 0.531 |
| NEW-L2 site+class sampler + pos_weight=1 | 0.531 | 0.591 | 0.524 | 0.549 | 0.524 |

## Delta from P5

| Protocol | I0002 | I0006 | S0001 | Macro | Worst-site |
|---|---:|---:|---:|---:|---:|
| NEW-L0 global class sampler + empirical pos_weight | +0.083 | +0.063 | +0.095 | +0.081 | +0.063 |
| NEW-L1 global class sampler + pos_weight=1 | +0.062 | +0.071 | +0.026 | +0.053 | +0.057 |
| NEW-L2 site+class sampler + pos_weight=1 | -0.115 | +0.117 | +0.019 | +0.007 | +0.050 |

## Fold diagnostics

| Protocol | Holdout | n (+/-) | Eligible pairs | Train AC | Holdout AC | Age-weighted | AUROC | AUPRC | Train-holdout gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0 | I0002 | 54 (8/46) | 48 | 0.861 | 0.729 | 0.682 | 0.736 | 0.289 | +0.132 |
| L0 | I0006 | 192 (20/172) | 441 | 0.894 | 0.537 | 0.521 | 0.598 | 0.137 | +0.356 |
| L0 | S0001 | 857 (56/801) | 5124 | 0.832 | 0.600 | 0.610 | 0.582 | 0.125 | +0.232 |
| L1 | I0002 | 54 (8/46) | 48 | 0.663 | 0.708 | 0.661 | 0.666 | 0.409 | -0.046 |
| L1 | I0006 | 192 (20/172) | 441 | 0.780 | 0.545 | 0.527 | 0.595 | 0.210 | +0.235 |
| L1 | S0001 | 857 (56/801) | 5124 | 0.824 | 0.531 | 0.563 | 0.559 | 0.118 | +0.293 |
| L2 | I0002 | 54 (8/46) | 48 | 0.649 | 0.531 | 0.500 | 0.516 | 0.163 | +0.117 |
| L2 | I0006 | 192 (20/172) | 441 | 0.687 | 0.591 | 0.638 | 0.589 | 0.168 | +0.097 |
| L2 | S0001 | 857 (56/801) | 5124 | 0.901 | 0.524 | 0.542 | 0.585 | 0.080 | +0.377 |

## Predeclared NEW-protocol decisions

- B1 ordering `L0 < L1 <= L2`: **not confirmed**.
- B2 versus NEW-L0: **FAIL** (1/3 sites improve; Macro -0.073; Worst -0.014).
- Deltas from OLD P5 above describe the extractor bridge only and are not used to promote B1 or B2.
