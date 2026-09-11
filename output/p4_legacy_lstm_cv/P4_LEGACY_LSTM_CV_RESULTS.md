# P4 legacy LSTM repeated outer-CV results

## Result

The experiment completed all 9 P3 outer folds. The primary raw-ranking result for `best_val_ac` is **AC-AUROC 0.701**, AUROC 0.719, and AUPRC 0.191. These are hierarchical repeated-CV means, not metrics computed after pooling nine-fold logits.

## Frozen setup and checks

- Base Git commit: `498380f1bee20afdbb78c55ca076262e6cd0a182`.
- Fold manifest SHA256: `d3642ebc22b932ca8c17daba9d94cffbd8444580e80d64f3cd7f7afa83076426`; outer seeds 7/17/29, model seed 7 for all folds.
- Input preprocessing: `legacy_clip`; no typed scaling, feature mask, sidecar, pooled EEG, data2, test, or external data.
- Fixed model-selection set: 158 records (12 positive, 146 negative); BDSPPatientID overlap with the 733 base records: 0.
- Each repeat's three holdouts are disjoint and cover exactly 733 records; every fold has zero train/holdout patient overlap.
- Both checkpoint prediction lists match their frozen fold manifests exactly; no duplicate, missing, or non-finite output was found.
- Runtime: Python 3.10.12, PyTorch 2.5.1+cu121, CUDA 12.1, NVIDIA H100 PCIe (`eeg_env`).
- Outer holdout loaders were constructed only after both complete checkpoints were frozen. Holdout results did not affect training, scheduling, early stopping, calibration, threshold selection, or checkpoint selection.

## Nine-fold results: best validation AC checkpoint

| Outer seed | Fold | Best/stop epoch | Holdout n (+/-) | Eligible age pairs | Val AC/AUROC | Train AC/AUROC/AUPRC | Holdout AC/AUROC/AUPRC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 12/27 | 244 (15/229) | 410 | 0.677/0.660 | 0.924/0.937/0.505 | 0.795/0.823/0.217 |
| 7 | 1 | 4/19 | 245 (19/226) | 377 | 0.634/0.623 | 0.874/0.876/0.397 | 0.679/0.666/0.235 |
| 7 | 2 | 2/17 | 244 (19/225) | 569 | 0.720/0.710 | 0.639/0.693/0.264 | 0.731/0.718/0.166 |
| 17 | 0 | 3/18 | 244 (15/229) | 366 | 0.763/0.713 | 0.813/0.833/0.289 | 0.727/0.744/0.231 |
| 17 | 1 | 4/19 | 245 (22/223) | 554 | 0.694/0.785 | 0.875/0.888/0.373 | 0.688/0.655/0.195 |
| 17 | 2 | 23/38 | 244 (16/228) | 457 | 0.741/0.731 | 0.994/0.994/0.930 | 0.646/0.710/0.149 |
| 29 | 0 | 10/25 | 244 (10/234) | 264 | 0.685/0.724 | 0.877/0.884/0.410 | 0.589/0.635/0.098 |
| 29 | 1 | 9/24 | 245 (22/223) | 561 | 0.784/0.781 | 0.912/0.922/0.438 | 0.765/0.762/0.220 |
| 29 | 2 | 13/28 | 244 (21/223) | 543 | 0.573/0.575 | 0.972/0.975/0.718 | 0.689/0.754/0.206 |

## Nine-fold results: fixed epoch 6 checkpoint

| Outer seed | Fold | Trajectory best/stop epoch | Holdout n (+/-) | Eligible age pairs | Epoch-6 val AC/AUROC | Train AC/AUROC/AUPRC | Holdout AC/AUROC/AUPRC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0 | 12/27 | 244 (15/229) | 410 | 0.522/0.569 | 0.752/0.820/0.273 | 0.817/0.810/0.353 |
| 7 | 1 | 4/19 | 245 (19/226) | 377 | 0.470/0.542 | 0.921/0.921/0.500 | 0.621/0.652/0.225 |
| 7 | 2 | 2/17 | 244 (19/225) | 569 | 0.603/0.672 | 0.841/0.862/0.372 | 0.663/0.691/0.154 |
| 17 | 0 | 3/18 | 244 (15/229) | 366 | 0.716/0.687 | 0.851/0.867/0.295 | 0.751/0.763/0.203 |
| 17 | 1 | 4/19 | 245 (22/223) | 554 | 0.582/0.644 | 0.792/0.826/0.329 | 0.552/0.535/0.103 |
| 17 | 2 | 23/38 | 244 (16/228) | 457 | 0.513/0.565 | 0.784/0.813/0.368 | 0.685/0.695/0.146 |
| 29 | 0 | 10/25 | 244 (10/234) | 264 | 0.612/0.684 | 0.805/0.832/0.348 | 0.640/0.638/0.085 |
| 29 | 1 | 9/24 | 245 (22/223) | 561 | 0.759/0.756 | 0.873/0.876/0.335 | 0.711/0.705/0.173 |
| 29 | 2 | 13/28 | 244 (21/223) | 543 | 0.487/0.517 | 0.904/0.915/0.499 | 0.702/0.746/0.188 |

## Hierarchical aggregation

| Checkpoint | Repeat seed | Holdout AC | AUROC | AUPRC | Train AC | Train-holdout AC gap |
|---|---:|---:|---:|---:|---:|---:|
| best_val_ac | 7 | 0.735 | 0.736 | 0.206 | 0.812 | 0.077 |
| best_val_ac | 17 | 0.687 | 0.703 | 0.192 | 0.894 | 0.207 |
| best_val_ac | 29 | 0.681 | 0.717 | 0.175 | 0.920 | 0.239 |
| **best_val_ac final** | **mean; range** | **0.701; 0.681-0.735** | **0.719; 0.703-0.736** | **0.191; 0.175-0.206** | **0.876** | **0.175** |
| fixed_epoch6 | 7 | 0.700 | 0.718 | 0.244 | 0.838 | 0.138 |
| fixed_epoch6 | 17 | 0.663 | 0.664 | 0.151 | 0.809 | 0.146 |
| fixed_epoch6 | 29 | 0.684 | 0.697 | 0.148 | 0.861 | 0.176 |
| **fixed_epoch6 final** | **mean; range** | **0.682; 0.663-0.700** | **0.693; 0.664-0.718** | **0.181; 0.148-0.244** | **0.836** | **0.153** |

## P3 comparison

| Model | Holdout AC | AUROC | AUPRC | AC vs compact30 | AC vs global59 |
|---|---:|---:|---:|---:|---:|
| legacy best_val_ac | 0.701 | 0.719 | 0.191 | +0.120 | +0.111 |
| legacy fixed_epoch6 | 0.682 | 0.693 | 0.181 | +0.101 | +0.092 |
| P3 demo10 | 0.567 | 0.776 | 0.260 | -0.014 | -0.023 |
| P3 compact30 | 0.581 | 0.772 | 0.283 | +0.000 | -0.009 |
| P3 global59 | 0.590 | 0.772 | 0.257 | +0.009 | +0.000 |
| P3 stage155 | 0.494 | 0.737 | 0.228 | -0.087 | -0.096 |

## Conclusions

1. The legacy model's primary mean Age-conditioned AUROC is **0.701**.
2. Its repeat means are **0.735**, **0.687**, and **0.681**; the range is **0.681-0.735** (width 0.054).
3. Best-val-AC exceeds fixed epoch 6 by **+0.018 AC**, **+0.026 AUROC**, and **+0.010 AUPRC**. Checkpoint choice matters, but it is not the main source of the full split variation.
4. The mean natural-train minus holdout AC gap remains large: **0.175** for best-val-AC and **0.153** for epoch 6. This indicates substantial fitting/split sensitivity.
5. At the pre-specified hierarchical level, both legacy checkpoints exceed compact30 (0.581) and global59 (0.590) in all three repeat means. The primary model gains **+0.120** and **+0.111** AC respectively. This is not uniform fold-by-fold: best-val-AC seed29/fold0 is 0.589, just below global59, and fixed-epoch6 seed17/fold1 is 0.552.
6. The repeated-CV result supports real ranking signal, because all best-val repeat means remain well above chance and above the P3 compact/global references. However, the 0.054 repeat-range, wide individual-fold spread (0.589-0.795), and large train-holdout gap show that the unusually high historical fixed-split Test/External results should be treated as substantially split-sensitive rather than as a stable point estimate of generalization.

Platt probabilities and thresholds are retained in each fold artifact for diagnostics only. Every CV conclusion above uses raw decision-logit ranking. No test/external evaluation or post-hoc experiment selection was performed.
