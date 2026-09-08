# P0 seed-7 paired training and evaluation results

## Run identity and permitted patch

- Base Git commit: `655c44bccef18f19e3078fd94a769c143c655d8d`.
- Runtime code: the base commit plus the recorded uncommitted patch in `launch_patch.diff`.
- The only production change skips `evaluate(model, external_loader)` at the end of training because all 54 cached external labels are `-1`; validation evaluation, calibration, threshold selection, training, and checkpoint saving are unchanged.
- Both arms used GPU 0, seed 7, the same `npz_new`, `split`, 691-column rules, model, optimizer, sampler, and training configuration. Neither arm resumed or loaded old weights.
- Checkpoint modes were verified as `legacy_clip` for A and `typed_v1` for B.

The true-label inputs were the existing files under `/database/home/gaohaojie/workspace/challenge2026/output/input`: test labels (158; 147 negative/11 positive), external labels (54; 46 negative/8 positive), and the shared prevalence file (1103; 1019 negative/84 positive). Their patient keys and order exactly matched the fixed test/external splits. NPZ `y=-1` values were not used for scoring.

## Training results

| Arm | Input preprocessing | Best epoch | Stop epoch | Best validation AC-AUROC | Best validation AUROC | Platt coefficient | Platt intercept | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | `legacy_clip` | 7 | 22 | 0.681034 | 0.675514 | 0.545814 | -3.057671 | 0.075693 |
| B | `typed_v1` | 16 | 31 | 0.590517 | 0.629566 | 0.028964 | -1.994392 | 0.073785 |

Both runs stopped by the unchanged patience-15 rule. Losses and predictions remained finite. The training log field named `val_tpr5` is an existing top-5%-of-records diagnostic and is **not** interpreted as TPR at FPR=5%.

## Official test metrics (158 participants)

| Metric | A: legacy_clip | B: typed_v1 | B - A |
|---|---:|---:|---:|
| Reward | 0.440277 | 0.349397 | -0.090881 |
| Age-conditioned AUROC | 0.812121 | 0.800000 | -0.012121 |
| Age-weighted AUROC | 0.825014 | 0.797149 | -0.027865 |
| AUROC | 0.814162 | 0.842919 | +0.028757 |
| AUPRC | 0.286727 | 0.565720 | +0.278992 |
| Accuracy | 0.734177 | 0.639241 | -0.094937 |
| F-measure | 0.300000 | 0.259740 | -0.040260 |

## Official external metrics (54 participants)

| Metric | A: legacy_clip | B: typed_v1 | B - A |
|---|---:|---:|---:|
| Reward | 0.166862 | 0.041411 | -0.125451 |
| Age-conditioned AUROC | 0.739583 | 0.520833 | -0.218750 |
| Age-weighted AUROC | 0.686223 | 0.548405 | -0.137817 |
| AUROC | 0.790761 | 0.611413 | -0.179348 |
| AUPRC | 0.419613 | 0.261975 | -0.157637 |
| Accuracy | 0.777778 | 0.537037 | -0.240741 |
| F-measure | 0.400000 | 0.242424 | -0.157576 |

## Execution status

All four prediction files exactly covered their true-label lists (158/158 test and 54/54 external for each arm), with unique patient keys, valid binary predictions, and finite probabilities. No cache miss, fallback, feature re-extraction, resume, non-finite loss/prediction, or runtime exception was observed. The official rounded values are retained in each arm's `test/scores.csv` and `external/scores.csv`; the corresponding prediction file is `demographics.csv` and the official age table is `table.csv`.
