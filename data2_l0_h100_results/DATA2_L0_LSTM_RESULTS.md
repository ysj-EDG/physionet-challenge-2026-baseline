# DATA2-L0 legacy LSTM LOSO results

## Run identity and verification

- Source HEAD before the experiment: `ee925dd317af950f35cb632da57a86400dc999be` on `official-submission`; the new runner and Medex launcher are delivered with this result commit.
- Frozen split preparation and one-batch smoke test ran locally in `sleepfm_env` on Quadro P6000; this is the environment recorded in `preflight.json`.
- Formal training ran sequentially on an NVIDIA H100 PCIe in Medex `eeg_env` (Python 3.10.12, PyTorch 2.5.1+cu121, CUDA 12.1). No P6000 checkpoint was used.
- All six raw-logit files were independently rescored with the repository's official age-conditioned AUROC implementation; values matched `aggregate_metrics.json` to floating-point precision.
- All six checkpoints were checked for nonempty model state and the frozen `legacy_clip`, global class-balanced sampler, empirical `pos_weight`, seed 7 protocol.
- The three NPZ/cache directories and the interrupted local P6000 output were not packaged.

Outer holdout sites were never used for training, preprocessing, scheduler, checkpoint selection, thresholding, or calibration. The primary result is the best seen-site validation macro-AC checkpoint; epoch 6 is diagnostic only.

| Holdout | Inner train | Inner val | Outer n | Best/stop epoch | Train AC | Seen-val macro AC | Outer AC | Epoch6 AC | Train-val gap | Val-outer gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 5024 | 1257 | 319 | 1/16 | 0.756 | 0.773 | 0.661 | 0.653 | -0.018 | +0.113 |
| I0006 | 4366 | 1092 | 1142 | 1/16 | 0.793 | 0.768 | 0.517 | 0.531 | +0.025 | +0.252 |
| S0001 | 1168 | 293 | 5139 | 19/34 | 0.866 | 0.774 | 0.439 | 0.502 | +0.092 | +0.336 |

## Aggregate and fixed comparisons

- Primary Macro AC: **0.539**; Worst-site AC: **0.439**.
- Epoch6 diagnostic Macro/Worst: 0.562/0.502.
- DATA2 compact30 reference: I0002/I0006/S0001 = 0.748/0.681/0.654; Macro/Worst = 0.694/0.654.
- Sites not below compact30: 0/3.
- Predeclared stable-value success: **FAIL**.
- Predeclared robustness-tradeoff condition: **NO**.
- NEW1103-L0 reference Macro/Worst = 0.622/0.537; OLD1103-L0 = 0.541/0.474. Cross-size differences are descriptive because sample size and composition change.

## Interpretation

- The S0001 holdout fold trains only on I0002+I0006 and therefore has by far the smallest inner-training set; interpret its domain gap together with this fixed data-size limitation.
- Metric-aligned ranking or typed LSTM may be considered only after reviewing this frozen result; neither experiment was launched here.
