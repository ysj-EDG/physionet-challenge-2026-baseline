# Physio Fusion V1 Gate4 Fusion192

## Status

**GATE4_FUSION192 = COMPLETE**

The nine paired Gate4 pipelines completed on H100. The original remote wrapper failed only while trying to load the Gate3 reference from a path absent in the Medex workspace; paired summaries were rebuilt locally from the returned Gate3 reference and nine Gate4 metrics.

## Protocol

- Locked coherence bottleneck: `d=16`; no d-search was run.
- 361-D physiology input: spectral54, BSR18, coherence15x16, HRV11, EMG24, Resp14.
- Stage/Event13, circadian, CAISR20, demographics, age, site ID, and validity inputs were excluded from the model.
- Nine pipelines: outer sites I0002/I0006/S0001 × seeds 7/17/27, paired to the corresponding Gate3 AE seed.
- H100 environment: `eeg_env`, NVIDIA H100 PCIe.

## Per-fold metrics

| Held-out | Seed | AC | AUROC | AUPRC | Best epoch |
|---|---:|---:|---:|---:|---:|
| I0002 | 7 | 0.62831389 | 0.65737540 | 0.29714939 | 2 |
| I0002 | 17 | 0.63202545 | 0.63079804 | 0.28766805 | 4 |
| I0002 | 27 | 0.61770944 | 0.63195045 | 0.26897427 | 5 |
| I0006 | 7 | 0.64638101 | 0.72511269 | 0.20399008 | 9 |
| I0006 | 17 | 0.61346857 | 0.70010402 | 0.20427885 | 5 |
| I0006 | 27 | 0.61457255 | 0.69325589 | 0.19040503 | 2 |
| S0001 | 7 | 0.60564361 | 0.66978945 | 0.12830766 | 1 |
| S0001 | 17 | 0.59145505 | 0.63866481 | 0.12164037 | 3 |
| S0001 | 27 | 0.60559974 | 0.67269000 | 0.13906653 | 1 |

## Site summaries

| Held-out | Gate3 AC mean ± SD | Gate4 AC mean ± SD | Paired Δ mean ± SD |
|---|---:|---:|---:|
| I0002 | 0.629551 ± 0.009143 | 0.626016 ± 0.007429 | -0.003535 ± 0.009127 |
| I0006 | 0.625106 ± 0.011979 | 0.624807 ± 0.018691 | -0.000299 ± 0.029060 |
| S0001 | 0.623242 ± 0.003213 | 0.600899 ± 0.008179 | -0.022342 ± 0.006319 |

## Macro and worst-site summaries

- Gate3 frozen Macro AC: `0.625966 ± 0.006085`
- Gate4 Macro AC: `0.617241 ± 0.008262`
- Paired Macro Δ: `-0.008725 ± 0.013374`
- Gate3 frozen Worst-site AC: `0.619135 ± 0.004096`
- Gate4 Worst-site AC: `0.600899 ± 0.008179`
- Paired Worst Δ: `-0.022342 ± 0.006319`
- Sites with positive mean paired Δ: `0/3`.

## Leakage and safety

- All nine Gate2.5 AE checkpoints were reused; no AE was retrained.
- Inner split and outer holdout separation were preserved.
- No extractor or NPZ was modified.
- No Gate5, CAISR20 residual, Stage/Event experiment, or additional seeds were run.
- Checkpoints, NPZ, raw data, and Medex workspace were not included in the Git result package.

## Decision

This is a paired Gate4 result record. Performance category is left for human review; no automatic Gate5 decision was made.
