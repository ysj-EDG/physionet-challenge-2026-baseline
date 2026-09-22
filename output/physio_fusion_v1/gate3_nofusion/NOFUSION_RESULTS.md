# Physio Fusion V1 Gate 3 NoFusion

## Decision

**GATE3_NOFUSION = COMPLETE**

Gate 2.5 coherence bottleneck was rerun at locked `d=16`; no additional d-search was performed. Stage/Event, circadian, CAISR20, demographics, validity inputs, and Gate 4 Fusion192 were excluded.

## Gate 2.5 reproducibility

| Held-out | seed 7 | seed 17 | seed 27 | mean | locked | relative difference |
|---|---:|---:|---:|---:|---:|---:|
| I0002 | 0.0041105691 | 0.0048368282 | 0.0065578654 | 0.0051684209 | 0.0051684209 | 0.0000% |
| I0006 | 0.0037098607 | 0.0062417594 | 0.0090664394 | 0.0063393532 | 0.0063393532 | 0.0000% |
| S0001 | 0.0047146907 | 0.0052963986 | 0.004892362 | 0.0049678171 | 0.0049678171 | 0.0000% |

## NoFusion metrics

| Held-out | AE seed | AC | AUROC | AUPRC | nonzero coefficients |
|---|---:|---:|---:|---:|---:|
| I0002 | 7 | 0.621421 | 0.65391818 | 0.32486898 | 260 |
| I0002 | 17 | 0.63944857 | 0.67696629 | 0.3182345 | 503 |
| I0002 | 27 | 0.62778367 | 0.66400173 | 0.29996743 | 261 |
| I0006 | 7 | 0.61450355 | 0.70539182 | 0.19842246 | 252 |
| I0006 | 17 | 0.63810115 | 0.72895284 | 0.21654828 | 261 |
| I0006 | 27 | 0.62271441 | 0.71456311 | 0.21039966 | 254 |
| S0001 | 7 | 0.62682499 | 0.69773813 | 0.15782221 | 132 |
| S0001 | 17 | 0.62061646 | 0.68920598 | 0.1456065 | 134 |
| S0001 | 27 | 0.62228377 | 0.69443132 | 0.14801313 | 134 |

## Protocol and leakage checks

- 361-D epoch order: spectral54, BSR18, coherence15x16, HRV11, EMG24, Resp14; pooled representation is 722-D.
- Outer test records were excluded from AE fitting, coherence scaler, epoch scaler, patient imputer, patient scaler, and classifier fitting.
- Age was used only for metrics; validity was never a classifier input.
- No Stage/Event13, circadian, CAISR20, demographics, calibration, thresholding, or Fusion192 was run.
