# 1103 timegrid_v2 bridge results

OLD and NEW contain the same 1103 patients, labels, sites, LOSO folds, model protocols, and seeds. The principal systematic change is extraction version; these are controlled observations, not strict causal proof.

## Input changes

Overlapping epochs are compared by index; this describes change and does not prove correctness.

- Epoch count changed: 1090/1103 records.
- Epoch difference median 6.0, range 0 to 6.
- Median aligned X_seq MAE: 0.00163087.
- Any static change: 1103/1103 records; unchanged static dimensions: 132/196.

## Site decodability OLD vs NEW

| Family | OLD | NEW | Delta |
|---|---:|---:|---:|
| global59 | 0.684 | 0.686 | +0.002 |
| respiration14 | 0.695 | 0.681 | -0.014 |
| static196 | 0.664 | 0.660 | -0.004 |
| emg24 | 0.665 | 0.654 | -0.011 |
| compact30 | 0.659 | 0.634 | -0.024 |
| stage_event13 | 0.619 | 0.632 | +0.013 |
| demo10 | 0.597 | 0.604 | +0.006 |
| eeg_spectral54 | 0.615 | 0.602 | -0.013 |
| coherence360 | 0.614 | 0.597 | -0.017 |
| ecg12 | 0.383 | 0.397 | +0.014 |
| bsr18 | 0.424 | 0.376 | -0.048 |

## Model bridge

| Model | OLD sites | OLD Macro/Worst | NEW sites | NEW Macro/Worst | Macro delta |
|---|---:|---:|---:|---:|---:|
| demo10 | 0.542/0.382/0.513 | 0.479/0.382 | 0.542/0.382/0.513 | 0.479/0.382 | +0.000 |
| compact30 | 0.583/0.533/0.522 | 0.546/0.522 | 0.625/0.531/0.526 | 0.560/0.526 | +0.014 |
| global59 | 0.646/0.494/0.491 | 0.544/0.491 | 0.646/0.494/0.491 | 0.544/0.491 | +0.000 |
| L0 | 0.646/0.474/0.505 | 0.541/0.474 | 0.729/0.537/0.600 | 0.622/0.537 | +0.081 |
| L1 | 0.667/0.532/0.563 | 0.587/0.532 | 0.708/0.545/0.531 | 0.595/0.531 | +0.008 |
| L2 | 0.646/0.562/0.617 | 0.608/0.562 | 0.531/0.591/0.524 | 0.549/0.524 | -0.060 |

## Predeclared decisions

1. global59 retention: **FAIL**. Versus NEW compact30: Macro -0.017, Worst -0.035, 1/3 sites improve. Downgrade it to diagnostic and stop the P3-style EEG-pooling main line.
2. B1 ordering `L0 < L1 <= L2`: **not confirmed**. NEW-L1 versus L0: Macro -0.027, Worst -0.006.
3. B2 versus NEW-L0: **FAIL**. Macro -0.073, Worst -0.014, 1/3 sites improve.
4. Historical L0 improves on all three sites: Macro +0.081, Worst +0.063.
5. For a later data2 LSTM stage, NEW-L0 is the primary supported protocol. L1 is at most a fixed mechanism control; L2 is not promoted. No data2 LSTM was launched.
6. Runtime note: demo10/I0002 reached the frozen max_iter=100000 without convergence; the warning was retained and no LR setting was changed or rerun.

## Directions

- Continue the frozen legacy LSTM architecture only under L0 if data2 training is later authorized.
- Downgrade global59 to diagnostic and stop P3-style EEG pooling as a primary path.
- Do not promote B1/B2 or tune sampling weights post hoc.
