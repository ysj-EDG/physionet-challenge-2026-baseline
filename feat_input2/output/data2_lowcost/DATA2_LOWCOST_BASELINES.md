# Data2 low-cost baselines

## 1. Data2 integrity

The complete local timegrid_v2 cache passed one-to-one manifest coverage, labels, sites, ages, version, dimensions, metadata alignment, and full finite-value checks: 6600 success, 0 failure.

## 2. Composition

| site | n | patients | positives | negatives | positive_prevalence | multi_session_patients |
|---|---|---|---|---|---|---|
| I0002 | 319 | 319 | 52 | 267 | 0.1630 | 0 |
| I0006 | 1142 | 1142 | 112 | 1030 | 0.0981 | 0 |
| S0001 | 5139 | 5139 | 334 | 4805 | 0.0650 | 0 |
| Total | 6600 | 6600 | 498 | 6102 | 0.0755 | 0 |

Eligible official gap=2 age-pair counts:

| site | n | positives | negatives | eligible_ac_pairs_gap2 |
|---|---|---|---|---|
| I0002 | 319 | 52 | 267 | 1886 |
| I0006 | 1142 | 112 | 1030 | 14493 |
| S0001 | 5139 | 334 | 4805 | 182330 |

## 3. Site decodability

| Family | Balanced accuracy | SD |
|---|---:|---:|
| static196 | 0.781 | 0.029 |
| global59 | 0.727 | 0.014 |
| emg24 | 0.722 | 0.018 |
| compact30 | 0.711 | 0.011 |
| respiration14 | 0.696 | 0.010 |
| eeg_spectral54 | 0.668 | 0.025 |
| coherence360 | 0.645 | 0.028 |
| stage_event13 | 0.640 | 0.019 |
| demo10 | 0.601 | 0.030 |
| ecg12 | 0.426 | 0.016 |
| bsr18 | 0.384 | 0.013 |

## 4. Three-site LOSO

| Model | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst AC |
|---|---:|---:|---:|---:|---:|
| demo10 | 0.605 | 0.490 | 0.577 | 0.557 | 0.490 |
| compact30 | 0.748 | 0.681 | 0.654 | 0.694 | 0.654 |
| global59 | 0.748 | 0.686 | 0.614 | 0.683 | 0.614 |

## 5. Descriptive old1103 comparison

| Model | Old1103 Macro/Worst | Data2 Macro/Worst |
|---|---:|---:|
| demo10 | 0.479 / 0.382 | 0.557 / 0.490 |
| compact30 | 0.546 / 0.522 | 0.694 / 0.654 |
| global59 | 0.544 / 0.491 | 0.683 / 0.614 |

## 6. Limited conclusions

- demo10 provides a modest population-only baseline (Macro AC 0.557), but its I0006 holdout is essentially chance-level (0.490).
- compact30 is the strongest and most stable low-cost baseline: Macro AC 0.694 and Worst-site AC 0.654, with all three sites above 0.65.
- global59 does not add cross-site AC over compact30 overall: Macro AC is 0.683 (-0.011), I0002 is tied, I0006 improves by 0.006, and S0001 decreases by 0.040. Its higher Macro AUPRC (0.364 vs 0.337) remains a secondary diagnostic.
- Site identity is most decodable from static196 (0.781), followed by global59 (0.727), EMG24 (0.722), and compact30 (0.711). High site decodability is diagnostic and is not evidence for automatic feature deletion.
- Data2 has 1,886 / 14,493 / 182,330 eligible AC pairs for I0002 / I0006 / S0001, versus 48 / 441 / 5,124 in old1103; evidence is much denser, but old-to-new differences are not causally attributable.

## 7. Questions deferred until the new1103 bridge

The comparison cannot determine whether timegrid_v2 itself improves performance, how much gain comes from 6600 records, whether B2 should be the final data2 LSTM protocol, whether typed beats legacy, or which 483-dimensional family should be removed. Old1103 to data2 changes extractor, sample size, and site composition simultaneously.

No LSTM experiment was launched.
