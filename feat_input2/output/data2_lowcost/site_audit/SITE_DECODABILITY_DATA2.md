# Site decodability on data2

The classifier protocol and patient-level summaries directly reuse P6-A. All 6600 records have unique patient IDs; five-fold splitting nevertheless remains patient-group-aware while stratifying site.

| Feature family | Dim | Data2 balanced accuracy | SD | Min-max | Macro F1 | Old1103 P6-A | Descriptive delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| static196 | 196 | 0.781 | 0.029 | 0.747-0.821 | 0.673 | 0.664 | +0.117 |
| global59 | 59 | 0.727 | 0.014 | 0.713-0.747 | 0.614 | 0.684 | +0.043 |
| emg24 | 48 | 0.722 | 0.018 | 0.703-0.738 | 0.575 | 0.665 | +0.057 |
| compact30 | 30 | 0.711 | 0.011 | 0.703-0.730 | 0.599 | 0.659 | +0.053 |
| respiration14 | 28 | 0.696 | 0.010 | 0.679-0.704 | 0.531 | 0.695 | +0.000 |
| eeg_spectral54 | 108 | 0.668 | 0.025 | 0.641-0.696 | 0.555 | 0.615 | +0.053 |
| coherence360 | 720 | 0.645 | 0.028 | 0.610-0.681 | 0.603 | 0.614 | +0.032 |
| stage_event13 | 13 | 0.640 | 0.019 | 0.614-0.661 | 0.515 | 0.619 | +0.021 |
| demo10 | 10 | 0.601 | 0.030 | 0.564-0.631 | 0.561 | 0.597 | +0.004 |
| ecg12 | 24 | 0.426 | 0.016 | 0.403-0.449 | 0.338 | 0.383 | +0.043 |
| bsr18 | 36 | 0.384 | 0.013 | 0.370-0.402 | 0.390 | 0.424 | -0.040 |

The old1103 comparison is descriptive only: extractor version, sample size, and site composition all changed. High decodability is diagnostic and does not imply automatic feature removal.
