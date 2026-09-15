# Site decodability on NEW1103 timegrid_v2

The classifier protocol and patient-level summaries directly reuse P6-A. All 1103 records have unique patient IDs; five-fold splitting nevertheless remains patient-group-aware while stratifying site.

| Feature family | Dim | NEW1103 balanced accuracy | SD | Min-max | Macro F1 | Old1103 P6-A | Descriptive delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| global59 | 59 | 0.686 | 0.095 | 0.594-0.833 | 0.601 | 0.684 | +0.002 |
| respiration14 | 28 | 0.681 | 0.048 | 0.624-0.736 | 0.547 | 0.695 | -0.014 |
| static196 | 196 | 0.660 | 0.078 | 0.577-0.767 | 0.630 | 0.664 | -0.004 |
| emg24 | 48 | 0.654 | 0.064 | 0.574-0.731 | 0.570 | 0.665 | -0.011 |
| compact30 | 30 | 0.634 | 0.058 | 0.561-0.687 | 0.565 | 0.659 | -0.024 |
| stage_event13 | 13 | 0.632 | 0.078 | 0.553-0.716 | 0.513 | 0.619 | +0.013 |
| demo10 | 10 | 0.604 | 0.067 | 0.502-0.678 | 0.561 | 0.597 | +0.006 |
| eeg_spectral54 | 108 | 0.602 | 0.068 | 0.514-0.677 | 0.555 | 0.615 | -0.013 |
| coherence360 | 720 | 0.597 | 0.036 | 0.562-0.655 | 0.588 | 0.614 | -0.017 |
| ecg12 | 24 | 0.397 | 0.041 | 0.357-0.439 | 0.334 | 0.383 | +0.014 |
| bsr18 | 36 | 0.376 | 0.025 | 0.352-0.417 | 0.378 | 0.424 | -0.048 |

The old1103 comparison is descriptive only: patients, labels, sites, and protocol are matched; extractor version changed. High decodability is diagnostic and does not imply automatic feature removal.
