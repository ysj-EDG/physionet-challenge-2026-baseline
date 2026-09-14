# P6-A site decodability audit

Base commit: `69e2aecef497f879b6a9e9c1953474191b8e6d91`. Input was exclusively the frozen 1103-record `npz_new` manifest from P5.

Each temporal continuous family was summarized by per-column median and IQR; stage/event columns used whole-night means. Five-fold site-stratified CV used fixed seed 7. Median imputation and StandardScaler were fitted within each training fold, followed by fixed L2 logistic regression (`C=1`, balanced class weights, no tuning).

| Feature family | Dim | Balanced accuracy mean | SD | Min-max | Macro F1 | Macro OVR AUROC |
|---|---:|---:|---:|---:|---:|---:|
| respiration14 | 28 | 0.695 | 0.038 | 0.669-0.759 | 0.553 | 0.858 |
| global59 | 59 | 0.684 | 0.034 | 0.645-0.730 | 0.598 | 0.850 |
| emg24 | 48 | 0.665 | 0.060 | 0.604-0.734 | 0.571 | 0.854 |
| static196 | 196 | 0.664 | 0.067 | 0.577-0.755 | 0.641 | 0.847 |
| compact30 | 30 | 0.659 | 0.036 | 0.616-0.708 | 0.580 | 0.842 |
| stage_event13 | 13 | 0.619 | 0.058 | 0.532-0.682 | 0.504 | 0.795 |
| eeg_spectral54 | 108 | 0.615 | 0.071 | 0.525-0.716 | 0.552 | 0.815 |
| coherence360 | 720 | 0.614 | 0.043 | 0.560-0.665 | 0.600 | 0.807 |
| demo10 | 10 | 0.597 | 0.055 | 0.552-0.677 | 0.559 | 0.783 |
| bsr18 | 36 | 0.424 | 0.016 | 0.401-0.444 | 0.449 | 0.558 |
| ecg12 | 24 | 0.383 | 0.018 | 0.361-0.408 | 0.342 | 0.564 |

## Diagnostic conclusions

1. Respiration was the most site-decodable individual family (balanced accuracy 0.695); global59, EMG, static196, and compact30 also carried strong site signatures.
2. Compact30 (0.659) was only slightly less site-decodable than static196 (0.664); compactness did not remove most site information.
3. Coherence360 was clearly site-decodable (0.614) but was not the strongest family despite its 720 summary dimensions.
4. EEG spectral features also showed a clear site signature (0.615), similar in magnitude to coherence.
5. Several high-dimensional physiological families carry site information consistent with a domain-shift concern, but this audit does not establish causality for P5 LOSO failure.

Random balanced-accuracy reference is approximately 0.333. High site decodability does not by itself justify feature removal.
