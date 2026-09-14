# P6 site-domain audit and balanced LOSO

## Scope and integrity

- Frozen source commit: `69e2aecef497f879b6a9e9c1953474191b8e6d91` on `official-submission`.
- P6 used only the P5 1103-record manifest and legacy `npz_new`; neither timegrid_v2 extraction output participated.
- P6-A used fixed five-fold site-stratified CV with fold-local imputation/scaling and fixed L2 multinomial logistic regression.
- P6-B reused the P5 legacy LSTM, `legacy_clip`, seed 7, six epochs, and all optimizer/model settings. Only the prespecified sampler and BCE `pos_weight` changed.
- All six P6-B checkpoints were frozen before holdout evaluation. All losses and raw logits were finite; no stopping condition occurred.

The pre-P6 read-only extraction snapshot is recorded separately in `EXTRACTION_STATUS_BEFORE_P6.md`. Neither extraction task was stopped, restarted, cleaned, or modified.

## P6-A: site decodability

| Feature family | Dimension | Site balanced accuracy | Fold SD | Macro F1 | Macro OVR AUROC |
|---|---:|---:|---:|---:|---:|
| respiration14 | 28 | 0.695 | 0.038 | 0.553 | 0.858 |
| global59 | 59 | 0.684 | 0.034 | 0.598 | 0.850 |
| emg24 | 48 | 0.665 | 0.060 | 0.571 | 0.854 |
| static196 | 196 | 0.664 | 0.067 | 0.641 | 0.847 |
| compact30 | 30 | 0.659 | 0.036 | 0.580 | 0.842 |
| stage/event13 | 13 | 0.619 | 0.058 | 0.504 | 0.795 |
| EEG spectral54 | 108 | 0.615 | 0.071 | 0.552 | 0.815 |
| coherence360 | 720 | 0.614 | 0.043 | 0.600 | 0.807 |
| demo10 | 10 | 0.597 | 0.055 | 0.559 | 0.783 |
| BSR18 | 36 | 0.424 | 0.016 | 0.449 | 0.558 |
| ECG12 | 24 | 0.383 | 0.018 | 0.342 | 0.564 |

Respiration was the most decodable individual family. Global59, EMG, static196, and compact30 were also strong. Compact30 carried only slightly less site information than static196 (difference -0.005), so reduced dimensionality did not remove the site signature. Coherence and EEG spectral summaries were both clearly above the approximately 1/3 random reference, but coherence was not uniquely dominant despite its dimensionality. These associations are consistent with multiple routes for site information to enter a complex model; they do not prove that any family caused P5's LOSO failure and do not justify automatic feature deletion.

## P6-B: fixed LOSO comparison

| Protocol | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site AC |
|---|---:|---:|---:|---:|---:|
| P5 class sampler + empirical pos_weight | 0.646 | 0.474 | 0.505 | 0.541 | 0.474 |
| P6-B1 class sampler + pos_weight=1 | 0.667 | 0.532 | 0.563 | 0.587 | 0.532 |
| P6-B2 site+class sampler + pos_weight=1 | 0.646 | 0.562 | 0.617 | 0.608 | 0.562 |

| Protocol | Delta I0002 | Delta I0006 | Delta S0001 | Delta Macro | Delta Worst |
|---|---:|---:|---:|---:|---:|
| P6-B1 vs P5 | +0.021 | +0.058 | +0.059 | +0.046 | +0.058 |
| P6-B2 vs P5 | +0.000 | +0.088 | +0.112 | +0.067 | +0.088 |

B1 improved all three sites, showing that removing empirical positive weighting from an already class-balanced sampler corrected meaningful over-weighting. B2 improved 2/3 sites, left the small-pair I0002 result unchanged, and produced the largest gains specifically on I0006 and S0001. Its gain is therefore not an I0002-only artifact. B2 raised Macro AC to 0.608 and Worst-site AC to 0.562.

## Conclusions

1. The strongest site-decodability was respiration (0.695), followed by global59 (0.684), EMG/static/compact30 (0.665/0.664/0.659); spectral and coherence were both about 0.615.
2. High-dimensional families can carry substantial site information, but compact30 also retained almost as much as static196. Site signature is not explained by feature count alone.
3. Removing empirical `pos_weight` improved LOSO Macro AC by 0.046 and Worst-site AC by 0.058, with all three sites improving.
4. Site+class balanced sampling further improved Macro/Worst-site to 0.608/0.562, gains of 0.067/0.088 over P5.
5. The B2 improvement occurred on I0006 (+0.088) and S0001 (+0.112), while I0002 was unchanged; it satisfies the prespecified requirement of improvement on at least two sites.
6. P6 supports carrying site+class balanced sampling with unit `pos_weight` as a prespecified candidate into both `npz_1103_timegrid_v2` and data2 6600. This is evidence for the training protocol, not permission to tune on their future holdouts.
7. The legacy `npz_new` has now answered the planned scaling, CV, site-conditioning, LOSO, site-decodability, and balance questions. It can be frozen as the legacy benchmark rather than used for further feature search.

No new timegrid_v2/data2 model training was launched.
