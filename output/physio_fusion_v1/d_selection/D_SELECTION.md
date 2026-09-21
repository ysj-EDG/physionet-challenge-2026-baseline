# Physio Fusion V1 Gate 2: coherence bottleneck selection

**Gate 2 is accepted and locked at `d=16`. No additional d-search is scheduled.**

Selection is CI-blind. Each outer fold uses only the other two sites, deterministic inner 80/20 record splits (seed=7), valid-pair-only median/IQR scaling, and equal site-weighted validation MSE.

## Per-fold d selection

| Held-out site | d=4 mean ± SE | d=8 mean ± SE | d=12 mean ± SE | d=16 mean ± SE | threshold | selected d |
|---|---:|---:|---:|---:|---:|---:|
| I0002 | 0.098147413 ± 0.0078451208 | 0.030920021 ± 0.0019649108 | 0.0099147755 ± 0.0006220885 | 0.0051684209 ± 0.00072566752 | 0.0058940884 | **16** |
| I0006 | 0.091109774 ± 0.003542876 | 0.026674557 ± 0.00065492366 | 0.0086198813 ± 0.00062292734 | 0.0063393532 ± 0.0015470808 | 0.007886434 | **16** |
| S0001 | 0.093602856 ± 0.0030086889 | 0.032767864 ± 0.0049588897 | 0.008516433 ± 0.00040624683 | 0.0049678171 ± 0.00017211054 | 0.0051399276 | **16** |

## Selected d=16 family metrics

Aggregated over the three seeds and three outer folds; these are reconstruction diagnostics, not CI-selection metrics.

| Family | Descriptor positions | MSE | Pearson | R² |
|---|---:|---:|---:|---:|
| means | 0,1,2,3,4 | 0.0071424879 | 0.99140738 | 0.9828276 |
| auc | 5,6,7,8,9 | 0.0077574182 | 0.99063833 | 0.98129975 |
| iqr | 10,11,12,13,14 | 0.00024330282 | 0.999855 | 0.99970212 |
| ratios | 15,16,17,18 | 0.00174276 | 0.99969419 | 0.99938023 |
| spectrum | 19,20,21 | 0.0053840511 | 0.99451422 | 0.98916371 |
| sigma | 22,23 | 0.016368228 | 0.98016786 | 0.96068725 |

## Selected d=16 per-feature metrics

Aggregated over the three seeds and three outer folds.

| Feature position | Family | MSE | Pearson | R² |
|---:|---|---:|---:|---:|
| 0 | means | 0.0065577007 | 0.99158566 | 0.9832028 |
| 1 | means | 0.010185509 | 0.9878483 | 0.97572659 |
| 2 | means | 0.0073815491 | 0.99138997 | 0.98278464 |
| 3 | means | 0.0029789892 | 0.99654743 | 0.99306495 |
| 4 | means | 0.0086086919 | 0.98966552 | 0.97935899 |
| 5 | auc | 0.007103304 | 0.99080669 | 0.9816404 |
| 6 | auc | 0.01090586 | 0.98695186 | 0.97396171 |
| 7 | auc | 0.0084884553 | 0.9900738 | 0.98015534 |
| 8 | auc | 0.0037605809 | 0.99560366 | 0.99120215 |
| 9 | auc | 0.0085288905 | 0.98975563 | 0.97953917 |
| 10 | iqr | 0.00018104141 | 0.99988477 | 0.99976572 |
| 11 | iqr | 0.00018419295 | 0.99989159 | 0.99977084 |
| 12 | iqr | 0.00016733948 | 0.99989791 | 0.99978923 |
| 13 | iqr | 0.0001168375 | 0.99992986 | 0.99984669 |
| 14 | iqr | 0.0005671028 | 0.99967088 | 0.9993381 |
| 15 | ratios | 0.0016006677 | 0.99971583 | 0.99942385 |
| 16 | ratios | 0.0015443718 | 0.99970487 | 0.9994009 |
| 17 | ratios | 0.0020409644 | 0.99973139 | 0.99945629 |
| 18 | ratios | 0.0017850361 | 0.99962469 | 0.99923989 |
| 19 | spectrum | 0.007612294 | 0.99531317 | 0.9906125 |
| 20 | spectrum | 0.0075845452 | 0.98878604 | 0.97800201 |
| 21 | spectrum | 0.00095531409 | 0.99944347 | 0.99887662 |
| 22 | sigma | 0.013493525 | 0.98425838 | 0.96870147 |
| 23 | sigma | 0.019242931 | 0.97607735 | 0.95267303 |

## Protocol checks

- Outer holdout records were excluded from scaler fitting, autoencoder training, and d selection.
- Inner validation used deterministic 80/20 record-level splits within each training site (seed=7).
- Training pairs used equal site weighting and valid-pair-only robust scaling.
- Stage/Event13, labels, demographics, circadian cosine, CAISR20, and outer-test metrics were not used.
