# Baseline registry

Old1103/data2 differences are descriptive; extractor, sample size, and site composition all change.

| ID | Dataset | Model/features | Protocol | I0002 | I0006 | S0001 | Macro | Worst | Role |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| OLD-D | OLD1103 / old npz_new | elastic-net LR / demo10 | three-site LOSO | 0.542 | 0.382 | 0.513 | 0.479 | 0.382 | historical LOSO benchmark |
| OLD-C | OLD1103 / old npz_new | elastic-net LR / compact30 | three-site LOSO | 0.583 | 0.533 | 0.522 | 0.546 | 0.522 | historical LOSO benchmark |
| OLD-G | OLD1103 / old npz_new | elastic-net LR / global59 | three-site LOSO | 0.646 | 0.494 | 0.491 | 0.544 | 0.491 | historical LOSO benchmark |
| OLD-L0 | OLD1103 / old npz_new | 2-layer LSTM / 483+12+196 | three-site LOSO fixed epoch6 | 0.646 | 0.474 | 0.505 | 0.541 | 0.474 | historical LOSO benchmark |
| OLD-L1 | OLD1103 / old npz_new | 2-layer LSTM / 483+12+196 | three-site LOSO fixed epoch6 | 0.667 | 0.532 | 0.563 | 0.587 | 0.532 | historical LOSO benchmark |
| OLD-L2 | OLD1103 / old npz_new | 2-layer LSTM / 483+12+196 | three-site LOSO fixed epoch6 | 0.646 | 0.562 | 0.617 | 0.608 | 0.562 | historical LOSO benchmark |
| P4-MIXED | OLD1103 / old npz_new | 2-layer LSTM / 483+12+196 | 3x3 mixed-site repeated CV | N/A | N/A | N/A | 0.701 | N/A | MIXED-SITE REFERENCE ONLY |
| DATA2-D | data2 timegrid_v2 | elastic-net LR / demo10 | three-site LOSO | 0.605 | 0.490 | 0.577 | 0.557 | 0.490 | current data2 low-cost baseline |
| DATA2-C | data2 timegrid_v2 | elastic-net LR / compact30 | three-site LOSO | 0.748 | 0.681 | 0.654 | 0.694 | 0.654 | current data2 low-cost baseline |
| DATA2-G | data2 timegrid_v2 | elastic-net LR / global59 | three-site LOSO | 0.748 | 0.686 | 0.614 | 0.683 | 0.614 | current data2 low-cost baseline |

`P4-MIXED` is a MIXED-SITE REFERENCE ONLY and must not be interpreted as the same protocol as LOSO.
