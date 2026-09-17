# DATA2 CAISR20-only LOSO baseline

- Git base: `07617864c97bd62392d5fd353c4bef4aac4cfbe0`
- DATA2: `/database/home/gaohaojie/workspace/python-example-2026/npz_data2` (6600 unique subjects; 498 positive / 6102 negative)
- Model config: `{"penalty": "elasticnet", "solver": "saga", "l1_ratio": 0.4, "C": 0.03, "class_weight": "balanced", "fit_intercept": true, "max_iter": 100000, "tol": 0.0001, "random_state": 7}`
- Preprocessing: identical frozen DATA2 compact30 fold-specific typed selected-column transform, then P3 median imputation/linear scaling fitted on outer training sites only.
- Official primary metric: Age-conditioned AUROC, gap=2, via the existing P5 wrapper.

## Exact CAISR20 manifest

1. `caisr_sleep_tst_sec` (x_static index 11)
2. `caisr_sleep_se` (x_static index 12)
3. `caisr_sleep_sol_sec` (x_static index 13)
4. `caisr_sleep_rem_latency_sec` (x_static index 14)
5. `caisr_sleep_waso_sec` (x_static index 16)
6. `caisr_sleep_n1_pct` (x_static index 17)
7. `caisr_sleep_n2_pct` (x_static index 18)
8. `caisr_sleep_n3_pct` (x_static index 19)
9. `caisr_sleep_rem_pct` (x_static index 20)
10. `caisr_sleep_transition_rate` (x_static index 32)
11. `caisr_sleep_short_bout_ratio` (x_static index 34)
12. `caisr_sleep_mean_stage_entropy` (x_static index 87)
13. `caisr_arousal_arousal_index` (x_static index 96)
14. `caisr_arousal_arousal_burden_ratio` (x_static index 97)
15. `caisr_arousal_mean_arousal_duration` (x_static index 98)
16. `caisr_respiratory_respiratory_event_index` (x_static index 126)
17. `caisr_respiratory_respiratory_burden_ratio` (x_static index 127)
18. `caisr_respiratory_mean_resp_duration` (x_static index 141)
19. `caisr_limb_limb_movement_index` (x_static index 168)
20. `caisr_limb_plmi` (x_static index 173)

## Results

| Model | I0002 | I0006 | S0001 | Macro | Worst |
|---|---:|---:|---:|---:|---:|
| demo10 | 0.605 | 0.490 | 0.577 | 0.557 | 0.490 |
| CAISR20 | 0.746 | 0.678 | 0.617 | 0.680 | 0.617 |
| compact30 | 0.748 | 0.681 | 0.654 | 0.694 | 0.654 |

## Deltas

- CAISR20 - demo10: I0002 +0.141, I0006 +0.188, S0001 +0.040, Macro +0.123, Worst +0.127.
- compact30 - CAISR20: I0002 +0.002, I0006 +0.003, S0001 +0.037, Macro +0.014, Worst +0.037.

## Pre-specified interpretation

Case B with a site-concentrated effect: compact30 improves all three sites, but nearly all material gain is on S0001 (+0.037); I0002 and I0006 differ by only +0.002 and +0.003. Macro and Worst gaps exceed 0.01, so CAISR20 does not meet Case A.

No feature-definition, patient/site leakage, or preprocessing-fit anomaly was detected. The concurrent R1 process and its files were not stopped, changed, staged, or committed.
