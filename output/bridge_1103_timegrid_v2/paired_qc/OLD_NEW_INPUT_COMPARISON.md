# OLD vs NEW1103 input comparison

Overlapping epochs are compared by index; this describes change and does not prove correctness.

- Epoch count changed: 1090/1103 records.
- Epoch difference median 6.0, range 0 to 6.
- Median aligned X_seq MAE: 0.00163087.
- Any static change: 1103/1103 records; unchanged static dimensions: 132/196.

## Site summary

| site | records | epoch_changed_records | epoch_difference_median | epoch_difference_q1 | epoch_difference_q3 | seq_mae_median | static_mae_median |
|---|---|---|---|---|---|---|---|
| I0002 | 54 | 53 | 6.0000 | 6.0000 | 6.0000 | 0.0140 | 2.1739 |
| I0006 | 192 | 190 | 6.0000 | 6.0000 | 6.0000 | 0.7079 | 2.0536 |
| S0001 | 857 | 847 | 6.0000 | 6.0000 | 6.0000 | 0.0016 | 2.1232 |

## Temporal family changes

| feature_family | mean | median | std | min | max |
|---|---|---|---|---|---|
| emg24 | 341121830.7006 | 0.0008 | 5079306377.8853 | 0.0000 | 154924462463.6492 |
| bsr18 | 0.3626 | 0.0000 | 4.0001 | 0.0000 | 66.6667 |
| stage_event13 | 0.0555 | 0.0551 | 0.0151 | 0.0000 | 0.1203 |
| coherence360 | 0.0062 | 0.0000 | 0.0692 | 0.0000 | 1.1265 |
| respiration14 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| eeg_spectral54 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

## Largest static dimension changes

| feature_index | mean_absolute_difference | changed_records |
|---|---|---|
| 10 | 401.3690 | 1103 |
| 13 | 312.6745 | 1103 |
| 150 | 30.3846 | 1085 |
| 125 | 25.7688 | 1085 |
| 110 | 22.7295 | 1041 |
| 167 | 17.4394 | 1082 |
| 154 | 11.1959 | 992 |
| 95 | 11.0384 | 1081 |
| 184 | 7.4188 | 922 |
| 186 | 5.8694 | 792 |
| 108 | 3.8095 | 1059 |
| 183 | 3.4732 | 1058 |
| 152 | 3.2359 | 788 |
| 109 | 3.2176 | 783 |
| 185 | 3.0859 | 1045 |
