# P5: three-site LOSO on the frozen legacy 1103 cache

## Scope

- Base commit: `3bda28a4b10152c86564e84bfd11237fba3dad7a`.
- Sites: I0002, I0006, S0001; each site is held out in full once with patient-level isolation.
- Inputs: the existing 1103 `npz_new` caches and existing P3 stage sidecars only. No feature extraction or NPZ modification occurred.
- Truth: the previously used official `prevalence.csv` (1103 unique site/patient keys), parsed with `helper_code.load_label`; NPZ labels were not used.
- LSTM: P4 `legacy_clip`, original 2-layer architecture, seed 7, balanced sampler + empirical pos_weight, exactly 6 epochs.
- LR: exact P3 v2 demo10, compact30, and global59 definitions/configuration; typed scaling and imputation fit only on outer training sites.
- All main results use raw decision-score ranking. No holdout calibration, thresholding, model selection, or epoch selection was performed.

## Main results

| Model | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site AC | Macro AUROC | Macro AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0_demo10 | 0.542 | 0.382 | 0.513 | 0.479 | 0.382 | 0.707 | 0.233 |
| L1_compact30 | 0.583 | 0.533 | 0.522 | 0.546 | 0.522 | 0.731 | 0.260 |
| L2_global59 | 0.646 | 0.494 | 0.491 | 0.544 | 0.491 | 0.734 | 0.275 |
| L3_legacy_lstm | 0.646 | 0.474 | 0.505 | 0.541 | 0.474 | 0.634 | 0.206 |

P4 legacy mixed-site repeated-CV AC=0.701 is retained only as a historical reference; its protocol differs from LOSO and the difference is not interpreted as a pure domain-shift effect.

## Fold details

| Model | Holdout | n (+/-) | Eligible pairs | Train AC | Holdout AC | Age-weighted | AUROC | AUPRC | Train-holdout AC |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L0_demo10 | I0002 | 54 (8/46) | 48 | 0.557 | 0.542 | 0.557 | 0.707 | 0.278 | +0.015 |
| L0_demo10 | I0006 | 192 (20/172) | 441 | 0.580 | 0.382 | 0.411 | 0.646 | 0.237 | +0.198 |
| L0_demo10 | S0001 | 857 (56/801) | 5124 | 0.648 | 0.513 | 0.550 | 0.768 | 0.184 | +0.135 |
| L1_compact30 | I0002 | 54 (8/46) | 48 | 0.689 | 0.583 | 0.607 | 0.736 | 0.292 | +0.106 |
| L1_compact30 | I0006 | 192 (20/172) | 441 | 0.690 | 0.533 | 0.543 | 0.703 | 0.281 | +0.157 |
| L1_compact30 | S0001 | 857 (56/801) | 5124 | 0.857 | 0.522 | 0.593 | 0.752 | 0.208 | +0.335 |
| L2_global59 | I0002 | 54 (8/46) | 48 | 0.751 | 0.646 | 0.650 | 0.764 | 0.324 | +0.105 |
| L2_global59 | I0006 | 192 (20/172) | 441 | 0.747 | 0.494 | 0.499 | 0.698 | 0.298 | +0.253 |
| L2_global59 | S0001 | 857 (56/801) | 5124 | 0.866 | 0.491 | 0.559 | 0.739 | 0.204 | +0.375 |
| L3_legacy_lstm | I0002 | 54 (8/46) | 48 | 0.842 | 0.646 | 0.653 | 0.772 | 0.333 | +0.196 |
| L3_legacy_lstm | I0006 | 192 (20/172) | 441 | 0.900 | 0.474 | 0.449 | 0.558 | 0.185 | +0.426 |
| L3_legacy_lstm | S0001 | 857 (56/801) | 5124 | 0.777 | 0.505 | 0.517 | 0.571 | 0.099 | +0.273 |

## Run integrity and exceptions

- Independent post-run verification confirmed exact manifest coverage, label/age alignment, finite scores, and exact reproduction of all 12 saved holdout metric bundles from raw logits.
- The three LSTM runs completed all six epochs with finite losses; each checkpoint was frozen before its holdout loader was constructed.
- `L0_demo10` for holdout I0002 reached the frozen P3 `max_iter=100000` without convergence. Its outputs are retained and flagged; no hyperparameter was changed and the fold was not rerun.

## Conclusions

1. Legacy LSTM holdout AC was 0.646 on I0002, 0.474 on I0006, and 0.505 on S0001.
2. Its site-macro AC was 0.541, and its worst-site AC was 0.474 (I0006).
3. Legacy was clearly above 0.5 only on I0002. S0001 was 0.505, effectively near chance, while I0006 was below chance at 0.474; therefore the experiment does not establish clear above-chance transfer on at least two unseen sites.
4. Training with S0001 transferred useful ranking to I0002 (0.646) but not to I0006 (0.474). Conversely, training only on I0002+I0006 did not reproduce the earlier S0001-dominated mixed-site signal (S0001 LOSO 0.505). The transfer is site-specific rather than general.
5. The earlier weak I0006 evidence persists under true LOSO: every model except compact30 was at or below 0.5 AC on I0006, and legacy reached only 0.474.
6. Compact30 dropped from the historical P3 mixed-site AC 0.581 to LOSO macro 0.546; global59 dropped from 0.590 to 0.544. These are protocol-level comparisons, not pure estimates of domain-shift magnitude.
7. The frozen global EEG summaries did not provide a stable unseen-site increment over compact30: global59 changed AC by +0.063 on I0002, -0.039 on I0006, and -0.031 on S0001, with macro AC lower by 0.002.
8. Legacy exceeded global59 on only one site (S0001, +0.014), tied it on I0002 to displayed and full precision, and was lower on I0006 (-0.020). Its macro AC was lower by 0.002, so the P4 legacy advantage over global59 did not survive LOSO.
9. The S0001 holdout fold had the smallest training pool (246 records, 28 positives) and showed large train-holdout gaps for compact30/global59 (0.335/0.375), so it is most structurally exposed to limited training-site composition. For legacy specifically, the largest gap occurred on I0006 (0.426), indicating additional site mismatch beyond training-set size.
10. The evidence supports priority B: investigate site-domain effects and data2 before further temporal/static branch ablation. Mixed-site P4 AC=0.701 did not translate into robust three-site LOSO performance; no next experiment is launched automatically.
