# SIREN + Matcha B Cross-Fold Weight Selection v1

This branch performs the second-stage prediction-level fusion analysis using only the frozen private payloads from `research/siren-matcha-ab-fusion-v1` at commit `2e151f9ec89e596469ba315eebcc312e44d20044`.

## Frozen inputs and prohibitions

- Input payloads: `siren_matcha_ab_v1/fused/holdout_{I0002,I0006,S0001}/payload.npz`.
- No A or Matcha retraining, feature/embedding/prediction regeneration, raw EDF access, chronology, residual, demographic, calibration, stacking, probability/logit fusion, or post-hoc reranking is permitted.
- The parent payload SHA256 is checked against the previous manifest before any analysis and checked again after artifact generation.

## Candidate and score definitions

`wB` is Matcha B's weight and the only candidate grid is `[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]`. For every record, the candidate score is `S(wB) = (1 - wB) * rA + wB * rB`. Thus `wB=0.4` is the frozen AB60 control and `wB=0.5` is the frozen AB50 control. No finer or adaptive search is performed.

## Cross-fold selection

For each target holdout, the candidate maximizing the mean age-conditioned AUROC over the other two holdouts is selected. The target holdout is excluded from both the observed selection score and nested-bootstrap selection. Ties within `1e-12` use the fixed rule: closest to `0.4`, then lower `wB`.

## Bootstrap

The nested bootstrap uses 2,000 independent record-level multiplicity draws per site with seed `20261023`. A replicate selects its target weight from the other two bootstrapped sites and then evaluates the selected weight on the target's bootstrap sample. The same replicate index is used for all routes. `FIXED_SELECTED_BOOTSTRAP` is an auxiliary analysis that holds the observed cross-fold selected weights fixed. The primary analysis is `NESTED_CROSSFOLD_BOOTSTRAP`.

## Privacy and artifacts

Public CSV/JSON/report files contain only aggregate metrics, candidate weights, selection audit values, and bootstrap summaries. Record IDs, labels, ages, individual scores, and bootstrap multiplicities remain under the private root: `/database/home/gaohaojie/workspace/feature-cache-extension/siren_matcha_ab_crossfold_v1`.
