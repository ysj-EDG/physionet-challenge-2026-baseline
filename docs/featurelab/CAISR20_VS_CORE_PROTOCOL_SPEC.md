# CAISR20 vs FeatureLab CORE — 同协议 Route-Level 对照

## Scope

`CAISR20_CORE_PROTOCOL` is a route-level comparison against the frozen
`CORE_COH_EMG` inductive-feature protocol. It is not a feature-only ablation
and it does not alter CORE, E0/E2, V5 LOO, Rank256, the candidate pool, or
Branch A.

The route is frozen to the historical pure 20-dimensional CAISR set, with no
demographic columns:

1. `caisr_sleep_tst_sec`
2. `caisr_sleep_se`
3. `caisr_sleep_sol_sec`
4. `caisr_sleep_rem_latency_sec`
5. `caisr_sleep_waso_sec`
6. `caisr_sleep_n1_pct`
7. `caisr_sleep_n2_pct`
8. `caisr_sleep_n3_pct`
9. `caisr_sleep_rem_pct`
10. `caisr_sleep_transition_rate`
11. `caisr_sleep_short_bout_ratio`
12. `caisr_sleep_mean_stage_entropy`
13. `caisr_arousal_arousal_index`
14. `caisr_arousal_arousal_burden_ratio`
15. `caisr_arousal_mean_arousal_duration`
16. `caisr_respiratory_respiratory_event_index`
17. `caisr_respiratory_respiratory_burden_ratio`
18. `caisr_respiratory_mean_resp_duration`
19. `caisr_limb_limb_movement_index`
20. `caisr_limb_plmi`

The corresponding `x_static` indices are
`[11,12,13,14,16,17,18,19,20,32,34,87,96,97,98,126,127,141,168,173]`.

## Frozen route contract

- Holdout-to-training mapping: `I0002 -> S0001`, `I0006 -> S0001`,
  `S0001 -> I0006`.
- Seeds: 42 through 56.
- Two training sites × 15 seeds = 30 models; three holdouts × 15 seeds = 45
  raw decision-score vectors.
- Model: historical elastic-net logistic regression
  (`solver=saga`, `l1_ratio=0.4`, `C=0.03`, balanced classes,
  `max_iter=100000`, `tol=1e-4`), with only `random_state` varied by seed.
- Preprocessing: frozen `typed_v1` selected-column transform, followed by
  median imputation and linear scaling fit only on the frozen selected training
  rows. No holdout or unused-site rows enter preprocessing.
- Final route score: average-tie percentile rank of all 15 seed decision
  vectors within each holdout.
- Primary comparator: existing `CORE_COH_EMG` inductive-feature result; no
  CORE refit, transductive comparison, or route fusion.

## Safety gates

The historical seed-7 pooled CAISR20 bridge must reproduce the archived
decision vectors before a new route is allowed to run. The 6600-record
inventory and all 30 frozen sample identities must then pass. Finally, raw
CAISR values and validity positions are compared feature-by-feature with the
immutable old-V5 source. Any value or validity drift is a hard stop; it cannot
be repaired by changing a feature definition or imputing a different value.

Private model, prediction, checkpoint, and patient-level diagnostics belong
under the private cache root and are never committed. Public artifacts may
contain only feature-contract metadata and aggregate gate/results summaries.

## Current implementation status

The contract and numerical helpers are implemented in
`per_epoch_features/extensions/caisr20_core_protocol.py`; the bounded runner
and no-clobber private-unit writer are in
`per_epoch_features/extensions/caisr20_core_protocol_runner.py`.

The current run is intentionally blocked at the source-value gate because
the DATA2 source has different validity positions from the immutable old-V5
source. No CAISR20 model, prediction, final fusion, bootstrap, or correlation
artifact is therefore validly produced in this revision.
