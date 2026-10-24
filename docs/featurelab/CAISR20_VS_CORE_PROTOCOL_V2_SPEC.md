# CAISR20 Historical Semantics vs CORE — Route-Level Continuation V2

## Scope

This V2 route is a frozen, aggregate-only comparison of a 20-feature CAISR route against the read-only `CORE_COH_EMG` inductive comparator. It does not modify Branch A, the old V5 validity mask, candidate-pool arrays, Rank256, CORE units, E0/E2, LOO outputs, or historical DATA2 artifacts.

## Frozen source bridge

The source bridge has three independent contracts:

1. **Identity bridge:** 6,600 rows, site order, record order, and the 20 frozen feature IDs must match exactly. Failure is a route stop.
2. **Finite-overlap numerical bridge:** finite old-V5 and DATA2 values are compared at absolute tolerance `1e-12`; any mismatch is a route stop.
3. **Validity-policy diagnostic:** old-V5 source-support validity is compared with historical typed-v1 validity/fill semantics and CAISR sentinel handling. A mismatch is diagnostic only (`OLD_V5_VALIDITY_MISMATCH_IS_BLOCKING = NO`). It is not a claim that either policy is wrong.

The historical feature order is the frozen 20-ID list in `caisr20_core_protocol.py`; no demographic columns are included.

## Historical typed_v1 bridge

The first stage consumes complete `x_static[196]` so the documented CAISR sentinel can be recovered exactly. Each selected training row is processed through `feature_scaling._prepare_values`, including missing-rule, hard-range tolerance, unit scaling, and `log1p`. For robust rules, selected-row weighted quantiles use the historical q05/q25/q50/q75/q95 procedure and scale priority IQR → p95-p05 → weighted std → unit fallback. Identity rules use center 0 and scale 1. A sentinel row forces CAISR indices ≥10 missing; robust missing values fill with the fitted center and identity missing values use `identity_fill` (default 0). The result must be finite.

The second stage is the frozen selected-row P3 median/mean/std scaler (`std <= 0` → 1). Both stages are fit only on the frozen selected training rows; neither the unused site nor holdout rows enter fitting.

Before formal training, the archived typed-v1 parameters are replayed for all three historical folds and compared with `p3_{holdout}_compact30[:,10:30]` at `1e-12`.

## Formal route

After Gate A (historical seed-7 bridge), Gate B (3/3 historical typed matrices), and selected-row unit tests pass, the route trains 2 train sites × 15 seeds = 30 ElasticNet logistic models and emits 3 holdouts × 15 seeds = 45 raw decision-score vectors. Each holdout fuses by within-holdout average-tie percentile rank per seed, followed by equal mean over 15 seeds. No probability calibration, thresholding, seed weighting, CORE+CAISR fusion, strict validity sensitivity, compact30 retraining, or feature search is performed.

## Evaluation and privacy

Holdout targets are loaded only after model, prediction, and three fused-vector completion. Site metrics use the frozen official AC-AUROC, AUROC, and AUPRC implementation. Public files contain only aggregate site/route metrics, stability summaries, bootstrap summaries, and correlations; raw scores, labels, ages, record IDs, patient IDs, and private paths remain outside Git.
