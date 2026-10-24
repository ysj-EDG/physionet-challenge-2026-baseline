# CAISR20 Historical Semantics vs CORE — V2 Report

## Status

`ROUTE_LEVEL_COMPARISON_V2_STATUS = COMPLETE_AFTER_WORST_SITE_DIAGNOSTIC_FIX`

V1 remains unchanged. Its 8,059 old-V5 validity mismatch is not a numerical error: old-V5 records source-support validity, while this route uses historical typed-v1 validity/fill semantics. The finite-overlap numerical bridge has 0 mismatches.

## Gates and execution

| Gate | Result | Evidence |
|---|---|---|
| Historical seed-7 bridge | PASS | max absolute prediction difference `4.440892098500626e-16`; 0 convergence warnings |
| Historical typed-v1 pipeline bridge | PASS 3/3 | 6,600 × 20 per fold; max absolute difference `0.0` |
| Frozen sample identity | PASS 30/30 | existing frozen sample artifacts, no resampling |
| Formal models | PASS 30/30 | all converged before `max_iter` |
| Formal raw predictions | PASS 45/45 | finite decision scores, frozen row order |
| Resume/no-refit | PASS | second run: 30 model skips and 45 prediction skips |
| Checkpoint replay | PASS | all 45 prediction units replayed with max absolute difference `0.0` |

## Source policy diagnostics

`FINITE_OVERLAP_VALUE_MISMATCH_TOTAL = 0`. `OLD_V5_VALIDITY_MISMATCH_TOTAL = 8059`, diagnostic/non-blocking. The largest component is the historical entropy column: old-V5 source-valid count 7 versus 6,600 finite DATA2 values. The old candidate pool was not modified.

## Route-level scores

The public site rows are in `audit/caisr20_core_protocol_v2/caisr20_vs_core_v2_site_scores.csv`. Aggregate AC-AUROC values are:

| Scope | CAISR20 | CORE | CORE − CAISR |
|---|---:|---:|---:|
| I0002 | 0.7343584305 | 0.6961823966 | -0.0381760339 |
| I0006 | 0.6571448285 | 0.6786724626 | 0.0215276340 |
| S0001 | 0.6116437229 | 0.6709153732 | 0.0592716503 |
| Macro | 0.6677156607 | 0.6819234108 | 0.0142077501 |
| Worst-site | 0.6116437229 | 0.6709153732 | 0.0592716503 |
| Small-site mean | 0.6957516295 | 0.6874274296 | -0.0083242000 |

The complete aggregate outputs include AUROC/AUPRC, pair counts, score-uniqueness counts, 15-seed paired stability, 2,000-record bootstrap, and site-level Pearson/Spearman correlations.

## Targeted Worst-site diagnostic correction

The original V2 route summary and site-level scores were correct. A diagnostic aggregation bug affected only the Worst-site rows in seed stability and paired bootstrap: the implementation used the minimum of site-level route differences instead of the difference between the two route-level minimum site scores. Macro, Small-site mean, individual-site diagnostics, models, predictions, and score correlations were unaffected.

The corrected semantics are `min(CORE site scores) - min(CAISR site scores)`. The formal Worst-site delta is `+0.059271650304393075`; the corrected seed-stability summary is in the public seed-stability CSV, and the corrected 2,000-draw paired bootstrap is in the public bootstrap CSV. Both public mirrors (the directory and flat audit paths) are synchronized.

## Limitations and next step

This is a complete route-level comparison, not a feature-only comparison: the validity contracts differ by design. Site heterogeneity is visible in the sign change between I0002 and S0001. Final CAISR and CORE scores are correlated but not identical (public correlation file); this supports only a future fixed prediction-level fusion design review, not an unapproved fusion.

`TARGETED_TEST = PASS (12/12); FROZEN_G1B_CONTRACT_TEST = PASS (13/13); EXTENSIONS_TEST = NOT_COMPLETED (bounded 120-second run timed out in existing long-running tests)`.

`CORE_REFIT_COUNT = 0`; feature re-extraction, Rank256 rebuild, route fusion, STRICT_CAISR20 sensitivity, compact30 retraining, and Batch2 implementation were not performed.
