# FeatureLab v2 E0--E3 comparison specification

After multi-seed closure, the old and v2 prediction artifacts are referenced
without re-inference.  E0 uses old views, E1 uses v2 V1--V4 and old V5, E2 uses
old V1--V4 and v2 V5, and E3 uses v2 for all five views.

Every experiment has 3 holdouts × 2 protocols × 15 seeds × 5 views × 3 models
= 1350 references; each holdout/protocol fusion has exactly 225 columns in
`seed -> view -> model` order.  Old `raw_ranking_score` is explicitly bridged
to the v2 `lambdarank_raw` semantic; unknown score types are rejected.

For each complete prediction vector, the frozen G5 `mean_ranks` function first
computes average-tie percentile ranks within the holdout cohort and then takes
the equal mean across all 225 columns.  Raw scores are never averaged before
ranking and the fused result is not reranked.

Only after all 24 fusion vectors and the E0 private vector regression are fixed
may the official G5 scoring functions read holdout labels and ages.  Public
outputs contain only aggregate site/protocol metrics and reference counts;
patient-level vectors, indices, labels, ages, checkpoints and private paths
remain outside Git.
