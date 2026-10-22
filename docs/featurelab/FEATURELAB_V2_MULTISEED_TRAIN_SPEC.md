# FeatureLab v2 multi-seed training specification

This document freezes the authorized extension from Seed 42 to Seeds 43--56.
It consumes the six completed Rank256 v2 units and does not rebuild features,
Rank256 arrays, the old baseline, or Branch A artifacts.

## Fixed contract

The new seeds are exactly `43..56`; the frozen Seed 42 v2 outputs are read-only.
Views are `V1..V5` with widths `161/177/493/257/376`.  The fixed LOSO roles are
`I0002 <- S0001`, `I0006 <- S0001`, and `S0001 <- I0006`; the unused site is
never loaded as a training role.  Both `transductive` and `inductive` inputs
share the same training role for a given holdout.

Each seed/site sample is produced by the frozen G4 sampler and each LambdaRank
query by the frozen G4 query builder.  LR fits its adapter on that seed's 70%
sample; HGB and LambdaRank retain native NaN values.  Model parameters are the
frozen G4-B dictionaries with only `random_state=seed` (and the query-derived
LambdaRank truncation level).

Model units are keyed by source version, train site, seed, view and model type,
plus input, sample, query and feature-order identities.  Prediction units add
holdout, protocol, model completion and test-index identities.  Unit publication
is atomic and no-clobber: a complete identity is `SKIP_VERIFIED`, an incomplete
or mismatched unit stops rather than being replaced.

Training code never materializes evaluation labels or ages.  Evaluation targets
are opened only by the later frozen comparison/scoring stage, after every
prediction reference and fusion vector is fixed.

## Expected closure

The new stage contains 28 sample/query bridges, 420 models and 1260 prediction
units.  Together with the read-only Seed 42 v2 outputs this yields 450 models
and 1350 predictions.  No score is used for seed/model selection, calibration,
weight optimization or hyperparameter search.
