# FeatureLab v2 Seed 42 Training Specification

## Scope

This specification freezes the v2 Seed 42 input and training bridge. It consumes the six published `featurelab_v2_rank256_v1` units read-only and does not rebuild the candidate pool or Rank256 references. The v2 view widths are V1=161, V2=177, V3=493, V4=257 and V5=376. The old G3-B/G4-A implementation remains unchanged.

Authorized outputs are 30 models (two training sites × five v2 views × three frozen model types), 90 raw prediction units (three holdouts × two protocols × five views × three models), and a reference-only E0–E3 manifest. Seeds 43–56, scoring, fusion, feature selection and Branch A migration are outside this specification.

## Input contract and role isolation

`extensions.train_inputs_v2.load_v2_fold()` validates the Rank256 completion marker, metadata identity, v2 view manifest, global `unique_column_index` mapping, frozen LOSO partitions, dtypes, shapes and Rank256 NaN semantics. Only `quantized` view values and `rank_valid_mask` are model inputs; percentile, level, source-mask, support and missing-reason arrays are not model features.

`V2FoldInputs` exposes `y_train`, `age_train` and `train_age_groups` only for the requested training rows. It intentionally has no `y_test`, `age_test` or unused-role target attributes. Labels and ages are parsed through the audited metadata join only after exact train indices are checked. Patient-overlap checks and site partitions are applied before target parsing. The loader never constructs a global target vector.

The frozen folds are:

| Holdout | Train | Test | Unused |
|---|---|---:|---|
| I0002 | S0001 | 319 | I0006 |
| I0006 | S0001 | 1142 | I0002 |
| S0001 | I0006 | 5139 | I0002 |

Transductive and inductive folds must have identical training rows, values, masks, labels, ages and feature order. Their test matrices remain protocol-specific.

## Seed 42 sampler, query and adapters

`extensions.train_seed42_v2.prepare_seed42_training()` calls the frozen `sample_training_rows()` with seed 42, five-year age groups and the original 70% constrained quotas. The selected sample sizes are 3597 for S0001 and 799 for I0006. The frozen `build_queries()` receives only selected training rows.

LR uses the frozen adapter fitted on the selected subset with `fit_scope=seed_subset`; finite-column medians and the 128/255 all-missing fallback are unchanged. HGB and LambdaRank receive float64 NaNs natively. Valid numeric zeros remain zeros. Test matrices are transformed only and never fit an adapter.

## Models and private publication

`extensions.seed42_runner_v2` reuses `fit_frozen_model`, `predict_scores`, `save_checkpoint` and `load_checkpoint`. The complete frozen parameter dictionaries are imported from `reproduction.siren_a.g4.config`; the only dynamic LambdaRank parameter is the query-derived truncation level. LR/HGB scores are positive-class probabilities; LambdaRank scores are raw ranking scores.

Each model identity includes the v2 schema, execution identity, train site, seed, view, model type, sample fingerprint, query fingerprint and feature-ID fingerprint. Each prediction identity additionally includes holdout, protocol, model completion fingerprint and test-index fingerprint. Units publish through temporary same-filesystem directories and no-clobber completion records. An identity-matching complete unit returns `SKIP_VERIFIED`; an incomplete, corrupt or mismatched unit raises an error and is never replaced.

The private output root is `/database/home/gaohaojie/workspace/feature-cache-extension/train_v2/seed42`. It is not a Git artifact. Patient-level values, labels, ages, row indices, checkpoints and predictions remain private.

## E0–E3 reference-only manifest

The reference manifest contains identities and source paths only; it does not train, score, fuse or calibrate anything:

- E0: old V1–V5.
- E1: v2 V1–V4 plus old V5.
- E2: old V1–V4 plus v2 V5.
- E3: v2 V1–V5.

The manifest is a comparison-input definition for a future authorized evaluation.
