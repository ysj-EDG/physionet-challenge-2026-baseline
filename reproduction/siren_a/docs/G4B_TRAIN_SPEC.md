# G4-B Full Multi-Seed Training Specification

## Scope

G4-B extends the reviewed G4-A Seed 42 baseline with Seeds 43 through 56. It builds exactly 420 new models and 1,260 new raw prediction units, independently verifies their sampling, query, adapter, checkpoint, and prediction semantics, then creates a reference-only G5-ready index closing 450 models and 1,350 predictions across all 15 seeds. It never retrains or modifies Seed 42, reads holdout outcomes, evaluates LOSO performance, or performs an ensemble.

## Frozen environment and sources

Every command uses `/database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python` with `PYTHONNOUSERSITE=1`, `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1` set before Python starts. Versions are Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, joblib 1.5.3, and LightGBM 4.6.0.

The run identity binds the G2-B1 fingerprint, G3-A aggregate fingerprint, G3-A-V completion, G3-B run and completion identities, G4-A run identity, Frozen Manifest and Spec SHA256, demographics SHA256, environment dependency fingerprint, G4-B code SHA256, thread configuration, seed schedule, sampler version, and cache schema. Any drift is `STOP_INVALID`.

## Seed schedule and identities

G4-B new seeds are exactly `43..56`. A model identity is `(train_site, seed, view, model_type)` and a prediction identity is `(holdout_site, protocol, view, model_type, seed)`. Each seed has 30 models and 90 predictions. Seed 42 remains referenced from G4-A. Models trained on S0001 serve I0002 and I0006; models trained on I0006 serve S0001. The training arrays, masks, indices, labels, ages, groups, and feature IDs must be exactly equal before sharing.

## Sampling and queries

Sampling calls the frozen G4-A `sample_training_rows` without copying or changing its algorithm. The SHA-derived PCG64 prefix remains `siren-g4a-v1`. S0001 selects 3,597 of 5,139 and I0006 selects 799 of 1,142 for every seed. Every nonempty `(floor(age/5), label)` stratum retains at least one row, quota closure is exact, and selections are without replacement and restored to original train order.

Production queries use the frozen G4-A `build_queries` on selected rows. Each five-year group must contain both labels; rows are stably sorted by age group and `lambdarank_truncation_level=max(group_sizes)`. Query artifacts save permutation, inverse permutation, sorted local/global indices, sorted groups/labels, group IDs/sizes/class counts, truncation, fingerprint, and AC-pair diagnostics.

Independent verification does not call `build_queries`: it reconstructs each array from selected ages, labels, and indices using independent NumPy operations and compares dtype, shape, and values. This applies to both Seed 42 G4-A queries and all new G4-B queries.

## Frozen models

LR, HGB, and LambdaRank retain all G4-A Baseline v1 parameters. A thin G4-B factory copies the frozen parameter dictionaries and changes only `random_state` to the current seed; LambdaRank additionally sets the current query truncation. G4-A constants are never mutated.

LR refits the G3-B median adapter on the actual selected seed subset with `fit_scope=seed_subset`; all-missing columns use `128/255`. HGB and LambdaRank preserve native NaNs. Independent adapter verification recomputes finite counts, medians, all-missing flags, fit-index SHA, and feature-order SHA without calling `fit_adapter`, and confirms transformed values. Seed 42's ten LR adapters and native-NaN adapters receive the same read-only audit.

Each model is fitted on the selected rows only, emits finite nonconstant training scores, is saved, reloaded, and replayed with maximum absolute difference at most `1e-12`. LR/HGB scores are class-1 probabilities; LambdaRank scores are raw ranking values. Metadata records actual parameters and loaded-estimator semantic checks.

## Predictions and privacy

Each model predicts the corresponding frozen G3-A test matrix for every applicable holdout and protocol. Test index order is preserved. Saved float64 predictions are finite, nonconstant, and independently replayed from the checkpoint and adapter within `1e-12`. Holdout labels and ages are never requested or stored. No calibration, ranking transform, ensemble, or performance metric is permitted.

## Artifacts and recovery

The private root is `/database/home/gaohaojie/workspace/python-example-2026-siren-a-g4b-private/seeds43_56_v1`. Seed 42 bridge evidence, each sample, query, model, prediction, seed, G5 manifest, and root are completion-last no-clobber units. Existing verified units are `SKIP_VERIFIED`; absent units are `BUILD_NEW`; invalid or incomplete units are `STOP_INVALID`. No failed artifact is overwritten or deleted.

The G5-ready manifests reference, rather than copy, Seed 42 and G4-B files. They contain exactly 450 unique model identities and 1,350 unique prediction identities; each holdout/protocol has 225 score vectors with identical test-index fingerprints. They perform no score transformation.

## Gate

PASS requires a verified Seed 42 bridge, 28 new samplers and queries, independent query-array and adapter semantics for all 15 seeds, 420 new models, 1,260 new predictions, 450/1,350 manifest closure, clean resume with no new fits or predictions, source and privacy closure, and all required tests. G5 remains not started.
