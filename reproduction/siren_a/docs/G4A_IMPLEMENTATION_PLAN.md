# G4-A Frozen Seed42 Implementation Plan

**Goal:** Build and verify the frozen Seed 42 Branch A training smoke without reading holdout outcomes.

**Architecture:** Small pure modules freeze configuration, sampling, and query math; a model module owns estimator construction and trusted serialization; the runner composes existing G3-B inputs into independently completed sampler, query, model, and prediction units; an independent verifier reloads and replays all artifacts.

**Tech Stack:** Python 3.10, NumPy 2.2.6, scikit-learn 1.7.2, joblib 1.5.3, LightGBM 4.6.0, unittest.

**Spec:** `reproduction/siren_a/docs/G4A_TRAIN_SPEC.md`

## Global constraints

- Use only Seed 42 and the frozen 70% quota algorithm.
- Use only G3-B label-safe `FoldInputs`; never load holdout labels or ages.
- Fit 30 unique models and generate 90 raw prediction units.
- Use the absolute `sleepfm_env` Python with user site disabled and frozen thread variables.
- Do not modify tracked G1/G2/G3 files or private upstream artifacts.
- No hyperparameter search, evaluation metric, ensemble, or Seeds 43–56.

## Review focus

- Same train site across folds/protocols must be byte-identical before model sharing.
- Every nonempty age×label stratum must survive exact target allocation.
- Query ordering, group sizes, and labels must remain aligned after stable sorting.
- LR adapter must be fit only on selected Seed 42 rows; native-NaN models must not impute.
- Completion hashes alone are insufficient: verifier must reload models and replay predictions.

## Task 1 — Frozen configuration and sampler

Create `config.py`, `sampler.py`, and `test_sampler.py`. Tests first cover target rounding, quota closure/ties, infeasible constraints, deterministic SHA-derived PCG64 sampling, no replacement, original-order restoration, and shared indices. Then implement immutable constants and `sample_training_rows(...)` returning local/global indices, quotas, and fingerprints.

## Task 2 — LambdaRank query contract

Create `queries.py` and `test_queries.py`. Tests first cover five-year boundaries, stable sorting, class validity, reversible indices, dynamic truncation, and AC-pair coverage. Then implement `build_queries(...)` and `audit_ac_pair_coverage(...)` using selected training rows only.

## Task 3 — Frozen model API

Create `models.py` and `test_models.py`. Tests first assert exact hyperparameters, selected-subset LR adapter behavior, native NaNs, probability class direction, raw rank scores, trusted serialization, reload equality, and corruption refusal. Then implement estimator factories, fit/predict helpers, and checkpoint save/load.

## Task 4 — Runner and artifact units

Create `run_g4a.py` and `test_run_g4a.py`. Tests first cover upstream identity closure, exact train-input sharing, no-clobber, source drift, first-six ordering, 30 unique model identities, 90 prediction identities, no holdout outcomes, and resume actions. Then implement preflight, sampler/query publication, model/prediction units, public aggregation, and CLI modes.

## Task 5 — Independent verifier

Create `verify_g4a.py` and `test_verify_g4a.py`. Tests first inject bad hashes, parameters, feature order, checkpoint corruption, prediction corruption, and incomplete units. Then implement sampler recomputation, query reconstruction, trusted model reload, prediction replay, and full 2/2/30/90 closure.

## Task 6 — Real execution

Run new tests RED→GREEN, then G3-B input tests and light historical regressions. Execute preflight, build the six V1 models, continue the remaining 24 only after verification, generate 90 predictions, run independent verify-only, and rerun resume expecting all units skipped without changed SHA256.

## Task 7 — Reporting and Git

Generate the private run report, 90-row public `g4a_smoke_summary.csv`, and `G4A_TRAIN_REPORT.md`. Run syntax checks, privacy scan, approved-file diff check, and fresh tests. Explicitly stage only 16 approved files, commit with the frozen message, push normally, verify local/remote SHA, and stop.
