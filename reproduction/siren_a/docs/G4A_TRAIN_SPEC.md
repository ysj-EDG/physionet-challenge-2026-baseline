# G4-A Frozen Baseline v1 Training Specification

## Scope

G4-A trains only Seed 42 for SIREN-inspired Branch A. It reuses the frozen G3-B `FoldInputs`, fits 30 unique models (two train sites × five views × three model families), and emits 90 raw prediction units (three holdouts × two normalization protocols × five views × three model families). No holdout labels or ages are loaded, no evaluation metric is computed, and Seeds 43–56 are not run.

## Environment and provenance

Every command uses `/database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python` with `PYTHONNOUSERSITE=1`, `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`. Frozen execution versions are Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, joblib 1.5.3, and LightGBM 4.6.0. The environment audit and wheel fingerprint remain private.

The run identity binds the G2-B1, G3-A, G3-A-V, and G3-B identities; Frozen Manifest and Spec SHA256; demographics SHA256; G4 source hashes; package versions; environment dependency fingerprint; and thread settings. Any drift is `STOP_INVALID`.

## Paper-derived and reconstructed rules

Paper-derived elements are five views, LR/HGB/LambdaRank model families, 70% training subsets, 15 seeds, and five-year age queries. Seed derivation, constrained quota allocation, exact model hyperparameters, query sorting, checkpoint format, and recovery behavior are project Baseline v1 engineering choices and are not claimed as the paper's unpublished implementation.

## Sampling

The only executed seed is 42. For train size `N`, `target_n=(7*N+5)//10`. Strata are `(floor(age/5), label)`. Initial quota is `max(1, floor(7*n_s/10))`; remaining places are allocated by descending `(7*n_s) mod 10`, then ascending `(age_group,label)`, without exceeding capacity. Infeasible allocations fail.

Each stratum uses a separate `numpy.random.Generator(PCG64(seed64))`, where `seed64` is the little-endian unsigned integer encoded by the first eight bytes of SHA256 over `siren-g4a-v1|train_site|seed|age_group|label`. Selection is without replacement, merged selections are restored to original G3-B train order, and local positions plus global G2 indices are saved privately. S0001 must yield 3597 rows; I0006 must yield 799.

## LambdaRank queries

Only selected training rows are used. Rows are stably sorted by `floor(age/5)`, yielding contiguous `group_sizes`. Every query must contain both labels, `sum(group_sizes)` must equal selected N, and `lambdarank_truncation_level=max(group_sizes)`. The query permutation, sizes, class counts, selected indices, and fingerprint are private. AC-pair coverage is the fraction of all opposite-label pairs within two years that fall in the same five-year query; it is diagnostic only.

## Frozen models

LR uses L2, `C=1`, lbfgs, `max_iter=2000`, `tol=1e-4`, intercept, no class weight, and random state 42. Its G3-B adapter is refit only on the selected Seed 42 rows with `fit_scope=seed_subset`; all-missing columns use `128/255`.

HGB uses log loss, learning rate 0.05, 250 iterations, 15 leaf nodes, depth 3, minimum leaf 30, L2 1, 255 bins, no early stopping or class weight, and random state 42. Native NaNs and all columns are retained.

LambdaRank uses objective lambdarank, 250 estimators, learning rate 0.05, 15 leaves, depth 3, minimum child 30, L2 1, full column and row sampling, `label_gain=[0,1]`, normalization, random state 42, four jobs, quiet output, deterministic column-wise histograms, and the dynamic truncation level. Native NaNs are retained.

## Model reuse and prediction

Model identity is `(train_site, seed, view, model_type)`. The S0001 model serves I0002 and I0006; I0006 serves S0001. Transductive and inductive training inputs must be identical before sharing. LR/HGB output the probability for class 1; LambdaRank outputs raw ranking scores. Prediction identity adds `(holdout_site, protocol)`. Test labels and ages are never requested or stored.

## Private artifacts and recovery

Private output root is `/database/home/gaohaojie/workspace/python-example-2026-siren-a-g4a-private/seed42_smoke_v1`. Samplers, queries, models, and predictions are completion-last no-clobber units. A unit is skipped only after source identity, file hashes, schemas, and numerical replay validate. Existing invalid units stop the run; they are never overwritten or deleted. Models are reloaded and predictions must agree within `1e-12`.

## Gate conditions

PASS requires 2 verified samplers, 2 valid query sets, 30 nonconstant verified models, 90 finite nonconstant prediction units, exact reload agreement, source closure, resume without retraining, privacy-safe public aggregates, and all required tests. No LOSO metric, calibration, ensemble, or later seed is permitted.
