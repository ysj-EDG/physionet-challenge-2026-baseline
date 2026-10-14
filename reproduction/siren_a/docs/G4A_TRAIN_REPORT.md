# G4-A Frozen Baseline v1 Seed 42 Training Report

`G4A_ENV_STATUS = PASS`

`G4A_STATUS = SUBMITTED_FOR_REVIEW`

## Scope and outcome

The frozen Seed 42 smoke completed without holdout outcomes, model selection, calibration, ensemble construction, or LOSO metric calculation. Two shared training sources produced 30/30 unique models, and those checkpoints produced 90/90 raw prediction units for three holdouts, two Rank256 protocols, five views, and three model families. Independent checkpoint reload and prediction replay passed for every unit.

This is a SIREN-inspired Baseline v1 implementation. Five views, three learner families, 70% subsampling, multiple seeds, and five-year age queries are paper-derived design elements; the exact quota allocation, random-seed derivation, model hyperparameters, serialization, and recovery protocol are project engineering choices.

## Environment recovery

All G4-A commands used `/database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python` with `PYTHONNOUSERSITE=1`. The fixed thread environment was `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `NUMEXPR_NUM_THREADS=1`.

| Component | Frozen execution version |
|---|---:|
| Python | 3.10.20 |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| joblib | 1.5.3 |
| LightGBM | 4.6.0 |

The LightGBM wheel was `lightgbm-4.6.0-py3-none-manylinux_2_28_x86_64.whl`, SHA256 `cb19b5afea55b5b61cbb2131095f50538bd608a00655f23ad5d25ae3e3bf1c8d`. Synthetic LR, HGB, and LambdaRank API/serialization checks passed before real training. The environment dependency fingerprint recorded by the run is `fff581a4b81945874a1c6f602b641972c7def918d74f4c34102d549e2623509d`. Package imports resolved inside `sleepfm_env`; user-site packages were disabled. The Conda base environment was not modified. Existing unrelated `pip check` findings for pooch/Jupyter remained unchanged and did not affect the required model APIs.

## Frozen sources

| Source | Verified identity |
|---|---|
| G2-B1 cache | `cf88b4e87f8b524258c685db696c278198dae1e6a10bd39fb63e893b2df031df` |
| G3-A artifacts | `4a0f3956432030e4fa2fb8b51d15aa8fa5269f990d7795cca40d580970920a64` |
| G3-A-V | six-unit independent verification PASS |
| G3-B run identity | `0936a4fcd91113f4ee76caa69c5cdfd3b501db87aa88b718d7a8b02ffccaf46d` |
| G3-B completion | `cd218356f047e39aee7950c2dab8c549616b8abaf075093bd8296e01b8320b57` |
| Demographics CSV | `b9009598589846e84462636bfcaa290ed1d8666c89b1bc1ad1a9eda26b43ff7f` |
| Frozen Manifest | `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a` |
| Frozen Spec | `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a` |
| G4-A run identity | `4d9654cc36142b23aaa058a82a0b154f8b5d9557feb1c506354374138f13945c` |

No G1, G2, or G3 tracked file or private upstream artifact was modified.

## Sampling

The exact half-up 70% target, constrained age-group-by-label quotas, SHA256-derived per-stratum PCG64 streams, no replacement, and original train-order restoration all passed recomputation.

| Train site | Full N | Target/selected | Positive | Negative | Nonempty strata | Sampling SHA256 |
|---|---:|---:|---:|---:|---:|---|
| S0001 | 5,139 | 3,597 | 235 | 3,362 | 18 | `7421a2495ea13080c8579f887e927745903370e45a1b79b2d5876280939e0e36` |
| I0006 | 1,142 | 799 | 78 | 721 | 18 | `c34ac640805e5ec28757f776959c46527018a7fcfcb9e22ab19fb52393bad318` |

Every nonempty stratum retained at least one record, all quotas closed exactly, and the same site/seed indices were reused across views, models, protocols, and applicable holdouts. Patient-level indices remain private.

## LambdaRank queries

| Train site | Queries | Query sizes | Maximum / truncation | All AC-valid pairs | Same-query pairs | Pair coverage | Query SHA256 |
|---|---:|---|---:|---:|---:|---:|---|
| S0001 | 9 | 875, 796, 646, 576, 383, 202, 94, 22, 3 | 875 | 89,802 | 68,219 | 0.759660 | `c13bfaf9abcb5ab0ddb740b34c20ade1c26fa907f78f5fefc90ad9b49744cdb5` |
| I0006 | 9 | 155, 170, 149, 153, 106, 40, 18, 6, 2 | 170 | 6,929 | 5,395 | 0.778612 | `97d59c969ee58b929b7479f8a24760980faebc3e5d8d384a270d20ec4036b4d2` |

Stable five-year grouping was used. Every query contained both relevance labels, group sizes summed to selected N, and age was not added to any feature matrix.

## Frozen models and checkpoints

LR used the Seed-specific selected-subset median adapter; HGB and LambdaRank retained native NaNs. The frozen parameters in `G4A_TRAIN_SPEC.md` were verified against every saved model metadata record. LR and HGB scores are class-1 probabilities; LambdaRank scores are uncalibrated raw ranking outputs.

| Train site | View | D | Model | Fit seconds | Iterations/trees | Checkpoint SHA256 |
|---|---|---:|---|---:|---:|---|
| I0006 | V1 | 113 | LR | 0.064 | 51 | `e34ea27fae12011f35a525ad97047a0dea46c251cd55707948453ec5a698da32` |
| I0006 | V1 | 113 | HGB | 3.719 | 250 | `dc2a84c31cd9598565066fdf8702070b8cc771d85ee711a55c7de55fc1f078e9` |
| I0006 | V1 | 113 | LambdaRank | 0.851 | 250 | `cc7114893beef7b3a646f358badde071ef7bb6cc476b357e28868635d62d4b1a` |
| I0006 | V2 | 129 | LR | 0.093 | 79 | `579c4461066f6fcd879f0c421d6f9aa1d46c8dcc1d99bf4ea1105a246eecc212` |
| I0006 | V2 | 129 | HGB | 2.258 | 250 | `0240f9b139603a0aa62a190ba4fcc74e3733854362bf41146ba2ffc26763d69d` |
| I0006 | V2 | 129 | LambdaRank | 5.181 | 250 | `9fdfcab517afaebeabfa96764dce2dc87357aed5c48f4869529048b1be522636` |
| I0006 | V3 | 349 | LR | 0.257 | 89 | `b753656de10b2a22ab3227af5c8c06e61a7da807e23a77002a446a9fa3b5ec34` |
| I0006 | V3 | 349 | HGB | 17.783 | 250 | `047694a645c55d01ab2f02180fef7051a6be96c3777a639c389433258e516b11` |
| I0006 | V3 | 349 | LambdaRank | 7.464 | 250 | `c187af686c4c9083b8722bbaa4aee2ca3ad5f34bc0406c2fa1cd067cdd9ad10b` |
| I0006 | V4 | 113 | LR | 0.100 | 74 | `278983871782588136a2be935915fecb08954323440775294cb24a196ec6b796` |
| I0006 | V4 | 113 | HGB | 5.407 | 250 | `9a5af653aa99d1df87e8a8c045293322083a649495d28c13d6343581e02e06c0` |
| I0006 | V4 | 113 | LambdaRank | 3.431 | 250 | `12c774cdf9a0dff4655391fb51b8dd8e36fa63dbc0a6a1a3cb38ad4a7164ccb1` |
| I0006 | V5 | 123 | LR | 0.096 | 82 | `a1f97785ba0dc633d4499e7beed225bcf7a9cd54dcd1e63044dfc7893f95ed50` |
| I0006 | V5 | 123 | HGB | 1.916 | 250 | `85a73e321da5a936b3f1d8ca6be373dd55a9b97f224397449737602fede0d8b4` |
| I0006 | V5 | 123 | LambdaRank | 0.778 | 250 | `355adcd08611bf9c93254c2c726c481692c6eb689b700edae2f9caa98df8dec2` |
| S0001 | V1 | 113 | LR | 0.160 | 60 | `778f63de8f8cfde28a53a4f9fcf82c9198a732b81d32b53b5b6aa1e8567766f3` |
| S0001 | V1 | 113 | HGB | 4.347 | 250 | `299d2f33fe5c8434997882f3aa53db87cb9c82351a0edbfc0f6394b517cc8ab5` |
| S0001 | V1 | 113 | LambdaRank | 5.142 | 250 | `894b97d72b9e4e8cdcace109469c6d849b94139f3257fcb8f5d21a7664b2b477` |
| S0001 | V2 | 129 | LR | 1.632 | 97 | `88829fc32d1aebd1f04aa079a57fb081a36ac200981969de6bb43098647599d5` |
| S0001 | V2 | 129 | HGB | 20.934 | 250 | `b00891e12d1bf8db458c2cc796a7d3c1e21bc0fc0e2c4bfc14863f13ec06bb1b` |
| S0001 | V2 | 129 | LambdaRank | 7.167 | 250 | `924cf66f77a297c50bf0e7eeb3f4190a973fb8ed2b2db2bdfa2d5d34d126929c` |
| S0001 | V3 | 349 | LR | 0.529 | 110 | `a4e8d72ba7312e9507a4aafe4b284d02b11f624486e667894746cfe79d596ba4` |
| S0001 | V3 | 349 | HGB | 99.857 | 250 | `3a0e4b9d46f9c4e940aa4d36e0168d02b40842c9889ae4169c644a0797115223` |
| S0001 | V3 | 349 | LambdaRank | 22.500 | 250 | `c63f9cc7fc30808d8bed6c5725b36070fafd0b5b3ce4b4354922528751467509` |
| S0001 | V4 | 113 | LR | 0.250 | 62 | `183ca446039f9afb6f33505ee093723cf3a2c98746209234cac89b2c224e056f` |
| S0001 | V4 | 113 | HGB | 16.573 | 250 | `c3dcf474a116c281999ed3a8eea0c39d04223c147ff084802fe5d36aef1860cf` |
| S0001 | V4 | 113 | LambdaRank | 14.540 | 250 | `175622b765de0b7bd63f4e4eea86299f976175fe3fa1106a5fd2e9c5579ff86d` |
| S0001 | V5 | 123 | LR | 0.296 | 87 | `e9f17e899cb02add22b511266328c9bc6137b0e222568f54b710abc60237eac1` |
| S0001 | V5 | 123 | HGB | 10.530 | 250 | `982413ef08fa6d16fcc91f185b1767c62809236e3bcf2da4fe1536cc6904d004` |
| S0001 | V5 | 123 | LambdaRank | 6.543 | 250 | `2c8d3944d8cdf6801f5db1cc0b5148f53794912b92dbed1ede56d4f60e2417a4` |

`Unique Models = 30/30`. Model fit time totaled 260.40 seconds; median was 3.57 seconds and the maximum was 99.86 seconds. All model reload comparisons had maximum absolute difference 0. Ten captured warnings were the benign LightGBM/sklearn message that NumPy inference arrays lacked feature names; there were no convergence warnings, failed fits, non-finite outputs, or strict-constant training outputs.

## Predictions, isolation, and reuse

`Prediction Units = 90/90`. Test row counts were 319 for I0002, 1,142 for I0006, and 5,139 for S0001. Every prediction array was finite and nonconstant; LR/HGB outputs were within the probability range, while LambdaRank outputs retained their raw-score scale. Saved predictions matched fresh inference from reloaded checkpoints exactly.

S0001's 15 models were shared across the I0002 and I0006 holdouts and both protocols; I0006's 15 models served the S0001 holdout and both protocols. Sharing occurred only after exact train-index, feature, mask, label, age-group, and feature-order checks. No holdout label or age was loaded by the training/prediction runner, and no test metric was computed.

## Recovery and integrity

The independent verifier returned PASS for 2 samplers, 2 query units, 30 models, and 90 predictions. The final Resume run reported:

- `models_built=0`, `models_skipped=30`
- `predictions_built=0`, `predictions_skipped=90`
- both sampler and query units `SKIP_VERIFIED`

No completed file was overwritten. The production build wall time was approximately 671 seconds across the first-six stage and the remaining build/prediction stage. Independent verification took 130.60 seconds; the final Resume audit took 250.34 seconds. Peak observed RSS was 372,104 KiB and swap use was zero. The private result root contains 406 files and occupies approximately 14 MiB.

## Tests and privacy

| Suite | Result |
|---|---:|
| G4-A new tests | 18/18 PASS |
| G3-B input-interface regression | 21/21 PASS |
| G3-A/G3-A-V lightweight regression | 49/49 PASS |
| G2 lightweight regression | 50/50 PASS |
| G1/audit lightweight regression | 22/22 PASS |
| Python syntax compilation | PASS |

The public CSV contains 90 aggregate rows plus its header. Public artifacts contain no patient identifiers, record paths, per-patient ages or labels, selected indices, or prediction vectors. Models, adapters, queries, sample indices, and predictions remain only under the private output root.

## Artifacts and gate decision

- Private environment evidence: `/database/home/gaohaojie/workspace/python-example-2026-siren-a-g4a-private/environment_sleepfm_lgbm460`
- Private training output: `/database/home/gaohaojie/workspace/python-example-2026-siren-a-g4a-private/seed42_smoke_v1`
- Public aggregate CSV: `reproduction/siren_a/audit/g4a_smoke_summary.csv`
- Public specification and plan: `reproduction/siren_a/docs/G4A_TRAIN_SPEC.md`, `reproduction/siren_a/docs/G4A_IMPLEMENTATION_PLAN.md`

No unresolved mathematical, source-identity, checkpoint, prediction, or leakage inconsistency was found. The pre-existing unrelated environment package-check findings are recorded privately and were not changed.

`G4A_ENV_VERIFIED = PASS`

`G4A_SAMPLING = PASS`

`G4A_QUERY_GROUPING = PASS`

`G4A_LR = PASS`

`G4A_HGB = PASS`

`G4A_LAMBDARANK = PASS`

`G4A_MODEL_REUSE = PASS`

`G4A_30_MODELS = VERIFIED`

`G4A_90_PREDICTIONS = VERIFIED`

`G4A_TRAIN_TEST_ISOLATION = PASS`

`G4A_RESUME = PASS`

`G2_G3_ARTIFACTS = UNCHANGED`

`G4B_FULL_15_SEEDS = NOT STARTED`

`G5_RANK_ENSEMBLE = NOT STARTED`
