# G4-B Full 15-Seed Training Report

`G4B_STATUS = SUBMITTED_FOR_REVIEW`

## Scope and frozen sources

G4-B extended the reviewed Seed 42 baseline with Seeds 43–56. It trained only the 420 missing models and produced only the 1,260 missing raw prediction units. Seed 42 remained read-only. No holdout label or age was loaded for training, selection, prediction, or evaluation; no LOSO metric, calibration, rank averaging, or ensemble was computed.

The run used branch `research/siren-a-stagepool-v1` at parent `bcbe09354b898fb19556235d67d90bfc0aa085f1`. The frozen source identities were:

- G2-B1 cache fingerprint: `cf88b4e87f8b524258c685db696c278198dae1e6a10bd39fb63e893b2df031df`
- G3-A aggregate fingerprint: `4a0f3956432030e4fa2fb8b51d15aa8fa5269f990d7795cca40d580970920a64`
- G3-B run identity: `0936a4fcd91113f4ee76caa69c5cdfd3b501db87aa88b718d7a8b02ffccaf46d`
- G3-B completion: `cd218356f047e39aee7950c2dab8c549616b8abaf075093bd8296e01b8320b57`
- G4-A run identity: `4d9654cc36142b23aaa058a82a0b154f8b5d9557feb1c506354374138f13945c`
- Frozen Manifest: `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a`
- Frozen Spec: `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a`
- Demographics source: `b9009598589846e84462636bfcaa290ed1d8666c89b1bc1ad1a9eda26b43ff7f`
- G4-B run identity: `544d4c192ed0a3b5fdbd5452d36edf436af6ed52ece5d88077f327e13470b412`

The frozen execution environment was Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, joblib 1.5.3, and LightGBM 4.6.0. User site packages were disabled. OpenMP used four threads; OpenBLAS, MKL, and NumExpr used one thread each.

## Seed 42 bridge

The existing G4-A source closed at 30 models and 90 predictions. Both saved Query units were independently reconstructed array by array from their frozen samples, labels, and ages. All ten LR adapter states were independently reconstructed from the actual selected rows; native-NaN HGB and LambdaRank adapters were also checked. Existing model and prediction completion evidence closed without calling `fit`, rewriting a checkpoint, or modifying any Seed 42 artifact.

`G4B_SEED42_BRIDGE = PASS`

## Sampling

All 14 new seeds used the unchanged `age_label_70pct_pcg64_v1` sampler and its `siren-g4a-v1` seed derivation prefix. Each of the 28 site/seed units had 18 nonempty age-by-label strata, exact quota closure, no replacement, no duplicate selected index, and at least one retained row per nonempty stratum.

| Train site | Full N | Selected N per seed | Positive | Negative | Seeds verified |
|---|---:|---:|---:|---:|---:|
| S0001 | 5,139 | 3,597 | 235 | 3,362 | 14/14 |
| I0006 | 1,142 | 799 | 78 | 721 | 14/14 |

Sampling indices were shared across all five views and all three model families for the same site/seed. S0001 models were also shared across the I0002 and I0006 holdouts. Per-seed index SHA256 values remain in the private audit only.

`G4B_SAMPLING = PASS`

## LambdaRank queries

All 28 new Query units and both Seed 42 Query units were independently reconstructed without calling the production query builder. Permutations, inverse permutations, sorted local/global indices, group IDs, labels, group sizes, class counts, fingerprints, truncation levels, and AC-pair diagnostics matched exactly. Every five-year age group contained both classes.

| Train site | Age groups | Maximum group / truncation | AC pair coverage across Seeds 43–56 |
|---|---:|---:|---:|
| S0001 | 9 | 875 | 0.760379–0.782812 |
| I0006 | 9 | 170 | 0.749857–0.789877 |

S0001 had 89,259–90,559 valid age-matched positive/negative pairs per seed; I0006 had 6,844–7,049. These values were diagnostic only and did not alter sampling, grouping, or parameters.

`G4B_QUERY_GROUPING = PASS`

`G4B_QUERY_SEMANTIC_VERIFICATION = PASS`

## Models and adapters

Exactly 420 new models were trained: 140 LR, 140 HGB, and 140 LambdaRank models. Each model used the frozen five-view dimensions (113, 129, 349, 113, 123), the correct 70% rows, and `random_state` equal to its Seed. Only LambdaRank's query-derived truncation varied in addition to the Seed.

LR medians, finite counts, all-missing flags, fallback values, selected-index fingerprints, and feature-order fingerprints were independently recomputed for all 140 new LR adapters and ten Seed 42 LR adapters. HGB and LambdaRank passthrough adapters preserved all finite values and NaN positions. No feature was added, deleted, rescaled, or reordered.

| Model | Count | Total fit seconds | Maximum fit seconds | Iterations/trees | Reload max difference |
|---|---:|---:|---:|---:|---:|
| LR | 140 | 28.868 | 1.071 | 41–141 | 0 |
| HGB | 140 | 2,009.447 | 44.672 | 250 | 0 |
| LambdaRank | 140 | 1,658.452 | 29.875 | 250 | 0 |
| Total | 420 | 3,696.767 | 44.672 | — | 0 |

All training outputs were finite and nonconstant. Every checkpoint was reloaded and replayed. The run recorded 140 LightGBM feature-name inference warnings, the same benign warning class observed in G4-A; they did not change shapes, scores, or replay equality.

`G4B_NEW_MODELS = 420/420`

`G4B_LR_ADAPTER_INDEPENDENT_VERIFICATION = PASS`

`G4B_HGB = PASS`

`G4B_LAMBDARANK = PASS`

## Raw predictions

Exactly 1,260 new prediction units were generated. Each holdout/protocol pair contains 210 new units (14 seeds × 5 views × 3 models), with test lengths 319 for I0002, 1,142 for I0006, and 5,139 for S0001. LR/HGB outputs are class-1 probabilities; LambdaRank outputs remain raw ranking scores.

All predictions were finite and nonconstant. Independent checkpoint reload, adapter reconstruction, frozen test-matrix loading, and prediction replay produced a maximum absolute difference of zero. Test-index fingerprints and feature-order fingerprints matched in every unit.

`G4B_NEW_PREDICTIONS = 1260/1260`

## G5-ready closure

The reference-only manifest combines Seed 42 from G4-A with Seeds 43–56 from G4-B. It contains exactly 15 seeds, 450 unique model identities, and 1,350 unique prediction identities. Every holdout/protocol contains 225 score vectors with one common test-index identity. Every prediction traces to one model, and every model traces to one sample and training source.

No prediction was ranked, averaged, calibrated, thresholded, or scored while building this manifest.

`FULL_SEEDS = 15/15`

`FULL_MODELS = 450/450`

`FULL_PREDICTIONS = 1350/1350`

`G5_MANIFEST = VERIFIED`

## Recovery and integrity

The independent no-fit verify-only pass checked 28 samples, 28 queries, 420 models, 1,260 predictions, 14 Seed completion units, the Seed 42 bridge, and the G5 manifest. The subsequent Resume pass built zero units and skipped 28 verified samples, 28 verified queries, 420 verified models, and 1,260 verified predictions.

Before and after Resume, the complete Seed tree aggregate remained `101cba3232fc3e656df2662d1b3eb65b48fcf85d7f5c6e0b4d87d12fc299b69d`. The root completion SHA256 remained `9da0f2864282f688a8473af1f0082fcf7c00b8a173deb2746e11ac0027629d4c`; the G5 completion SHA256 remained `9c7998c4d2ffd47d5acae670a0a4339d29e53f39e678ca7315d2bb599535f959`. No `.building` directory remained.

`G4B_RESUME = PASS`

The final short preflight additionally verified the current Git branch and remote SHA, the exact frozen G4-A executable-code hashes, all source identities, and the historical G4-B execution identity. The validator preserves the truthful original code fingerprint for the completed production run; changed code is not presented as if it generated the existing artifacts. Corruption, symlink, interrupted-build, no-clobber, source-drift, and holdout-target/scoring-surface guards were exercised in isolated tests.

## Resources

- Seed 43 V1 smoke: 6 min 31 s; peak RSS 304,428 KiB.
- Full build (including reuse of smoke units): 2 h 30 min 57 s; peak RSS 353,100 KiB.
- Independent no-fit verify-only: 46 min 52 s; peak RSS 315,012 KiB.
- Resume verification: 1 h 22 min 48 s; peak RSS 327,756 KiB.
- Observed wall-clock total for these four production commands: approximately 4 h 47 min 8 s.
- Peak observed RSS: 353,100 KiB; observed swap: 0.
- Private G4-B root: 5,654 files, 178,104,620 bytes (169.85 MiB).

## Tests and privacy

The production-close regression previously ran 183 G1-B through G4-B unittests with 183 passes, plus 15/15 historical G1-A function-style checks. After the final review hardening, the focused G4-B suite ran 26/26 tests, including three new source-drift, corruption/symlink, interruption/no-clobber, and leakage-surface guards. The long production pipelines were not rerun merely to duplicate established evidence. Python syntax compilation, public-output privacy scanning, and the approved-file scope check are part of final Git closure.

The public model CSV contains 420 aggregate rows and the public prediction CSV contains 1,260 aggregate rows. Neither contains patient identifiers, record IDs, ages, row-level labels, selected indices, feature vectors, or individual predictions. Production model files and private audits remain outside Git.

## Gate decision

All frozen-source, Seed 42 bridge, sampling, query, adapter, model, prediction, replay, recovery, and manifest checks passed. No unexplained training or mathematical inconsistency remains.

`G4B_ENVIRONMENT = PASS`

`G4B_UPSTREAM_IDENTITY = PASS`

`G4B_SEED42_BRIDGE = PASS`

`G4B_MODEL_RANDOM_STATES = PASS`

`G4B_MODEL_REUSE = PASS`

`G4B_420_NEW_MODELS = VERIFIED`

`G4B_1260_NEW_PREDICTIONS = VERIFIED`

`G4B_FULL_450_MODELS = VERIFIED`

`G4B_FULL_1350_PREDICTIONS = VERIFIED`

`G4B_TRAIN_TEST_ISOLATION = PASS`

`G4B_G5_READY_MANIFEST = VERIFIED`

`G1_G2_G3_G4A_ARTIFACTS = UNCHANGED`

`G5_RANK_ENSEMBLE = NOT STARTED`

`G5_LOSO_SCORING = NOT STARTED`
