# FeatureLab v2 Seed 42 Training Report

This public report contains aggregate verification only. Real feature values, prediction values, labels, ages, row indices, record identities, model files and private paths below the named output root are not included in Git.

## Status

```ini
V2_SEED42_STATUS = COMPLETE
TRAIN_METADATA_ROLE_ISOLATION = PASS
SEED42_SAMPLER_OLD_MATCH = 2/2
QUERY_OLD_MATCH = 2/2
TRAIN_PROTOCOL_EQUALITY = PASS (15/15 real holdout-view training contracts)
OLD_VIEW_PREFIX_INPUT_MATCH = PASS (10/10 LR adapter prefix contracts)
LR_SEED_SUBSET_ADAPTER_CHECK = PASS (10/10 models, fit_scope=seed_subset)
NATIVE_NAN_PASSTHROUGH_CHECK = PASS (synthetic adapter tests)
NEW_MODELS = 30/30
NEW_PREDICTION_UNITS = 90/90
CHECKPOINT_REPLAY_MAX_ABS_DIFF = 0.0
OLD_SEED42_PREDICTION_BRIDGE = 90/90
E0_E1_E2_E3_REFERENCE_MANIFEST = PASS (360 identity-only entries)
RESUME_WITHOUT_REFIT = PASS (30/30 model skips, 90/90 prediction skips)
WARNINGS_OR_NEEDS_REVIEW = LightGBM sklearn feature-name warnings were observed during raw prediction replay; numerical replay remained exact and no model metadata warning was recorded.
SEEDS43_56 = NOT_STARTED
FUSION_AND_PERFORMANCE_SCORING = NOT_STARTED
BATCH2 = AUDIT_ONLY
INPUT_ARTIFACT_IMMUTABILITY = PASS (Rank256 source units read-only; completion/file identities revalidated)
BRANCH_A_STATUS = UNCHANGED
EXECUTION_CODE_SHA = 5339b8d6895c0c362273950f1cd5a8cbb44cef8a
FINAL_REPORT_COMMIT = PENDING
LOCAL_REMOTE_SHA_STATUS = PENDING FINAL PUSH
```

## Inputs and role controls

The audited metadata join contained 6600 records. The v2 loader exposed targets only for the requested training role and exposed no test/unused target attributes. The three frozen LOSO partitions and all 15 real transductive/inductive training-side equality checks passed. No evaluation label or age was read during prediction replay.

The two real Seed 42 sample arrays and two query arrays matched the immutable G4-A artifacts element-for-element, including ordering and fingerprints. The old-view adapter prefix check covered two training sites and five views. All ten LR adapters used the selected-subset fit scope; HGB/LambdaRank native-NaN behavior passed synthetic checks.

## Models and predictions

The private v2 output contains 30 complete model units and 90 complete prediction units. Model dimensions are distributed as six units each at 161, 177, 493, 257 and 376 features. Score semantics are 60 positive-probability units (LR/HGB) and 30 raw LambdaRank units. No feature selection, calibration, fusion or performance metric was computed.

All 30 checkpoints reloaded successfully. The 90 prediction files were reloaded with exact test-row order checks; maximum absolute replay difference was 0.0. No incomplete temporary unit directories remained. The private output footprint was 14,590,268 bytes across 393 files at verification time.

The immutable old Seed 42 prediction bridge verified all 90 old units for completion metadata, file hashes, score shape/finite/nonconstant checks and test-row ordering. E0–E3 is an identity-only reference manifest and does not contain prediction values.

## Tests and execution boundary

Focused unittest coverage passed 10/10 tests:

```text
extensions.tests.test_train_inputs_v2   4/4
extensions.tests.test_train_seed42_v2   3/3
extensions.tests.test_seed42_run_v2     3/3
```

The real preflight used the frozen sleepfm environment (Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, joblib 1.5.3 and LightGBM 4.6.0) with the approved single-process thread settings. No EDF, candidate-pool or Rank256 recomputation occurred. Branch A, old Rank256 units, old G4-A models/predictions and the old five-view manifests were not modified.

Seeds 43–56, model scoring, fusion/performance analysis and Batch2 implementation remain unauthorized/not started. The next approval boundary is review of this Seed 42 execution and the identity-only E0–E3 manifest.
