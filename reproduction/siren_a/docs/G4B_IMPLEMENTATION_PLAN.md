# G4-B Full Multi-Seed Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the frozen Seed 42 baseline with Seeds 43–56, independently verify all semantics, and close a reference-only 450-model/1,350-prediction manifest.

**Architecture:** New `g4b` modules wrap but never modify G4-A sampling, query production, prediction, and checkpoint primitives. G4-B owns dynamic-seed fitting, completion-last orchestration, independent query/adapter verification, and cross-gate manifests. Production data remain in a separate private root; Git receives only code, tests, specs, and aggregate reports.

**Tech Stack:** Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, joblib 1.5.3, LightGBM 4.6.0, unittest.

**Spec:** `reproduction/siren_a/docs/G4B_TRAIN_SPEC.md`

## Global Constraints

- Use only Seeds 43–56 for new fits and preserve Seed 42 read-only.
- Use only frozen G3-B `FoldInputs`; never load holdout labels or ages.
- Reuse G4-A sampler and production query functions without modifying tracked G1–G4-A files.
- Fit exactly 420 new models and produce exactly 1,260 new predictions.
- Use the absolute `sleepfm_env` Python and frozen thread environment.
- Do not evaluate, calibrate, ensemble, or start G5.
- Only the 16 user-approved new files may become tracked.

## Review Focus

- Seed injection must alter only `random_state` plus query-derived LambdaRank truncation.
- Seed 42 bridge validation must be semantic and read-only, not hash-only.
- Independent query and adapter expectations must not call their production builders.
- Resume must detect synchronized metadata/content tampering and never retrain verified units.
- G5 manifests must reference exact artifacts with complete identity closure and no score computation.

---

### Task 1: Seed schedule and dynamic model factory

**Files:** Create `g4b/config.py`, `g4b/models.py`, `g4b/test_seed_schedule.py`, and `g4b/test_models.py`.

**Interfaces:** `new_model_identities()`, `new_prediction_identities()`, `parameters_for_seed()`, and `fit_seed_model()` provide the frozen schedule and seed-aware estimators.

- [ ] Write tests for 14 seeds, 420/1,260 identities, parameter immutability, per-seed random states, selected-subset LR, native NaNs, and checkpoint replay.
- [ ] Run tests and confirm failure because G4-B modules do not exist.
- [ ] Implement minimal configuration and model wrappers.
- [ ] Run both test modules and confirm PASS.

### Task 2: Independent semantic verifier

**Files:** Create `g4b/verify_g4b.py` and `g4b/test_verify_g4b.py`.

**Interfaces:** `rebuild_query_expected()`, `verify_query_artifact()`, `verify_adapter_state()`, `verify_model_semantics()`, and `verify_prediction_artifact()` reject semantic drift.

- [ ] Write mutation tests for every query array, LR median/count/index/order fields, native-NaN adapters, parameters, checkpoints, and predictions.
- [ ] Run tests and confirm expected missing-interface failures.
- [ ] Implement independent NumPy reconstruction and comparisons without production query/adapter fit calls.
- [ ] Run verifier tests and confirm PASS.

### Task 3: Runner, recovery, and Seed 42 bridge

**Files:** Create `g4b/run_g4b.py` and `g4b/test_run_g4b.py`.

**Interfaces:** `preflight()`, `build_seed42_bridge()`, `build_seed()`, `build_all()`, and CLI modes implement completion-last units and verified resume.

- [ ] Write tests for preflight drift, exact train sharing, bridge read-only behavior, no-clobber, incomplete artifacts, smoke scope, build scope, and no fit in verify-only.
- [ ] Run tests and confirm RED.
- [ ] Implement the runner by reusing G4-A sampler/query/checkpoint/prediction primitives.
- [ ] Run runner tests and confirm PASS.

### Task 4: G5-ready manifest closure

**Files:** Create `g4b/manifest.py` and `g4b/test_manifest.py`.

**Interfaces:** `build_model_manifest()`, `build_prediction_manifest()`, and `verify_g5_manifest()` close Seed 42 references plus G4-B artifacts.

- [ ] Write tests for 450/1,350 uniqueness, per-seed/per-holdout counts, missing/duplicate references, test-index agreement, and traceability.
- [ ] Run tests and confirm RED.
- [ ] Implement reference-only manifests and closure checks.
- [ ] Run manifest tests and confirm PASS.

### Task 5: Real preflight, Seed 42 bridge, and Seed 43 smoke

**Files:** Private G4-B root only.

**Interfaces:** Frozen CLI preflight and smoke.

- [ ] Run all G4-B tests and syntax checks.
- [ ] Run `--preflight-only` and verify frozen identities and capacity.
- [ ] Build the read-only Seed 42 semantic bridge.
- [ ] Run `--smoke`: two Seed 43 samples/queries, six V1 models, and 18 predictions.
- [ ] Independently verify smoke artifacts before continuing.

### Task 6: Full Seeds 43–56 build and verification

**Files:** Private G4-B root only.

**Interfaces:** `--build`, `--verify-only`, and `--resume`.

- [ ] Complete Seed 43 then Seeds 44–56 serially with per-seed completion.
- [ ] Independently verify 420 models, 1,260 predictions, all query arrays, and all adapters without fitting.
- [ ] Build and verify the 450/1,350 G5-ready manifests.
- [ ] Run resume and require built=0, models skipped=420, predictions skipped=1,260.

### Task 7: Reports, regression, and Git closure

**Files:** Create `docs/G4B_TRAIN_REPORT.md`, `audit/g4b_model_summary.csv`, and `audit/g4b_prediction_summary.csv`.

**Interfaces:** Public aggregate outputs and final gate evidence.

- [ ] Generate exactly 420 model rows and 1,260 prediction rows without patient-level fields.
- [ ] Run G4-B tests, G4-A/G3-B and historical lightweight regressions, syntax, privacy, and scope checks.
- [ ] Perform a whole-branch self-review against the spec and ledger; fix Important findings by RED→GREEN only.
- [ ] Explicitly stage only 16 approved files, commit, push normally, verify local/remote SHA and clean worktree, then stop.
