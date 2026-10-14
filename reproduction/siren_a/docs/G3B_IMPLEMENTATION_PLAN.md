# G3-B Model Input Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build a label-safe, provenance-complete model input pipeline for frozen G3-A five-view Rank256 artifacts without fitting any model.

**Architecture:** label_join.py owns exact identity and train-only target access; input_data.py loads frozen fold/view artifacts into FoldInputs; adapters.py implements LR subset-median and native-NaN passthrough contracts; run_inputs.py coordinates preflight, smoke, build, atomic private evidence, resume, and public aggregates.

**Tech Stack:** Python 3.9, NumPy 1.26, standard-library csv/json/hashlib/pathlib/unittest.

**Spec:** reproduction/siren_a/docs/G3B_INPUT_SPEC.md

## Global Constraints

- Add exactly eleven approved files and modify no existing tracked file.
- Read G2-B1, G3-A, G3-A-V, Frozen Manifest/Spec, and demographics.csv only.
- Never expose test/unused labels or ages through FoldInputs.
- Never call a model fit method, rescale Rank256, alter views, select features, or generate a new feature cache.
- Use allow_pickle=False, float64 matrices, bool masks, exact row ordering, no-clobber publication, and completion-last verification.
- Public outputs contain aggregates only.

## Review Focus

- Metadata reorder or same-count wrong identity must not silently misalign training targets.
- FoldInputs must not retain a reference to full metadata, test labels, test ages, or patient IDs.
- LR state must depend only on selected fit rows and preserve all columns including all-missing columns.
- G3-A-V PASS text alone is insufficient; file hashes, code hashes, source fingerprint, and G3-A aggregate must close.
- Resume must reject semantic-invalid adapter evidence and incomplete runs.

### Task 1: Metadata identity and train-only targets

**Files:** Create label_join.py and test_label_join.py.

**Interfaces:** Produces MetadataIndex, load_metadata_index, check_identity_join, load_train_targets, make_age_groups, and audit_age_groups.

- [x] Write synthetic tests for composite-key joins, shuffled CSV invariance, mismatch and duplicate cases, strict labels, finite ages, patient overlap, and forbidden global target exposure.
- [x] Run tests and confirm RED because label_join.py is absent.
- [x] Implement the minimum parser, join, train-only target loader, and age grouping.
- [x] Run focused tests GREEN.

### Task 2: FoldInputs and frozen views

**Files:** Create input_data.py and test_input_data.py.

**Interfaces:** Consumes MetadataIndex and frozen G3-A units; produces FoldInputs, load_fold, get_view_feature_ids, and check_fold_inputs.

- [x] Write tests for exact fold rows, dimensions, feature order, no test targets or age, protocol-train equality, forbidden columns, and source drift.
- [x] Run RED.
- [x] Implement package-safe imports and strict read-only G3-A artifact loading without normalization.
- [x] Run GREEN.

### Task 3: Model adapters

**Files:** Create adapters.py and test_adapters.py.

**Interfaces:** Consumes FoldInputs matrices and optional selected fit rows; produces AdapterState, fit_adapter, transform_adapter, and check_adapter.

- [x] Write tests for LR median, 128/255 fallback, subset-only fitting, test isolation, unchanged observations, no residual NaN, HGB/LambdaRank native NaN, dimensions, and fingerprints.
- [x] Run RED.
- [x] Implement deterministic states and transforms without model libraries.
- [x] Run GREEN.

### Task 4: Orchestration, provenance, and recovery

**Files:** Create run_inputs.py; extend approved tests only as needed.

**Interfaces:** Consumes Tasks 1-3; produces prepare_inputs, verify_inputs, CLI modes, atomic private evidence, resume decisions, and aggregate rows.

- [x] Write tests for restricted modes, no model calls, no-clobber, partial-run rejection, verified resume, source drift, and public privacy.
- [x] Run RED.
- [x] Implement preflight, smoke/build orchestration, compact private states/checks, completion-last publication, and verify-only.
- [x] Run GREEN.

### Task 5: Real identity and I0002 smoke

- [x] Run preflight and require 6,600 exact matches, legal labels/ages, zero patient overlap, and unchanged frozen fingerprints.
- [x] Run I0002 two-protocol, five-view, three-adapter smoke and require all 30 combinations.
- [x] Independently re-read smoke evidence and verify sources, masks, adapter states, labels, age groups, and isolation.

### Task 6: Full 90-combination build

- [x] Run all three folds, two protocols, five views, and three adapters.
- [x] Verify cross-protocol train equality, dimensions, labels, age groups, LR transforms, and passthrough NaNs.
- [x] Exercise verified resume and source-drift rejection without overwriting evidence.

### Task 7: Public audit and report

**Files:** Create g3b_input_summary.csv and G3B_INPUT_REPORT.md.

- [x] Generate exactly 90 aggregate rows and a privacy-safe report from verified private evidence.
- [x] Include identity closure, per-training-site labels/age groups, adapter behavior, combination dimensions, resources, and explicit no-training state.
- [x] Scan for patient identifiers, paths, row-level values, and private mappings.

### Task 8: Regression and submission

- [ ] Run all new tests, required historical regressions, syntax checks, privacy scan, and immutable-source fingerprints.
- [ ] Perform whole-branch self-review if no subagent tool is available; fix Critical/Important findings with RED-GREEN evidence.
- [ ] Explicitly stage only eleven approved files, commit with the required message, push normally, verify Local/Remote SHA, and stop before G4.
