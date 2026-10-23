# V5 Family Leave-One-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and audit the four frozen V5 family leave-one-out configurations using frozen Rank256 inputs, resumable private model/prediction units, and aggregate-only public reports.

**Architecture:** A dedicated `v5_family_loo` module validates the frozen view manifest and maps each LOO configuration to manifest-derived unique-column indices. A private runner reuses the existing label-safe fold loader, sampling/query contracts, model checkpoint format, and official scoring helpers while keeping models, predictions, references, and diagnostics outside Git. Public JSON/CSV/Markdown outputs contain only configuration identities and aggregate metrics.

**Tech Stack:** Python 3, NumPy, unittest, existing FeatureLab/Branch-A model and scoring helpers, atomic JSON/NPZ publication.

**Spec:** User-approved V5 family LOO prompt in the conversation; frozen source contracts in `audit/featurelab_v2_view_manifest.csv`, Rank256 v2, and existing E0/E2 comparison artifacts.

## Global Constraints

- Four configs only: `LOO_NO_COHERENCE` (286), `LOO_NO_CAISR` (341), `LOO_NO_EMG` (304), `LOO_NO_RESP` (320).
- V5 full view is 376 columns: old807 prefix 123 plus coherence90, CAISR35, EMG72, Resp56.
- Reuse frozen Rank256 arrays; do not rebuild Rank256, candidate pools, EDF/raw features, or V1–V4 models.
- Train 4 configs × 2 train sites × 15 seeds (42–56) × 3 models = 360 model units; create 1080 prediction units.
- Do not load evaluation targets until all model and prediction units are complete.
- No-clobber, exact payload validation, checkpoint reload replay tolerance ≤1e-12, and resumable private units are mandatory.
- Public outputs contain aggregate metrics only; no record IDs, labels, raw predictions, bootstrap multiplicities, or private paths.

## Review Focus

- Manifest-derived column selection must preserve repeated old features by `unique_column_index`; test exact V5 widths and source-block counts.
- LOO subset matrices/masks must be bitwise/equal-NaN slices of frozen Rank256 arrays; test rank projection without recomputation.
- Source identity must be independent of requested configuration order; test deterministic config and feature-ID hashes.
- A partially published model/prediction unit must never be treated as complete; test no-clobber and resume behavior in temporary directories.
- Bootstrap and public serialization must be deterministic and privacy-safe; test fixed seed, finite aggregate rows, and forbidden-field rejection.

### Task 1: Frozen manifest and LOO selection

**Files:**
- Create: `per_epoch_features/extensions/v5_family_loo.py`
- Create: `per_epoch_features/extensions/tests/test_v5_family_loo.py`
- Create: `audit/featurelab_v2_v5_family_loo_manifest.json`

- [ ] Write failing tests for manifest validation, four widths, source-block removal, deterministic hashes, and repeated-old-column preservation.
- [ ] Run the focused unittest module and observe the expected missing-module failures.
- [ ] Implement immutable config definitions, manifest reader/validator, `select_v5_columns`, and public manifest serialization.
- [ ] Re-run focused tests; review that no fallback or hard-coded contiguous slicing is used.
- [ ] Commit the contract and selection implementation.

### Task 2: Rank projection and sample/query bridge

**Files:**
- Modify: `per_epoch_features/extensions/v5_family_loo.py`
- Modify: `per_epoch_features/extensions/tests/test_v5_family_loo.py`

- [ ] Add RED tests for exact-NaN rank projection and 30 seed/site sample-query identity checks against frozen artifacts.
- [ ] Implement `project_v5_fold` by slicing the existing V5 fold arrays and masks with manifest indices, preserving feature IDs and row identities.
- [ ] Implement the label-safe sample/query bridge using train-only targets and the existing query contract.
- [ ] Verify synthetic tests and the bounded 30-match bridge; commit.

### Task 3: Private model/prediction units and resumable execution

**Files:**
- Create: `per_epoch_features/extensions/v5_family_loo_runner.py`
- Modify: `per_epoch_features/extensions/tests/test_v5_family_loo.py`

- [ ] Write RED tests for deterministic run identity, model/prediction no-clobber, save/reload replay, and resume without refit/re-predict.
- [ ] Implement atomic private publication and exact payload/metadata validation, then the 360-model/1080-prediction execution stages without target loading in prediction stage.
- [ ] Add a run checkpoint that records completed counts and identity hashes without patient data.
- [ ] Run synthetic resume tests and commit.

### Task 4: Private references, fused groups, scoring, stability, and bootstrap

**Files:**
- Modify: `per_epoch_features/extensions/v5_family_loo_runner.py`
- Modify: `per_epoch_features/extensions/tests/test_v5_family_loo.py`

- [ ] Add RED tests for 5400 reference-key construction, 24 fused groups, delta labels, deterministic 2000-replicate bootstrap, and privacy checks.
- [ ] Reuse frozen old V1–V4 sources and newly created LOO V5 predictions; build private references and rank-fused groups only after all predictions complete.
- [ ] Load evaluation targets only in this scoring stage; emit 24 site rows, 8 protocol summaries, 96 seed-stability rows, and 96 bootstrap rows with official scoring functions.
- [ ] Verify the fixed seed and aggregate-only serialization; commit.

### Task 5: Public specifications and reports

**Files:**
- Create: `docs/featurelab/FEATURELAB_V2_V5_FAMILY_LOO_SPEC.md`
- Create: `docs/featurelab/FEATURELAB_V2_V5_FAMILY_LOO_REPORT.md`
- Create: `audit/featurelab_v2_v5_family_loo_site_scores.csv`
- Create: `audit/featurelab_v2_v5_family_loo_protocol_summary.csv`
- Create: `audit/featurelab_v2_v5_family_loo_seed_stability.csv`
- Create: `audit/featurelab_v2_v5_family_loo_bootstrap.csv`

- [ ] Write RED privacy/row-count tests for public artifacts.
- [ ] Generate aggregate-only reports and fixed interpretation fields; preserve existing E0/E2/stability/bootstrap outputs byte-for-byte.
- [ ] Run the full extensions unittest suite plus the frozen contract checks, review Git diff for private-data leakage, and commit/push.

## Verification

Before completion, verify the exact model/prediction/reference counts, feature-width and ID uniqueness contracts, checkpoint replay tolerance, deterministic bootstrap seed, unchanged Branch A and existing public artifacts, and clean local/remote SHA. Do not claim real extraction or training success unless fresh command output demonstrates each required count.
