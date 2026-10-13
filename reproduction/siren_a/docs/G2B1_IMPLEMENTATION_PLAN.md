# G2-B1 Full Night-level Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and independently verify deterministic 807/827 night-level
caches for all 6,600 frozen v2.1 records.

**Architecture:** A G2-B1-specific contract validates arrays and provenance;
an independent single-worker runner owns immutable inventory/index files,
no-clobber build/resume, verify-only traversal, CAISR closure, and aggregate QC.
Existing feature and IO modules remain unchanged.

**Tech Stack:** Python 3.9, NumPy 1.26, standard library unittest/json/csv/hashlib,
existing G2-A feature and IO modules.

**Spec:** `reproduction/siren_a/docs/G2B1_FULL_SPEC.md`

## Global Constraints

- Create or modify only the eight files approved by the G2-B1 prompt.
- Do not modify frozen feature definitions or existing tracked files.
- Do not access labels, train models, impute, normalize, rank by site, or drop
  patients.
- Private identities, caches, checkpoints, and detailed run data remain outside
  Git at the approved G2-B1 private root.
- Full writes require explicit `--build`; source and historical caches are
  read-only.

## Review Focus

- A cache with G2-B0 provenance but a forged G2-B1 schema must be rejected.
- A valid-looking cache whose source or execution-code SHA changes must stop
  resume.
- A crash after temporary write but before publication must leave the record
  computable, not skipped.
- Annotation-missing records must retain independent modalities while all
  stage-conditioned families remain invalid.
- Verify-only must detect synchronized unique/View support tampering and a
  complete-but-duplicated/missing private index.

---

### Task 1: Independent G2-B1 cache contract

**Files:**
- Create: `reproduction/siren_a/night_cache/g2b1_cache_contract.py`
- Create: `reproduction/siren_a/night_cache/test_g2b1_cache_contract.py`

**Interfaces:**
- Consumes: `FrozenManifest`, cache arrays, and expected provenance.
- Produces: `CACHE_SCHEMA_VERSION`, `REQUIRED_PROVENANCE`,
  `validate_cache_arrays`, `assert_valid_cache_arrays`,
  `validate_existing_cache`, and `decide_resume_action`.

- [x] Write tests for schema/provenance independence, malformed arrays,
  family support semantics, mapping tampering, source/code hash changes, and
  COMPUTE/SKIP_VERIFIED/STOP_INVALID.
- [x] Run tests and confirm RED because the module does not exist.
- [x] Implement the minimal independent validator and safe exception boundary.
- [x] Run the contract tests and existing safety tests; expect all pass.

### Task 2: Inventory, run identity, and bounded execution modes

**Files:**
- Create: `reproduction/siren_a/night_cache/g2b1_full_run.py`
- Create: `reproduction/siren_a/night_cache/test_g2b1_full_run.py`

**Interfaces:**
- Consumes: Task 1 validator, frozen IO/features, NPZ/audit/reference roots.
- Produces: `build_inventory`, `build_run_identity`, `preflight`,
  `build_or_resume`, `verify_full_cache`, `aggregate_coverage`, and CLI modes.

- [x] Write tests for 6,600-count closure with injectable small fixtures,
  symlink/duplicate rejection, output separation, explicit modes, capacity,
  no forbidden fields, audit 70/6/76 logic, and anonymous stable names.
- [x] Run tests and confirm RED because the runner does not exist.
- [x] Implement preflight and immutable no-clobber run identity/inventory/index.
- [x] Add RED tests for build, interrupted output, resume skip, invalid stop,
  source/code drift, and verify-only completeness.
- [x] Implement sequential compute/publish/checkpoint and independent verify/QC.
- [x] Run both G2-B1 test files and existing G2-A/G2-B0 safety tests; expect all
  pass.

### Task 3: Reference regressions and full preflight

**Files:**
- Test: the two G2-B1 test modules
- Private output only: preflight and SHA snapshots

**Interfaces:**
- Consumes: Task 2 reference replay and preflight functions.
- Produces: evidence for Golden/Pilot exactness, capacity, inventory, CAISR
  closure, and fixed run identity.

- [x] Snapshot historical Golden/Pilot hashes.
- [x] Replay 18 Golden and two accepted Pilot sources against current frozen
  computation; require exact values, NaNs, masks, support, shapes, and dtypes.
- [x] Run preflight-only; require 319/1142/5139, 6,600 unique files, zero
  symlinks, correct hashes, 70/6/76 audit closure, and safe disk margin based
  on actual Pilot size.
- [x] Freeze the final runner/contract code hashes before build.

### Task 4: Full build and safe resume

**Files:**
- Private output only: 6,600 caches, index, immutable checkpoints, run logs.

**Interfaces:**
- Consumes: frozen run identity and Task 2 build API.
- Produces: exactly 6,600 `g2b1_full_v1` record caches.

- [x] Run explicit `--build` single-worker mode and monitor every 100 records.
- [x] On interruption, preserve outputs and use only `--resume`; never remove or
  overwrite.
- [x] Require every published cache to pass immediate contract validation.
- [x] Record elapsed time, peak RSS, bytes, computed count, and resume skips.

### Task 5: Independent full verification and aggregate QC

**Files:**
- Create: `reproduction/siren_a/audit/g2b1_feature_coverage_aggregate.csv`
- Create: `reproduction/siren_a/docs/G2B1_FULL_REPORT.md`

**Interfaces:**
- Consumes: Task 4 caches, private index/audit, frozen Manifest.
- Produces: verified totals and public feature-only aggregate CSV/report.

- [x] Run verify-only over all 6,600 caches and source SHA256 values.
- [x] Require 5,326,200 unique and 5,458,200 View positions and all array/
  provenance/support invariants.
- [x] Verify CAISR 70/6/76 semantics without opening EDFs.
- [x] Aggregate family/site and per-feature/site coverage, support percentiles,
  lawful zeros, record valid-fraction distributions, and entropy coverage.
- [x] Write privacy-safe CSV and report; scan both for identity/path leakage.

### Task 6: Final regression, review, commit, and stop

**Files:** all eight approved G2-B1 deliverables only.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: reviewed Git commit and matching remote branch SHA.

- [x] Run G1-A, G1-B, G2-A, G2-B0, G2-B0-R, and G2-B1 tests plus syntax and
  diff checks.
- [x] Re-hash Golden/Pilot, frozen files, and source inventory; require
  unchanged evidence.
- [x] Self-review the full diff because no subagent tool is available; resolve
  Important findings with RED→GREEN tests and record any deferred minors.
- [x] Explicitly stage only eight approved files, commit with the approved
  message, push normally, and verify clean worktree and equal local/remote SHA.
- [x] Stop with G3, normalization, ranking, and training not started.
