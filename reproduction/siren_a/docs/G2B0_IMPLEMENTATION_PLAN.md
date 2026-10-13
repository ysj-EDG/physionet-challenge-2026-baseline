# G2-B0 Full-Run Readiness and Targeted Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that frozen one-record pooling can be extended safely to a later 6,600-record run without performing that run.

**Architecture:** A prepared-record layer removes repeated whole-array conversions while preserving the public calculator. A hard-link publisher and three-state validator provide no-clobber and Resume safety. A bounded Pilot orchestrator consumes at most 80 distinct NPZ contents, writes at most 12 new caches, and emits private evidence plus a public aggregate report.

**Tech Stack:** Python 3.9.18, NumPy 1.26.4, standard library `unittest`, `pathlib`, `hashlib`, `json`, `os`, `tempfile`, and `tracemalloc`/process resource observations.

**Spec:** `reproduction/siren_a/docs/G2B0_SPEC.md`

## Global Constraints

- Work only on `research/siren-a-stagepool-v1` from parent `ed5707531bfc0c3cc404b33baafae9627106c4b0`.
- Do not modify Frozen Manifest/Spec, G2-A files other than the two approved implementation modules, Golden caches, source NPZs, or the old workspace.
- Keep 807 unique features, 827 view positions, float64 outputs, threshold five, and all frozen formulas and missingness rules.
- Read no more than 80 distinct NPZ contents and publish no more than 12 new Pilot records.
- Do not expose a full-run mode, install dependencies, read labels, drop records, impute, scale, normalize, or train.
- Commit only the ten approved files with the exact requested message after all verification passes.

## Review Focus

- Concurrent publishers targeting one cache: exactly one succeeds and the winner is never overwritten.
- A structurally valid cache with one mismatched code/source fingerprint: Resume must return `STOP_INVALID`.
- A prepared record containing non-contiguous or float32 arrays: one float64 preparation per full input array and exact frozen output.
- A boundary record whose invalid source affects only one feature family: unrelated families remain available and the record remains present.
- A Pilot interrupted between temporary validation and publication: no partial final file and a subsequent decision remains safe.

---

### Task 1: Prepared calculation and Golden regression

**Files:** modify `night_cache/g2a_features.py`; create `night_cache/test_g2b0_optimization.py`.

**Interfaces:** preserve `compute_night_features(record, manifest) -> NightFeatureResult` and `spatial_value_and_mask(row, record)`; add internal `PreparedRecord` and preparation helpers only.

- [ ] Write tests that count one full `X_seq` float64 conversion, preserve the public APIs and hand-derived Mean/IQR, EEG, and spatial results, and compare every array from all 18 Golden caches exactly.
- [ ] Run the new test file and observe failure from repeated conversions or absent preparation instrumentation.
- [ ] Implement one-time float64 arrays plus reusable stage/lead masks; route all row calculations through prepared state without changing formulas.
- [ ] Run optimization tests and existing G2-A feature tests; require exact 18/18 Golden agreement.

### Task 2: Atomic no-clobber publication and Resume

**Files:** modify `night_cache/g2a_io.py`; create `night_cache/g2b0_batch_safety.py` and `night_cache/test_g2b0_safety.py`.

**Interfaces:** `validate_existing_cache(path, expected_provenance, manifest) -> ValidationResult`; `decide_resume_action(...) -> Action`, where Action is only `COMPUTE`, `SKIP_VERIFIED`, or `STOP_INVALID`.

- [ ] Write tests for normal publication, existing target, real two-writer competition, interrupted write, corrupt/partial cache, wrong shape, each provenance mismatch, exact unique/view mapping, and verified skip.
- [ ] Run safety tests and observe failures under `exists() + os.replace()` and missing validator APIs.
- [ ] Implement same-directory temporary write, fsync, exact round-trip verification, atomic hard-link publication, directory fsync, cleanup, structured validation, and three-state decisions.
- [ ] Run safety tests and all G2-A tests; require no overwrite and deterministic Resume results.

### Task 3: Bounded Pilot orchestration and synthetic boundaries

**Files:** create `night_cache/g2b0_pilot.py` and `night_cache/test_g2b0_pilot.py`.

**Interfaces:** bounded candidate selector, Pilot cache payload builder, probability classifier, inventory builder, checkpoint writer, and CLI without any full-run option.

- [ ] Write tests for deterministic seed, Golden exclusion, 12/30/80 caps, four required synthetic boundaries and feature-local missingness, probability categories, inventory rejection, privacy, and absence of label/drop/imputation/scaling/full modes.
- [ ] Run Pilot tests and observe failure because orchestration does not exist.
- [ ] Implement only the bounded Pilot workflow and private evidence writers using Task 2 safety interfaces.
- [ ] Run Pilot tests and existing suites; require all synthetic boundaries and caps to pass.

### Task 4: Private evidence, Golden replay, targeted Pilot, inventory, and benchmark

**Files:** private target only; no additional tracked files.

**Interfaces:** execute the reviewed Task 1-3 code with frozen fingerprints captured before execution.

- [ ] Create the previously absent private target safely; snapshot all 18 Golden SHA256 values and the complete 6,600-name inventory.
- [ ] Replay all 18 sources through optimized code and require exact values, NaNs, dtypes, masks, supports, and views without rewriting Golden.
- [ ] Classify the six I0006 Golden probability anomalies before selecting bounded candidates.
- [ ] Deterministically inspect at most 62 non-Golden candidates, publish at most 12 anonymous Pilot caches, validate replay and Resume, and record uncovered categories.
- [ ] Benchmark source from commit `ed570753` against optimized code on at least one record per site with warmup/repeats, recording runtime, conversion counts, and peak memory.
- [ ] Re-hash Golden and sampled source files; require no changes.

### Task 5: Aggregate report, full verification, and delivery

**Files:** create `docs/G2B0_REPORT.md`; verify all ten approved files.

**Interfaces:** consume private aggregate evidence only; publish no identity, path, age, label, record-level feature, or recoverable ordering.

- [ ] Write the report with Git, Golden, optimization, safety, Pilot, entropy, inventory, tests, and gate sections; distinguish alerts, uncovered categories, and blockers.
- [ ] Run G1-A, G1-B, G2-A, and all G2-B0 tests plus syntax checks; inspect complete output.
- [ ] Verify Frozen, Golden, and source fingerprints; verify exact Git scope and report privacy.
- [ ] Explicitly stage only approved files, force-add only the three approved ignored docs, commit with `feat(siren-a): harden G2-B0 cache execution and targeted pilot`, push normally, and verify remote/local SHA equality.
- [ ] Stop with G2-B1, G3, normalization, and training unstarted.
