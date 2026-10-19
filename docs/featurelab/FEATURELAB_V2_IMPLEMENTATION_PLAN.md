# FeatureLab v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify an isolated, restartable family-level cache and the six approved 637-dimensional NPZ-derived FeatureLab families, then deliver a real one-record smoke and a review-only Batch 2 preprocessing audit.

**Architecture:** A strict read-only source adapter loads one frozen NPZ once, six pure family extractors return validated `FeatureResult` objects, and a registry-driven runner writes one no-clobber cache per record/family/recipe. Recipe identity follows only declared numerical dependencies; public artifacts contain aggregate or synthetic information, while real record values and identity remain private.

**Tech Stack:** Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3 and pandas 2.2.3 only where already required by repository code, standard-library `unittest`, NPZ/JSON/SHA256, Git worktrees.

**Spec:** `docs/featurelab/FEATURELAB_V2_SPEC.md`

## Global Constraints

- Work only in `/database/home/gaohaojie/workspace/python-example-2026-featurelab` on `research/featurelab-extractors-v2`, based on `7b3e5f8ac63096199969c75eb99a646d1be2cae8`.
- Keep `research/siren-a-stagepool-v1`, all source NPZs, the frozen 807-feature cache, and existing manifests read-only.
- Batch 1 reads only existing `timegrid_v2.1.0 / validity_v0.2` NPZs. It never calls `TimegridV2Extractor.extract_all()`, reads an EDF, recomputes old epoch features, or reads `y`, disease labels, TTE, predictions, or scores.
- Preserve finite valid zero; represent every invalid output as `NaN`, `valid_mask=False`, an explicit reason, and a nonnegative support count.
- Final cache granularity is exactly one record × one family × one recipe. Do not cache complete overnight waveforms.
- Recipe identity includes only actual numerical inputs and declared transitive module-qualified numerical code units/constants; Git HEAD remains provenance only.
- Tracked/public reports contain aggregate data and labeled synthetic examples only. Every real smoke value, threshold, support detail, and identity mapping stays outside Git.
- Use `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python`. Do not install, upgrade, or downgrade dependencies; use standard-library `unittest`, because this environment has no `pytest`.
- Never run all 6,600 records, full EDF extraction, historical G1–G5 regression, training, ranking, LOSO, or view generation.
- A distinct defect receives at most two targeted repair cycles. After two failed cycles, preserve the error, mark the affected unit `NEEDS_REVIEW`, and continue independent work.
- Run only the affected test module after a local edit. Run the complete extension suite once before the real smoke and once at final verification. Do not repeat an unchanged real calculation.
- The real smoke budget is one build, one readback, one identical resume, one isolated parameter-invalidation test, and one isolated missing-family recovery test.
- Stage only explicit paths; never use `git add .`, force push, merge to A, or copy FeatureLab code back to A.

## Review Focus

1. Path/NPZ identity disagreement, cross-site duplicate IDs, and a forced record-key collision must fail before computation; Task 1 tests all three.
2. Upstream finite zero fills must remain invalid when modality metadata says invalid, while valid physiological zero remains valid; Tasks 4, 6, 8, and 9 pin this behavior.
3. Wake, invalid epochs, and non-30-second time gaps must break temporal and CAISR adjacency without compressing the grid; Tasks 5 and 7 exercise each boundary.
4. Corrupt, stale, wrong-order, source-mismatched, or concurrently published caches must become `STOP_INVALID` or a no-clobber race result, never an overwrite; Task 3 tests these cases.
5. No real patient value or threshold may enter a tracked report, even if identifiers are hashed; Task 12 validates a public-report allowlist and private-report location.

---

## File Map

### Extension code

- `per_epoch_features/extensions/__init__.py` — stable public exports only.
- `per_epoch_features/extensions/contracts.py` — immutable source, family, recipe, result, and resume types plus invariant validation.
- `per_epoch_features/extensions/sources.py` — frozen-NPZ allowlist loader, record-key construction, source hashing, and inventory collision checks.
- `per_epoch_features/extensions/common.py` — stage definitions, time-grid adjacency, relative EEG power, linear quantiles, and Mean/IQR helpers.
- `per_epoch_features/extensions/recipe.py` — canonical recipe material, AST-normalized declared numerical-symbol/constant hashes, dependency audit, and recipe SHA256.
- `per_epoch_features/extensions/registry.py` — explicit ordered six-family registry, versions, configs, source requirements, and dependency declarations.
- `per_epoch_features/extensions/cache.py` — cache path, logical content digest, strict validation, no-clobber publication, and resume decision.
- `per_epoch_features/extensions/runner.py` — selected-family execution over one already-loaded source and CLI.
- `per_epoch_features/extensions/eeg_tail.py` — 288 EEG Q88/Q95 columns.
- `per_epoch_features/extensions/eeg_temporal.py` — 96 position-preserving temporal columns.
- `per_epoch_features/extensions/eeg_coherence_stage.py` — 90 staged coherence columns.
- `per_epoch_features/extensions/caisr_stage.py` — 25 transition plus 10 reused bout columns.
- `per_epoch_features/extensions/emg_stage.py` — 72 staged EMG columns.
- `per_epoch_features/extensions/resp_stage.py` — 56 staged respiration columns.
- `per_epoch_features/extensions/manifest.py` — deterministic candidate-manifest construction and frozen-ID overlap audit.
- `per_epoch_features/extensions/smoke.py` — bounded metadata-only source selection, synthetic cache checks, one-record smoke, and private/public summaries.

### Tests

- `per_epoch_features/extensions/tests/__init__.py`
- `per_epoch_features/extensions/tests/fixtures.py` — deterministic synthetic `SourceRecord` and NPZ builders with no patient data.
- `per_epoch_features/extensions/tests/test_contracts_sources.py`
- `per_epoch_features/extensions/tests/test_recipe.py`
- `per_epoch_features/extensions/tests/test_cache_runner.py`
- `per_epoch_features/extensions/tests/test_eeg_tail.py`
- `per_epoch_features/extensions/tests/test_eeg_temporal.py`
- `per_epoch_features/extensions/tests/test_eeg_coherence_stage.py`
- `per_epoch_features/extensions/tests/test_caisr_stage.py`
- `per_epoch_features/extensions/tests/test_emg_stage.py`
- `per_epoch_features/extensions/tests/test_resp_stage.py`
- `per_epoch_features/extensions/tests/test_manifest.py`
- `per_epoch_features/extensions/tests/test_smoke.py`
- `per_epoch_features/extensions/tests/test_document_contract.py`

### Tracked outputs

- `docs/featurelab/CACHE_SCHEMA.md`
- `docs/featurelab/FEATURELAB_BATCH1_REPORT.md`
- `docs/featurelab/BATCH2_PREPROCESSING_REVIEW.md`
- `audit/featurelab_batch1_manifest.csv`

Private caches, source inventories with identities, smoke values, and smoke thresholds remain under `/database/home/gaohaojie/workspace/feature-cache-extension` and are never staged.

---

### Task 1: Source contracts, safe record keys, and read-only NPZ loading

**Files:**
- Create: `per_epoch_features/extensions/__init__.py`
- Create: `per_epoch_features/extensions/contracts.py`
- Create: `per_epoch_features/extensions/sources.py`
- Create: `per_epoch_features/extensions/tests/__init__.py`
- Create: `per_epoch_features/extensions/tests/fixtures.py`
- Create: `per_epoch_features/extensions/tests/test_contracts_sources.py`

**Interfaces:**
- Produces: `SourceIdentity`, `SourceRecord`, `FeatureResult`, `FeatureLabError`, `SourceContractError`, `RecordKeyCollisionError`, `validate_feature_result(result)`.
- Produces: `record_key_for(site_id: str, record_id: str) -> str`.
- Produces: `load_source_npz(path: Path, source_root: Path, required_fields: Collection[str]) -> SourceRecord`.
- Produces: `build_source_inventory(paths: Iterable[Path], source_root: Path) -> tuple[SourceIdentity, ...]`.

- [ ] **Step 1: Write failing contract and source tests**

Add tests asserting that the same canonical tuple always yields `rec_[0-9a-f]{64}`, equal record IDs at two sites yield different keys, hostile identifier characters never enter the key, and a monkeypatched digest collision with different tuples raises `RecordKeyCollisionError`. Add synthetic NPZ tests for path/scalar mismatch, duplicate canonical identity with different file SHA, forbidden/absent fields, exact array shapes, `allow_pickle=False`, and the absence of `y` from `SourceRecord.arrays`. Add a `FeatureResult` invariant test covering finite-valid, NaN-invalid, valid zero, lengths, unique IDs, supports, and missing reasons.

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python -m unittest per_epoch_features.extensions.tests.test_contracts_sources -v`

Expected: import failure for the not-yet-created contracts/source interfaces.

- [ ] **Step 3: Implement the minimal source and result contracts**

Use canonical compact sorted JSON over namespace/site/record and the full SHA256 digest. Validate `<site>/<record>.npz` against NPZ scalars before reading arrays into the allowlist. Hash source bytes in chunks, store exact required shapes/dtypes, copy arrays needed after closing the NPZ, and never access `z["y"]`.

- [ ] **Step 4: Run the focused test and confirm GREEN**

Expected: all Task 1 tests pass; a before/after SHA and stat check proves the synthetic source is unchanged.

- [ ] **Step 5: Commit the task**

Stage only the six Task 1 files and commit `feat(featurelab): add strict read-only source contracts`.

**Acceptance:** deterministic safe keys, explicit collision detection, strict v2.1/v0.2 loading, no label exposure, and validated missingness invariants.

---

### Task 2: Common numerical helpers and recipe fingerprinting

**Files:**
- Create: `per_epoch_features/extensions/common.py`
- Create: `per_epoch_features/extensions/recipe.py`
- Create: `per_epoch_features/extensions/registry.py`
- Create: `per_epoch_features/extensions/tests/test_recipe.py`
- Modify: `per_epoch_features/extensions/tests/fixtures.py`

**Interfaces:**
- Consumes: Task 1 contract types.
- Produces: `FamilySpec` with family ID/version, extractor, default config, required fields, feature definitions, validity contract, and declared module-qualified numerical symbols/constants.
- Produces: `build_recipe_identity(spec: FamilySpec, config: Mapping[str, object], code_root: Path) -> RecipeIdentity`.
- Produces common helpers `stage_mask`, `adjacent_pairs`, `relative_eeg_power`, `linear_quantile`, and `pool_mean_iqr`.

- [ ] **Step 1: Write failing recipe and common-helper tests**

Assert canonical parameter ordering, fixed stage mappings, strict 30-second adjacency, linear quantiles, and lawful zero behavior. In a temporary code tree, assert that changing a declared family extractor, source-materialization callable, stage/adjacency helper, aggregation helper, numerical constant, validity contract, source column, or NumPy version changes the recipe. Assert that changing record-key logic, another undeclared function in the same module, `cache.py`, a document, test, other family, or Git SHA leaves it unchanged. Unresolved symbols, missing constant fingerprints, and an invoked numerical helper omitted from the declared dependency audit must raise.

- [ ] **Step 2: Run the focused test and confirm RED**

Run the `test_recipe` module; expect missing recipe/registry symbols.

- [ ] **Step 3: Implement explicit recipe material and helpers**

Fingerprint only normalized family material plus canonical AST definitions for the explicit module-qualified numerical symbol closure and serialized declared constants. Exclude comments, docstrings, locations, and undeclared symbols. Record Git SHA separately in provenance. Keep the initial registry declarative and empty of family imports until Tasks 4–9.

- [ ] **Step 4: Run the focused test and confirm GREEN**

Expected: relevant changes invalidate; unrelated changes do not.

- [ ] **Step 5: Commit the task**

Commit the five explicit Task 2 paths as `feat(featurelab): define numerical recipes and shared math`.

**Acceptance:** the recipe boundary covers source adaptation, stage validity, shared calculations, and family code without whole-repository invalidation.

---

### Task 3: Atomic family cache and selected-family runner

**Files:**
- Create: `per_epoch_features/extensions/cache.py`
- Create: `per_epoch_features/extensions/runner.py`
- Create: `per_epoch_features/extensions/tests/test_cache_runner.py`
- Create: `docs/featurelab/CACHE_SCHEMA.md`

**Interfaces:**
- Consumes: `SourceRecord`, `FeatureResult`, `FamilySpec`, and `RecipeIdentity`.
- Produces: `ResumeAction = BUILD_NEW | SKIP_VERIFIED | STOP_INVALID`.
- Produces: `cache_path(root, family_id, recipe_id, record_key) -> Path`, `read_cache`, `decide_resume_action`, `write_cache_atomic`, and `run_record(source_path, source_root, cache_root, family_ids, config_overrides) -> RunSummary`.

- [ ] **Step 1: Write failing cache and runner tests with dummy families**

Cover exact feature order, schema/version, family/recipe/source mismatch, typed logical payload hash, truncated/corrupt files, incomplete status, no-clobber publication, temporary-file cleanup, and a simulated concurrent winner. Assert that one existing invalid target stops only before overwrite and that requesting one family never invokes an unrequested extractor.

- [ ] **Step 2: Run the focused test and confirm RED**

Run `test_cache_runner`; expect missing cache/runner interfaces.

- [ ] **Step 3: Implement strict cache validation and publication**

Serialize non-object arrays and canonical JSON scalars. Write to a same-directory temporary file, flush/fsync, reload with `allow_pickle=False`, validate and digest, publish with a no-replace operation, fsync the directory, and remove only the owned temporary file. Never replace a final path.

- [ ] **Step 4: Add and pass dummy-family resume tests**

First run must be `BUILD_NEW`; identical second run must be `SKIP_VERIFIED` with zero extractor calls; a missing dummy family must build alone.

- [ ] **Step 5: Document the exact cache schema**

`CACHE_SCHEMA.md` lists keys, dtypes, invariants, digest construction, source identity, recipe provenance, decisions, and recovery behavior without patient values.

- [ ] **Step 6: Commit the task**

Commit the four Task 3 paths as `feat(featurelab): add atomic family cache runner`.

**Acceptance:** cache files are independently resumable, verifiable, no-clobber, and opaque to family internals.

---

### Task 4: EEG tail quantiles (288 columns)

**Files:**
- Create: `per_epoch_features/extensions/eeg_tail.py`
- Create: `per_epoch_features/extensions/tests/test_eeg_tail.py`
- Modify: `per_epoch_features/extensions/registry.py`
- Modify: `per_epoch_features/extensions/tests/fixtures.py`

**Interfaces:**
- Produces: `EEG_TAIL_SPEC` and `extract_family(source, config) -> FeatureResult`.
- Feature order: lead, band, stage scope, then Q88/Q95 using explicit tuples from the spec.

- [ ] **Step 1: Write failing mathematical and validity tests**

Assert 288 stable unique IDs, exact source offsets, NumPy linear Q88/Q95 equality, 19/20 support boundary, per-stage missingness, lead availability, clean-subsegment validity, valid zero retention, and invalid finite-zero rejection.

- [ ] **Step 2: Run `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python -m unittest per_epoch_features.extensions.tests.test_eeg_tail -v` and confirm RED**

- [ ] **Step 3: Implement only the approved NPZ-derived calculation**

Use Delta=4, Theta=3, Alpha=2, Sigma=1, beta=7, and P0=6 within each nine-column lead block. Do not compute PSD or read EEG waveforms.

- [ ] **Step 4: Run the same focused command and confirm GREEN**

- [ ] **Step 5: Commit**

Commit explicit Task 4 paths as `feat(featurelab): add EEG tail quantiles`.

**Acceptance:** 288 real formulas, no placeholders, linear quantiles, per-column support/missingness, and stable IDs.

---

### Task 5: EEG overnight temporal organization (96 columns)

**Files:**
- Create: `per_epoch_features/extensions/eeg_temporal.py`
- Create: `per_epoch_features/extensions/tests/test_eeg_temporal.py`
- Modify: `per_epoch_features/extensions/registry.py`
- Modify: `per_epoch_features/extensions/tests/fixtures.py`

**Interfaces:**
- Produces: `EEG_TEMPORAL_SPEC` and `extract_family(source, config) -> FeatureResult`.
- Produces 24 lead-band groups × `range80`, `drift`, `tail_hour_concentration`, and `tail_entry_rate_per_hour`.

- [ ] **Step 1: Write failing position-preserving tests**

Construct raw grids where removing Wake/NaN would change thirds, create 119- and 120-position windows, create windows with 95/96 valid sleep epochs, and insert a non-30-second timestamp gap. Assert strict `> Q88`, zero entries for constants, no transition across Wake/invalid/gap, first/last-third support of 19/20, and denominator conversion from legal 30-second pairs to hours.

- [ ] **Step 2: Run `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python -m unittest per_epoch_features.extensions.tests.test_eeg_temporal -v` and confirm RED**

- [ ] **Step 3: Implement the four approved statistics on raw indices**

Calculate the Q88 reference on all valid sleep observations, but never compress positions for thirds, windows, or adjacency.

- [ ] **Step 4: Run the same focused command and confirm GREEN**

- [ ] **Step 5: Commit**

Commit explicit Task 5 paths as `feat(featurelab): add position-preserving EEG temporal features`.

**Acceptance:** 96 columns, original time grid retained, one-hour coverage enforced, and transition rate based only on legal adjacent observation time.

---

### Task 6: Stage-specific EEG coherence (90 columns)

**Files:**
- Create: `per_epoch_features/extensions/eeg_coherence_stage.py`
- Create: `per_epoch_features/extensions/tests/test_eeg_coherence_stage.py`
- Modify: `per_epoch_features/extensions/registry.py`
- Modify: `per_epoch_features/extensions/tests/fixtures.py`

**Interfaces:**
- Produces: `EEG_COHERENCE_STAGE_SPEC` and `extract_family(source, config) -> FeatureResult`.

- [ ] **Step 1: Write failing column and support tests**

Assert pair positions 0/9/14, band offsets 0–4 within each 24-column pair, stages REM/N2/N3, Mean/IQR, 4/5 support, both-lead availability, `eeg_common_clean_subsegment_count > 0`, valid zero retention, and invalid upstream zero exclusion.

- [ ] **Step 2: Run `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python -m unittest per_epoch_features.extensions.tests.test_eeg_coherence_stage -v` and confirm RED**

- [ ] **Step 3: Implement the approved reuse path**

Read only `X_seq[:,54:414]` plus verified stage/channel/common-clean metadata; never call coherence or EEG waveform code.

- [ ] **Step 4: Run the same focused command and confirm GREEN**

- [ ] **Step 5: Commit**

Commit explicit Task 6 paths as `feat(featurelab): add staged coherence pooling`.

**Acceptance:** 90 columns use exact stored mean-coherence columns and recover upstream invalid-zero semantics.

---

### Task 7: CAISR transitions and reused bout statistics (35 columns)

**Files:**
- Create: `per_epoch_features/extensions/caisr_stage.py`
- Create: `per_epoch_features/extensions/tests/test_caisr_stage.py`
- Modify: `per_epoch_features/extensions/registry.py`
- Modify: `per_epoch_features/extensions/tests/fixtures.py`

**Interfaces:**
- Produces: `CAISR_STAGE_SPEC` and `extract_family(source, config) -> FeatureResult`.
- Internal pure helper: `bout_lengths_by_stage(stage_code, stage_valid, epoch_start_sec) -> Mapping[int, np.ndarray]`.

- [ ] **Step 1: Write failing transition tests**

Assert source-major N3/N2/N1/REM/Wake ordering, self-transition inclusion, row normalization, no transition across invalid or timestamp gaps, and five NaNs for a source row with zero legal outgoing transitions.

- [ ] **Step 2: Write the required static-versus-grid semantic test**

Create a synthetic aligned grid containing every stage, repeated bouts, an invalid gap inside an otherwise same-stage run, and a timestamp gap. Directly compute stage bout seconds and NumPy-linear P50/P95. Place those values at exact `x_static[44:49]` and `x_static[69:74]` positions in upstream Wake/N1/N2/N3/REM order. Assert the extractor returns identical values after N3/N2/N1/REM/Wake reordering, with bout counts as support. Also assert absent stages convert upstream zero fills to invalid NaN.

- [ ] **Step 3: Run `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python -m unittest per_epoch_features.extensions.tests.test_caisr_stage -v` and confirm RED**

- [ ] **Step 4: Implement transitions, bout support, and exact static reuse**

Use the raw valid grid only to define transitions, bout boundaries, support, and semantic validation. Read P50/P95 values from the existing static indices; do not read CAISR EDF or recompute static output for production.

- [ ] **Step 5: Run the same focused command and confirm GREEN**

- [ ] **Step 6: Commit**

Commit explicit Task 7 paths as `feat(featurelab): add CAISR stage dynamics`.

**Acceptance:** 25 transition probabilities and 10 static bout values have correct stage order, gap semantics, source indices, validity, and support.

---

### Task 8: Stage-specific EMG descriptions (72 columns)

**Files:**
- Create: `per_epoch_features/extensions/emg_stage.py`
- Create: `per_epoch_features/extensions/tests/test_emg_stage.py`
- Modify: `per_epoch_features/extensions/registry.py`
- Modify: `per_epoch_features/extensions/tests/fixtures.py`

**Interfaces:**
- Produces: `EMG_STAGE_SPEC` and `extract_family(source, config) -> FeatureResult`.

- [ ] **Step 1: Write failing column, stage, and modality-validity tests**

Assert channel block offsets 0/8/16, selected offsets 0/3/4/6, Wake/REM/NREM grouping, Mean/IQR, 4/5 support, channel availability, preprocessing success, epoch success, valid zero retention, and independent per-channel missingness.

- [ ] **Step 2: Run `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python -m unittest per_epoch_features.extensions.tests.test_emg_stage -v` and confirm RED**

- [ ] **Step 3: Implement pooling over stored EMG24 only**

Do not call EMG preprocessing or redefine baseline, burst, tonic, or phasic thresholds.

- [ ] **Step 4: Run the same focused command and confirm GREEN**

- [ ] **Step 5: Commit**

Commit explicit Task 8 paths as `feat(featurelab): add staged EMG descriptions`.

**Acceptance:** 72 columns use only stored EMG values and exact channel/epoch validity.

---

### Task 9: Stage-specific respiration descriptions (56 columns)

**Files:**
- Create: `per_epoch_features/extensions/resp_stage.py`
- Create: `per_epoch_features/extensions/tests/test_resp_stage.py`
- Modify: `per_epoch_features/extensions/registry.py`
- Modify: `per_epoch_features/extensions/tests/fixtures.py`

**Interfaces:**
- Produces: `RESP_STAGE_SPEC` and `extract_family(source, config) -> FeatureResult`.

- [ ] **Step 1: Write failing feature-mask and stage tests**

Assert all 14 stored columns, REM/NREM, Mean/IQR, 4/5 support, per-feature `resp_feature_valid`, independent missing channels, valid zero retention, and invalid finite-zero exclusion.

- [ ] **Step 2: Run `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python -m unittest per_epoch_features.extensions.tests.test_resp_stage -v` and confirm RED**

- [ ] **Step 3: Implement pooling over stored Resp14 only**

Do not filter, resample, or call the old Resp extractor.

- [ ] **Step 4: Run the same focused command and confirm GREEN**

- [ ] **Step 5: Commit**

Commit explicit Task 9 paths as `feat(featurelab): add staged respiration descriptions`.

**Acceptance:** 56 columns preserve the exact per-feature validity contract and minimum support.

---

### Task 10: Six-family registry, manifest, and synthetic integration

**Files:**
- Create: `per_epoch_features/extensions/manifest.py`
- Create: `per_epoch_features/extensions/tests/test_manifest.py`
- Modify: `per_epoch_features/extensions/registry.py`
- Modify: `per_epoch_features/extensions/tests/test_cache_runner.py`
- Create: `audit/featurelab_batch1_manifest.csv`

**Interfaces:**
- Produces: `get_family(family_id)`, `list_families() -> tuple[FamilySpec, ...]`, `build_manifest_rows()`, and `write_manifest(path)`.
- Consumes: all six family specs and the frozen 807-ID manifest.

- [ ] **Step 1: Write failing dimension, ordering, and overlap tests**

Assert ordered family dimensions 288/96/90/35/72/56, total 637, 637 globally unique IDs, no intersection with the 807 frozen unique IDs, exact manifest columns/statuses, and proposed-view additions V1=48, V2=48, V3=144, V4=144, V5=253.

- [ ] **Step 2: Add failing six-family recipe-isolation tests**

Using a synthetic source and isolated cache root, assert that altering only EEG-tail quantiles changes only its recipe/path; the other five recipes and files remain identical and return `SKIP_VERIFIED`.

- [ ] **Step 3: Run `test_manifest` and `test_cache_runner` with separate `python -m unittest ... -v` commands and confirm RED**

- [ ] **Step 4: Complete the explicit registry and manifest generator**

Declare each family's required source fields, validity material, config, feature definitions, and module-qualified numerical dependency symbols/constants. Generate the CSV deterministically; never modify the frozen manifest.

- [ ] **Step 5: Run the same two focused commands and confirm GREEN**

- [ ] **Step 6: Commit**

Force-add only the ignored audit CSV plus the explicit code/tests, then commit `feat(featurelab): register batch1 families and manifest`.

**Acceptance:** registry, recipes, manifest, dimensions, future view mapping, and old-ID non-overlap close exactly at 637.

---

### Task 11: Batch 2 preprocessing-difference audit

**Files:**
- Create: `docs/featurelab/BATCH2_PREPROCESSING_REVIEW.md`
- Create: `per_epoch_features/extensions/tests/test_document_contract.py`

**Interfaces:**
- Consumes: repository-local extractor code and the user-approved external reference sources.
- Produces: review-only documentation; no Batch 2 implementation or cache.

- [ ] **Step 1: Write a failing document-contract test**

Require one section/table row for EEG spectral 288, EEG complexity 192, ECG/fragmentation 20, CAISR intervals 9, Resp 24, and SpO2 20. For each require source channel, current reader/preprocessing, candidate preprocessing, sample rate, filters, windows, validity, threshold reference, reuse decision, new branch, intermediate cache, EDF need, cost, overlap, unresolved parameters, expected dimension, reference/adaptation note, and status restricted to `READY_TO_IMPLEMENT` or `REQUIRES_REVIEW`.

- [ ] **Step 2: Run `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python -m unittest per_epoch_features.extensions.tests.test_document_contract -v` and confirm RED**

- [ ] **Step 3: Audit local preprocessing and references without copying files**

Trace the current local EEG, ECG/NN, Resp, CAISR event, and SpO2 paths. Compare exact semantics with the cited ID565 algorithms and reliable library documentation. Mark any concatenation of separated epochs, resampling/filter difference, event-definition ambiguity, or unavailable dependency `REQUIRES_REVIEW`.

- [ ] **Step 4: Write the review document**

Include the requested comparison matrix, risks, reusable intermediate-cache recommendations, and parameter questions. Do not invent formulas to fill the 553-column budget and do not add code.

- [ ] **Step 5: Run the same focused command and confirm GREEN**

- [ ] **Step 6: Commit**

Force-add the two explicit paths and commit `docs(featurelab): audit batch2 preprocessing differences`.

**Acceptance:** all 553 proposed dimensions are accounted for as audit scope, with no claim that Batch 2 is implemented.

---

### Task 12: Bounded synthetic cache exercises and one real-record smoke

**Files:**
- Create: `per_epoch_features/extensions/smoke.py`
- Create: `per_epoch_features/extensions/tests/test_smoke.py`
- Create: `docs/featurelab/FEATURELAB_BATCH1_REPORT.md`
- Modify: `per_epoch_features/extensions/runner.py`

**Interfaces:**
- Produces: `select_smoke_source(source_root, max_candidates=80) -> SourceIdentity`.
- Produces: `run_synthetic_cache_checks(temp_root) -> SyntheticCacheCheckSummary`.
- Produces: `run_real_smoke(source_path, source_root, cache_root, private_report_path) -> PublicSmokeSummary`.

- [ ] **Step 1: Write failing smoke-budget and privacy tests**

Assert selection reads at most 80 NPZ metadata records, never reads `y`, uses only modality/stage support, and is deterministic. Assert the public summary schema rejects feature values, thresholds, record/path/site IDs, per-column supports, and identity mappings; assert the private report is outside the repository and includes the required diagnostic values.

- [ ] **Step 2: Write failing synthetic cache A/B/C tests**

Test A: first six-family build followed by an identical run returns six `SKIP_VERIFIED` decisions and zero extractor calls. Test B: change only EEG-tail configuration and assert one new recipe/build while five families skip unchanged. Test C: copy caches to a temporary root, remove one family result there, and assert only that family rebuilds. All destructive simulation targets must be under `TemporaryDirectory`.

- [ ] **Step 3: Run `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python -m unittest per_epoch_features.extensions.tests.test_smoke -v` and confirm RED**

- [ ] **Step 4: Implement bounded smoke orchestration and public/private schemas**

Keep real values and the record mapping only in the private report. Public output contains family dimensions, valid/missing counts, aggregate support summaries, elapsed seconds, cache bytes, resume decisions, and labeled synthetic examples.

- [ ] **Step 5: Run every extension test once before real data**

Run: `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python -m unittest discover -s per_epoch_features/extensions/tests -v`

Expected: all extension tests pass. If a failure occurs, apply at most two targeted repair cycles to its owning module; do not rerun unaffected tests during the repair.

- [ ] **Step 6: Snapshot and select one real source without labels**

Record the source file SHA256/stat privately, select from the verified NPZ inventory using only the bounded support rules, and record which families the selected source can exercise. Do not search EDFs or inspect model/training artifacts.

- [ ] **Step 7: Run exactly one real six-family build**

Use cache root `/database/home/gaohaojie/workspace/feature-cache-extension` and a private report below that root. Capture family dimensions, valid/missing counts, support, time range, required example values/thresholds, elapsed time, and bytes privately.

- [ ] **Step 8: Perform one readback and one identical resume**

Strictly reload all produced caches, then run the identical request once. Require six `SKIP_VERIFIED` decisions and zero extractor calls. Do not run the same build again.

- [ ] **Step 9: Verify source immutability**

Compare source SHA256/stat to the private pre-smoke snapshot. Any change is a hard stop.

- [ ] **Step 10: Write the public Batch 1 report**

Report only aggregates and labeled synthetic numerical examples. Include each family's plan/implemented dimension, valid/missing counts, source type, EDF=no, smoke status, review flags, overlap result, elapsed time, bytes, and resume behavior. Never copy real values or thresholds from the private report.

- [ ] **Step 11: Commit**

Stage only `smoke.py`, its test, runner changes, and the public report; commit `test(featurelab): complete bounded batch1 smoke`.

**Acceptance:** synthetic A/B/C pass, one real record is built/read/resumed without EDF or label access, six family caches validate, source NPZ is unchanged, and public artifacts contain no real-patient values.

---

### Task 13: Final bounded verification and Git submission

**Files:**
- Modify only if evidence requires: `docs/featurelab/FEATURELAB_BATCH1_REPORT.md`
- Verify: every tracked FeatureLab path from Tasks 1–12.

**Interfaces:**
- Consumes: completed extension, private smoke evidence, tracked reports and manifest.
- Produces: final clean branch with matching local/remote SHA; no A-branch migration.

- [ ] **Step 1: Run the complete extension suite once**

Run the same `unittest discover` command from Task 12. Expected: all tests pass. Do not repeat it if unchanged.

- [ ] **Step 2: Run the one relevant frozen-contract regression**

Run: `PYTHONNOUSERSITE=1 /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python reproduction/siren_a/audit/test_g1b_contract.py`

Expected: 13 tests pass. Do not run other G1–G5 suites.

- [ ] **Step 3: Run static scope and privacy checks**

Confirm no tracked EDF/NPZ/cache/patient table/private report/model/prediction/temp file; no changed old manifest, A/G1–G5 module, extractor default, or five-view file; no production import of label/diagnosis/model-score readers; and exactly 637 manifest rows/IDs.

- [ ] **Step 4: Verify workspace and cache evidence**

Confirm FeatureLab branch/path/source commit, A worktree SHA and clean status, old NPZ hash equality, old 807 cache untouched, six cache files, family/recipe counts, cached bytes, six repeated skips, one-family invalidation, and one-family recovery.

- [ ] **Step 5: Update report statuses from evidence only**

Use `COMPLETE/PASS` only for verified units; retain `REQUIRES_INPUT_REVIEW` or `NEEDS_REVIEW` verbatim for unresolved units. State Batch 2 is `READY_FOR_REVIEW`, not implemented.

- [ ] **Step 6: Stage explicit paths and commit final report adjustments**

List every staged path before committing. Never use `git add .`.

- [ ] **Step 7: Push the FeatureLab branch and verify SHA equality**

Push without force, query `refs/heads/research/featurelab-extractors-v2`, and require local HEAD equals remote SHA.

- [ ] **Step 8: Stop**

Do not merge, migrate, train, score, run LOSO, implement Batch 2, or start slow-wave/spindle/coupling work. Return the required five-part status report and wait for review.

**Acceptance:** evidence-backed final report, bounded tests only, clean FeatureLab and A worktrees, matching remote SHA, and no work beyond the authorized deliverables.
