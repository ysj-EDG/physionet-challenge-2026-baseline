# G3-A-V Independent Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove independently that all six frozen G3-A Rank256 units can be reconstructed exactly from the validated G2-B1 cache and identity mapping.

**Architecture:** `verify_rank.py` contains pure independent reference, percentile, and Rank256 mathematics. `verify_g3a.py` owns strict source loading, identity/index/unit/audit verification, private no-clobber evidence, and the CLI; it parses the Manifest directly and never calls old G3-A mathematical helpers.

**Tech Stack:** Python 3.9, NumPy 1.26, Python standard library unittest/csv/json/hashlib/pathlib.

**Spec:** `reproduction/siren_a/docs/G3AV_VERIFICATION_SPEC.md`

## Global Constraints

- Add exactly eight approved tracked files; modify no existing tracked file.
- Treat G2-B1 and G3-A private roots and all G1/G2/G3-A files as read-only.
- Do not read source physiological NPZ/EDF, age, labels, diagnoses, follow-up, or outcomes.
- Do not call old ranking, quantization, views, audit, preparation, or validation functions as the independent mathematical result.
- Require exact identities, exact masks/indexes/levels, exact NaN placement, and exact valid float values.
- Use `np.load(..., allow_pickle=False)`; forbid object arrays and symlinks.
- Publish only aggregate public output; keep identities, paths, matrices, and diagnostic indexes private.
- No model adapter, training, scoring, normalization selection, or G3-B/G4/G5 work.

## Review Focus

- Same-site index swaps must fail even when counts and sets remain plausible.
- Rehashed tampered payloads must fail independent mathematical verification.
- References with zero, one, and six observations must preserve frozen missing/tie semantics.
- Adjacent but self-consistent Rank256 level changes must fail formula verification.
- Private identifiers or paths must never enter the public CSV/report.

---

### Task 1: Source identity and frozen Manifest

**Files:**
- Create: `reproduction/siren_a/pipeline/test_verify_g3a.py`
- Create: `reproduction/siren_a/pipeline/verify_g3a.py`

**Interfaces:**
- Produces: `load_manifest_contract(path)`, `load_sources(config)`, `check_identity(source)`.
- Consumes later: validated 6,600×807 values/masks, feature IDs, site vector, fingerprints.

- [ ] Write tests that inject index swaps, site mismatch, cache-name reorder, duplicate mapping, malformed cache provenance, symlink, and wrong code SHA.
- [ ] Run the tests and confirm RED because verifier interfaces do not exist.
- [ ] Implement direct Manifest parsing, canonical JSON/SHA helpers, strict Index/Inventory closure, cache schema/provenance checks, and G3-A fingerprint/code identity checks.
- [ ] Run Task 1 tests GREEN and verify real preflight returns 6,600 records and the expected cache fingerprint.

### Task 2: LOSO exact indexes and unit integrity

**Files:**
- Modify: `reproduction/siren_a/pipeline/test_verify_g3a.py`
- Modify: `reproduction/siren_a/pipeline/verify_g3a.py`

**Interfaces:**
- Produces: `rebuild_loso(sites)`, `check_indices(unit, fold)`, `check_unit_integrity(unit, identity)`.

- [ ] Add tests for same-site swaps, wrong-site replacement, correct-length wrong arrays, overlap, out-of-range indexes, broken symlink, incomplete directory, corrupt NPZ, and rehashed payload tampering.
- [ ] Run focused tests RED.
- [ ] Implement exact ordered index comparison, manifest/metadata agreement, strict file set/SHA/object-dtype checks, and incomplete-directory rejection.
- [ ] Run focused tests GREEN.

### Task 3: Independent reference, percentile, and Rank256 mathematics

**Files:**
- Create: `reproduction/siren_a/pipeline/test_verify_rank.py`
- Create: `reproduction/siren_a/pipeline/verify_rank.py`

**Interfaces:**
- Produces: `build_reference(values, mask)`, `rank_values(values, mask, reference)`, `quantize_values(percentile, rank_mask)`, `check_reference(saved, expected, prefix)`.

- [ ] Add literal expected-value tests for ties, continuous/binary/constant/singleton/all-missing, zero, NaN, invalid Inf, below/above range, and order invariance.
- [ ] Add mutation tests for deleted/duplicated/reordered reference values, bad offsets/counts/fingerprint, mixed inductive reference, first-rank, wrong denominator, filled NaN, merged masks, banker's rounding, adjacent level, self-consistent wrong k/q, and off-grid q.
- [ ] Run tests RED because independent math module is absent.
- [ ] Implement unique-count/cumulative reference construction, compatible reference serialization fingerprint, independent rank query, and half-up Rank256.
- [ ] Run tests GREEN and mutation-check each frozen branch.

### Task 4: Full unit matrix and view verification

**Files:**
- Modify: `reproduction/siren_a/pipeline/test_verify_g3a.py`
- Modify: `reproduction/siren_a/pipeline/verify_g3a.py`

**Interfaces:**
- Produces: `verify_unit(source, fold, protocol, unit_path)` with aggregate mismatch counters.

- [ ] Add synthetic unit tests where percentile, quantized, reference, or view payload and completion SHA are jointly tampered.
- [ ] Run tests RED.
- [ ] Implement per-unit reference reconstruction, exact source/rank mask and percentile comparison, exact Rank256 comparison, independent Manifest view mapping, Demo5-repeat checks, and cross-protocol train invariants.
- [ ] Run tests GREEN.

### Task 5: Audit CSV closure

**Files:**
- Modify: `reproduction/siren_a/pipeline/test_verify_g3a.py`
- Modify: `reproduction/siren_a/pipeline/verify_g3a.py`

**Interfaces:**
- Produces: `verify_audit(unit_result, private_rows, public_rows, coverage_rows)`.

- [ ] Add tests for wrong totals, masks, reference counts, occupied bins, ties, errors, overflow, duplicate keys, missing rows, and rehashed altered private audit.
- [ ] Run tests RED.
- [ ] Implement independent per-column audit recomputation and exact private/public/G2 coverage closure.
- [ ] Run tests GREEN, including BMI, entropy, 5,121,732 source-valid positions, and 9,684 public rows.

### Task 6: Private evidence and recovery

**Files:**
- Modify: `reproduction/siren_a/pipeline/test_verify_g3a.py`
- Modify: `reproduction/siren_a/pipeline/verify_g3a.py`

**Interfaces:**
- Produces: atomic private evidence writer, completion verifier, CLI modes `--preflight-only|--smoke|--full|--verify-output`.

- [ ] Add tests for no-clobber, interrupted output, mismatched verifier/source identity, file hash drift, safe repeated verification, and public privacy.
- [ ] Run tests RED.
- [ ] Implement code/source identity, aggregate JSON/CSV publication, completion-last semantics, output verification, and STOP_INVALID behavior.
- [ ] Run tests GREEN.

### Task 7: Real I0002 smoke

**Files:** No tracked-file change.

- [ ] Record SHA256 snapshots of G2-B1/G3-A roots' immutable identity and completion evidence.
- [ ] Run preflight-only, then full I0002 Transductive and Inductive reconstruction over all 807 columns.
- [ ] Require zero index/reference/percentile/level/view/audit mismatch and exact mask equality.
- [ ] Read back private smoke evidence and verify its completion marker.

### Task 8: Full six-unit verification

**Files:** No tracked-file change.

- [ ] Run single-worker full verification for I0002, I0006, and S0001 under both protocols.
- [ ] Independently re-read private evidence and verify six aggregate rows and all hashes.
- [ ] Confirm known BMI, entropy, overflow, valid-position, and maximum-error values from new computation.
- [ ] Confirm G2-B1/G3-A source SHA snapshots remain unchanged.

### Task 9: Public outputs and complete regression

**Files:**
- Create: `reproduction/siren_a/audit/g3av_verification_summary.csv`
- Create: `reproduction/siren_a/docs/G3AV_VERIFICATION_REPORT.md`

- [ ] Generate only six aggregate public rows and a privacy-safe report from verified private evidence.
- [ ] Run G3-A-V tests, all required G1/G2/G3-A regressions, Python syntax checks, privacy scan, and diff-scope checks.
- [ ] Perform a separate whole-branch self-review because no subagent tool is available; record any ruling or deferred minor.
- [ ] Run final independent `--verify-output` and source-fingerprint checks after report creation.

### Task 10: Commit, push, and stop

**Files:** Exactly the eight approved files.

- [ ] Explicitly stage the four code/test files, aggregate CSV, and three ignored docs by exact path.
- [ ] Verify staged file list, diff, no private artifacts, and no existing tracked-file modification.
- [ ] Commit with `test(siren-a): independently verify G3-A rank256 artifacts`.
- [ ] Push normally; verify local/remote SHA equality and clean worktree.
- [ ] Stop before G3-B/G4/G5.
