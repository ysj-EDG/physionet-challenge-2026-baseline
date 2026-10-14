# G3-A LOSO Rank256 Implementation Plan

## Constraints

- Add only approved G3-A pipeline, test, specification, report and aggregate-audit files.
- Keep every G1/G2 tracked file and all private source caches read-only.
- Use only existing Python 3.9, NumPy and standard-library facilities.
- Develop each task RED then GREEN; stop on any frozen-input or recovery inconsistency.

## Task 1 — Data and LOSO

- [x] Add failing tests for 807-column loading, source-mask preservation, fingerprint validation, coverage closure, three complete folds and largest-site training selection.
- [x] Implement `data.py` and `split.py`.
- [x] Validate the real 6,600-cache input without writing G3 artifacts.

## Task 2 — Percentile mathematics

- [x] Add failing tests for average ties, continuous/binary/constant/singleton/all-missing columns, zeros, negatives, NaNs, Inf rejection and order invariance.
- [x] Implement transductive site ranks and explicit inductive fit/apply references.
- [x] Prove train output equality and holdout-reference isolation.

## Task 3 — Rank256

- [x] Add failing tests for half-up boundaries, 0.5 to 128/255, range, monotonicity, error bound, missing sentinel and no more than 256 levels.
- [x] Implement int16 levels and float64 grid outputs.

## Task 4 — Views and audit

- [x] Add failing tests for 807-to-827 mapping, view sizes, Demo5 repetition and audit invariants.
- [x] Implement `views.py` and `audit.py` with private detail/public-safe summaries.

## Task 5 — Recovery runner

- [x] Add failing tests for explicit modes, no-clobber, interrupted unit rejection, valid-unit skip, provenance drift and private/public boundaries.
- [x] Implement `run_prep.py`, immutable run identity, atomic unit publication and independent verification.

## Task 6 — Real smoke

- [x] Run preflight and exact G2 coverage closure.
- [x] Build S0001-to-I0002 Transductive and Inductive units.
- [x] Verify 807 columns, train equality, protocol difference, reference isolation, views and all audit invariants.

## Task 7 — Full six units

- [x] Build both protocols for holdouts I0006 and S0001.
- [x] Independently verify six units, references, matrices, views, masks, hashes and completion markers.
- [x] Re-run build to prove all six units safely skip.

## Task 8 — Report and Git

- [x] Generate the public feature-level aggregate CSV and report without patient information.
- [x] Run all G1/G2/G3 tests, syntax, privacy, fingerprint and diff checks.
- [ ] Explicitly stage approved files, commit, push, verify local/remote SHA and stop before G3-B/G4.
