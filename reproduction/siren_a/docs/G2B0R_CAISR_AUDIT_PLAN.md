# G2-B0-R CAISR EDF Audit Implementation Plan

> Execute this approved plan in the dedicated SIREN workspace using test-first,
> evidence-preserving changes. Stop after this gate.

**Goal:** Complete the real 6,600-record CAISR EDF census and correct the two
bounded G2-B0 safety defects without changing frozen features or historical
caches.

**Architecture:** A standalone audit module constructs exact EDF/NPZ paths,
reads only the CAISR stage channel and authorized NPZ metadata, writes verified
private checkpoints, and emits aggregate summaries. Existing cache safety and
pilot modules receive narrowly scoped corrections.

**Tech stack:** Python 3.9, NumPy 1.26, edfio 0.4, csv/json/hashlib, unittest.

## Task 1 — EDF mapping and reader

1. Add failing synthetic tests for missing/unreadable/no-stage files, normalized
   labels, valid and invalid codes, 19/20/21 boundaries, and sample-frequency
   warnings.
2. Implement strict three-column identity loading and exact, contained paths.
3. Implement lazy `stage_caisr` loading and mutually exclusive file statuses.
4. Run the focused test module and a small real three-site compatibility smoke.

## Task 2 — NPZ alignment and full audit

1. Add failing tests for shorter/longer source vectors, raw>=20/aligned<20,
   annotation/stage matches and mismatches, count closure, and safe recovery.
2. Implement authorized-key NPZ reads and NaN-aware cross-validation.
3. Implement fixed sorted chunks, atomic no-clobber publication, and verified
   resume tied to source/code identity.
4. Run the 6,600-record single-worker audit, then verify site/overall closure
   and generate only private record-level artifacts.

## Task 3 — Resume safety correction

1. Add RED tests for invalid mask/value/support dtypes and shapes, negative or
   family-inconsistent support, corrupted NPZ, provenance changes, and a valid
   cache.
2. Make array validation short-circuit safely before indexing.
3. Enforce Manifest-family support semantics and convert all malformed caches
   to an invalid validation result / `STOP_INVALID`.
4. Re-run focused and prior G2-B0 safety tests.

## Task 4 — Annotation-missing correction

1. Replace the old synthetic premise with upstream-faithful all-NaN stage code
   and all-false stage validity.
2. Assert EEG648, Spatial36, and CAISR20 missing; HRV22, EMG48, Resp28, and
   Demo5 independently computed; record retained.
3. Update boundary selection to identify annotation missing without requiring
   stage-conditioned EEG output.
4. Re-run pilot tests without rebuilding accepted Pilot or Golden caches.

## Task 5 — Verification, reporting, and Git

1. Verify 18/18 Golden values, masks, and support arrays and compare Golden and
   accepted Pilot file SHA256 snapshots before/after.
2. Run G1-A, G1-B, G2-A, G2-B0, new tests, and Python syntax checks.
3. Build the aggregate public report, including raw/aligned distributions,
   EDF/NPZ consistency, the SIREN 18-record comparison, and unresolved issues.
4. Confirm exactly the nine approved tracked files changed, explicitly stage
   them, commit with the approved message, push normally, and verify local and
   remote SHA equality.

## Global constraints

- Never read expert/human annotation sources, labels, age, diagnoses, or
  follow-up data.
- Never generate the 6,600-record night cache, train a model, alter source EDF/
  NPZ, or modify Golden/Pilot caches.
- Never modify the frozen Manifest/Spec or approved files outside this gate.
- Public artifacts contain aggregate statistics only.

