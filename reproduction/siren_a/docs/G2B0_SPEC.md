# G2-B0 Full-Run Readiness and Targeted Pilot Specification

## Scope and frozen boundary

G2-B0 verifies that the reviewed one-record G2-A calculation can safely scale later; it does not run the 6,600-record cache build. The only feature contract is `G1B_FEATURE_MANIFEST_FROZEN.csv` SHA256 `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a` and `G1B_FEATURE_SPEC_FROZEN.md` SHA256 `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a`. The 807 unique features, 827 view positions, formulas, source indices, float64 calculations, support threshold five, NaN semantics, and strict stage-entropy rule remain unchanged.

The only source root is the audited `timegrid_v2.1.0` / `validity_v0.2` 6,600-record cache. Source NPZs, the 18 G2-A Golden caches, their private mapping, and all historical files are read-only. No label, diagnosis, follow-up, age-based selection, imputation, scaling, site transformation, record drop, model operation, or full-run mode is permitted.

## Prepared single-record calculation

`compute_night_features(record, manifest) -> NightFeatureResult` remains the public interface. It prepares `X_seq`, `X_ecg`, and `x_static` as float64 once per record and prepares reusable stage and lead masks once. Internal row calculations consume that prepared state. The compatibility interface `spatial_value_and_mask(row, record)` remains available and delegates to the same prepared implementation.

The optimization may change allocation and dispatch only. It must not change operation order within Mean, linear-interpolation IQR, relative-power formulas, same-epoch spatial intersections, validity masks, support counts, output dtypes, unique order, or view order. Every one of the 18 Golden records must compare exactly, including NaN positions, finite float64 values, bool masks, int64 supports, and every view array.

## Cache publication and validation

`write_cache_atomic` writes a same-directory temporary NPZ, fsyncs it, reloads it with `allow_pickle=False`, and verifies exact keys, arrays, dtypes, shapes, masks, supports, finite/NaN invariants, and required provenance. Publication uses a same-filesystem hard link from the verified temporary file to the final name. `link(2)` supplies atomic create-if-absent semantics: an existing target or a concurrent winner causes `FileExistsError`, never replacement. The directory is fsynced after publication, and the temporary file is removed in all paths.

No partially written final file is accepted. G2-A's generic writer interface remains compatible; strict night-cache structure is enforced by the G2-B0 safety validator.

## Resume and provenance

`validate_existing_cache(path, expected_provenance, manifest)` returns a structured result after validating cache schema `g2b0_pilot_v1`, 807/827 dimensions, dtypes, value/mask/support invariants, unique-to-view mapping, and all expected scalar provenance. `decide_resume_action(...)` has exactly three outcomes:

- `COMPUTE`: target does not exist.
- `SKIP_VERIFIED`: the complete target and provenance validate.
- `STOP_INVALID`: any existing target is corrupt, partial, mismatched, unreadable, or from different data/code.

Invalid targets are never deleted or overwritten. Checkpoints use the same no-clobber publication rule and describe only completed anonymous cache names. Pilot provenance records source NPZ, frozen Manifest and Spec, `g2a_features.py`, `g2a_io.py`, `g2b0_batch_safety.py`, and `g2b0_pilot.py` SHA256 values, Python and NumPy versions, schema version, and seed. Execution fingerprints are captured before calculation and are not rewritten after execution.

## Golden evidence and private outputs

The G2-A Golden root is read-only. Before Pilot execution, all 18 cache files are safely loaded, validated, source-checked, and SHA256-snapshotted. The snapshot is private and distinguishes the historical G2-A script fingerprint from current G2-B0 code fingerprints.

The only new private target is `/database/home/gaohaojie/workspace/python-example-2026-siren-a-g2b0-private/g2b0_pilot_v1`. An existing target is a hard stop. It contains anonymous Pilot caches, identity mappings, Golden fingerprints, inventory, detailed probability QC, benchmark evidence, checkpoints, and private logs. Public files contain aggregates only and no patient key, filename, path, age, label, feature vector, or recoverable ordering.

## Targeted Pilot selection

Seed `20261013` governs deterministic ordering. Eighteen Golden sources count toward the maximum of 80 distinct NPZ contents read. Up to 62 additional candidates may be opened, and at most 12 non-Golden records may be published, keeping Golden plus Pilot at no more than 30.

Selection is independent of labels, age, outcomes, and model behavior. Existing audited evidence may nominate candidates for: no valid HRV window, missing algorithm annotation, Race Unavailable, incomplete probability channels or source lengths, I0006 non-9 out-of-range probabilities, and valid/invalid-stage probability contrasts. Every nominated condition is revalidated in the source NPZ. A category absent within the fixed budget is reported as `NOT COVERED BY REAL PILOT` and remains covered synthetically.

Each Pilot record must demonstrate the expected feature-local missingness, 807/827 invariants, atomic write/read, exact replay, source SHA stability, and `SKIP_VERIFIED` Resume. Missingness never excludes the record or contaminates unrelated families.

## Probability QC

The six original I0006 Golden sources are checked first. QC separates value 9, non-9 values below zero, non-9 values above one, values in `(1,9)`, values above 9, stage-valid versus stage-invalid placement, channels, anomalous rows, record concentration, and source-length agreement. Values are not clipped, converted, or used to recompute `x_static[87]`. The frozen strict entropy rule remains authoritative. Zero strict entropy coverage is an alert, not an automatic calculation failure; unexplained metadata contradiction yields `NEEDS_REVIEW`.

## Inventory, benchmark, and gate

Inventory enumerates names and filesystem metadata only for all 6,600 files. It verifies I0002=319, I0006=1,142, S0001=5,139, `.npz` suffixes, unique relative paths and record-key candidates, regular non-symlink files, confinement under the input root, and separation from output. The private inventory is not committed.

Performance compares current code with read-only source extracted from commit `ed5707531bfc0c3cc404b33baafae9627106c4b0` on at least one typical record per site, after warmup and repeated runs in the same environment. It records median runtime, full `X_seq` float64-conversion count, and reproducible peak-memory observations. The old source cannot be replaced by the optimized function. Stable material runtime or memory regression yields `NEEDS_REVIEW`.

Submission requires Golden 18/18 exact agreement, safe no-clobber and Resume behavior, bounded Pilot and candidate reads, passing inventory, no unexplained computation conflict, all old/new tests passing, no material performance regression, unchanged sources/Golden/Frozen files, and exact Git scope. G2-B1, preprocessing, normalization, and training remain outside this gate.
