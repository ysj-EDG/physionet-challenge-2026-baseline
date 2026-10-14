# G3-A-V Independent Verification Specification

## Scope and authority

G3-A-V is a read-only verification gate for the committed G3-A artifacts at parent commit `d7cff5549f15eddd113aea84830f56d873c1bbb9`. It independently reconstructs identity, LOSO indexes, reference distributions, average-tie percentiles, Rank256 values, five views, and audit aggregates from the frozen G2-B1 night cache. It never calls G3-A ranking, quantization, view, audit, preparation, or validation functions as mathematical authority.

The verifier may read only the frozen G2-B1 cache, G3-A artifacts, Frozen Manifest/Spec, and public G2/G3 aggregate CSV files. It does not read source physiological NPZ files, EDFs, age, labels, diagnoses, follow-up data, or outcomes. It never modifies any G1/G2/G3-A artifact and never trains a model.

## Frozen inputs

- G2-B1 root: `night_cache_v1`, schema `g2b1_full_v1`, 6,600 records.
- Sites: I0002=319, I0006=1,142, S0001=5,139.
- G3-A root: `rank256_v1`, expected cache fingerprint `cf88b4e87f8b524258c685db696c278198dae1e6a10bd39fb63e893b2df031df`.
- Manifest SHA256: `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a`.
- Spec SHA256: `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a`.
- Feature dimensions: 807 unique and 827 view positions; V1/V2/V3/V4/V5 = 113/129/349/113/123.

## Source identity

The verifier loads `run_identity.json`, `inventory.json`, and `private_index.json`. For every position i, inventory and index must agree exactly on anonymous cache name, relative source path, and site. Cache names must be `cache_%06d.npz`; cache names and relative paths must be unique; site counts and order must close over all 6,600 records.

The inventory fingerprint is independently rebuilt from canonical JSON. Every anonymous cache must be a regular non-symlink file. Its SHA256 participates in the independently rebuilt G3-A source fingerprint:

1. canonical G2-B1 run identity JSON;
2. frozen inventory fingerprint;
3. for each ordered cache, cache name then cache-file SHA256.

Each cache is opened with `allow_pickle=False`. Schema, Manifest/Spec provenance, code provenance, 807 feature IDs/order, float64 values, bool source mask, finite-valid/NaN-invalid semantics, and all five view mappings are checked. The per-cache source-NPZ SHA is checked for a valid SHA256 representation but is not re-derived because source physiological NPZ access is outside this gate.

The reconstructed source fingerprint must match G3-A run identity, data manifest, and expected fingerprint. G3-A's recorded hashes for `data.py`, `split.py`, `normalize.py`, `views.py`, `audit.py`, and `run_prep.py` must match current files. New verifier hashes are separate and never alter old identity.

## LOSO indexes

The verifier independently constructs ordered indexes from the validated site vector:

| Holdout | Train | Test | Unused |
|---|---|---|---|
| I0002 | S0001 | I0002 | I0006 |
| I0006 | S0001 | I0006 | I0002 |
| S0001 | I0006 | S0001 | I0002 |

Saved train/test/unused arrays must be int64, in range, unique, pairwise disjoint, cover exactly 0..6599, preserve frozen source order, and equal the independently rebuilt arrays element by element. Split manifest, unit metadata, and actual arrays must agree.

## Independent reference and percentile mathematics

For each feature, `build_reference(values, mask)` uses `np.unique(..., return_counts=True)` to store sorted unique float64 values, multiplicities, valid count, cumulative counts, and a fingerprint compatible with the frozen serialized duplicate-value reference. It does not call old normalization functions.

For query x in a reference of n valid observations:

`u(x) = (L(x) + 0.5 E(x)) / n`

where L is the count strictly below x and E is the count equal to x. NaNs are excluded by the source mask; valid infinities are rejected. Ties, constants, singletons, legitimate zeroes, binary features, and out-of-range values follow the frozen rule. If n=0, rank validity is false and output is NaN without changing source validity.

Transductive train/test references use their own site cohorts. Inductive train/test both use only the selected training site. Saved reference arrays, offsets, counts, and fingerprints are checked exactly. Saved source/rank masks must match reconstructed masks exactly. Percentiles are required to match exactly; any nonzero float difference is recorded and fails the gate.

## Rank256

For each independently reconstructed valid percentile u:

`k = floor(255*u + 0.5)`

`q = k/255`

Levels are int16 in 0..255, missing levels are -1, outputs are float64, and missing outputs are NaN. Saved levels and q are compared exactly. Grid membership, half-up behavior, monotonicity, maximum error `1/510 + 1e-14`, and no more than 256 occupied levels are checked independently. A self-consistent but wrong adjacent level fails.

## Five views

The verifier parses the Frozen Manifest directly with the standard CSV module. It validates 827 rows, 807 unique IDs, continuous zero-based view positions, expected view sizes, and unique-feature ordering. It independently maps Rank256 matrices into five views and compares every value and rank mask exactly. Repeated Demo5 positions must map to the same unique column.

## Audit closure

Each private `rank_audit.csv` must contain 807 train and 807 test rows with unique keys and independently recomputed counts and floating statistics. Public `g3a_rank_summary.csv` must equal the ordered concatenation of all six private audits and contain 9,684 rows. The G2-B1 aggregate must contain 807 rows for each of I0002, I0006, S0001, and Overall, and close exactly against source masks.

The verifier recomputes valid counts, reference counts, unique percentile/grid counts, occupied bins, ties/collapsed levels, quantization errors, endpoints, and inductive overflow. Float fields use exact equality after CSV parsing except NaN-to-NaN.

## Integrity and recovery

Each unit must contain only the seven payload files plus `complete.json`; no symlinks or object arrays are allowed. Completion schema, file set, and SHA256 values are checked before mathematical verification. No `.building` directory may exist. Completion hashes are necessary but never sufficient: fault-injection tests rewrite payload and completion hashes and must still be rejected mathematically.

Private verification output is written only below the approved G3-A-V private root. Output uses same-directory temporary files, fsync, hard-link no-clobber publication, read-back validation, and an atomically last completion marker. Existing matching completed output is verified; mismatched or partial output is rejected rather than overwritten.

## Execution and gate

CLI modes are `--preflight-only`, `--smoke`, `--full`, and `--verify-output`. Smoke verifies both I0002 protocols and every feature position. Full verifies all six units. Verify-output rechecks private verification files and hashes without recomputing G3-A.

PASS requires all source identity, indexes, references, percentiles, Rank256 matrices, views, audits, fault-injection tests, and historical regressions to pass with zero unexplained mismatch. Any data-semantic mismatch returns NEEDS_REVIEW; Git or required-input failure returns BLOCKED. G3-A artifacts are never repaired or regenerated.
