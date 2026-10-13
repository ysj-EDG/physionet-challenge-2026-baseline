# G2-A Independent Night-level Smoke Specification

## Scope

G2-A implements one-record deterministic pooling from the audited `timegrid_v2.1.0` / `validity_v0.2` NPZ schema into 807 unique features and the frozen 827 five-view positions. It runs only a bounded, deterministic smoke of at most 30 records. It does not expose a full-cohort mode, read labels, fit transforms, drop records, regenerate upstream features, or modify source NPZ files.

## Inputs and locked sources

The only data root is `/database/home/gaohaojie/workspace/python-example-2026/validity_v0_2_full_6600/workspace/result_bundle/npz`. The loader requires X_seq `(T,483)`, X_ecg `(W,12)`, x_static `(196,)`, v2.1 versions, and every validity field named by the frozen contract. NPZ access always uses `allow_pickle=False`; JSON uses `json.loads` only. The frozen Manifest SHA256 is `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a`, and the frozen Spec SHA256 is `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a`.

## Interfaces and outputs

`g2a_io.load_manifest()` validates SHA, 827 rows, 807 IDs, view dimensions, positions, and required columns. `g2a_io.load_npz_record()` returns only approved feature and validity fields and never reads `y`. `g2a_features.compute_night_features(record, manifest)` returns unique values/masks/support counts plus five same-shaped view arrays. Float64 is used throughout; invalid values are NaN with false masks, valid values are finite, and supports are nonnegative.

The unique order is first appearance of each `unique_feature_id` in Manifest order. Views are reconstructed exclusively from `(view_id, view_position, unique_feature_id)`. Static supports are 1/0; pooled supports are their actual valid observation counts. Patients are never excluded.

## Frozen calculations

EEG648, Spatial36, CAISR20, HRV22, EMG48, Resp28, and Demo5 exactly follow G1-B-R. Mean/IQR share masks, require five observations, and IQR uses NumPy linear quantiles. EEG leads pool independently. Spatial features use same-epoch intersections. CAISR validity is metadata-driven without an all-zero block sentinel. Entropy retains x_static[87] only under the complete posterior-source rule. Demo5 is computed once and repeated by ID; age is private evaluation metadata only.

## Sampling and private cache

Seed `20261013` selects per site sorted first/middle/last plus three unique RNG choices, totaling 18 base records. Optional bounded probing may add no more than 12 records, inspect no more than 80 NPZ contents, and never exceed 30 outputs. The private root is `/database/home/gaohaojie/workspace/python-example-2026-siren-a-g2a-private/g2a_smoke_v1`; an existing target is rejected. Anonymous cache names and a private sidecar hold identity mapping.

Caches include unique and view arrays, source/cache fingerprints, versions, runtime versions, seed, private age, and support summaries. Writes use a same-directory temporary file, fsync, validation, and atomic replace. Existing final files are never overwritten. Replays recompute from source and compare values with NaN positions, masks, counts, and source SHA.

## QC, privacy, and determinism

QC is aggregate by site and family. Posterior QC counts values equal to 9, out-of-range/nonfinite values, finite/in-range/zero-sum rows, stage-valid placement, channel occurrence, source-length mismatches, and strict entropy coverage. Zero entropy coverage triggers `ENTROPY_COVERAGE_ALERT`, not record deletion. Published artifacts contain no record IDs, paths, age, labels, patient-level vectors, or recoverable ordering.

## Gate and recovery

Tests must cover schema rejection, formulas, 4/5 boundaries, static missingness, mapping, cache round-trip, atomic no-overwrite, bounded selection, privacy, and absence of full modes. G1-A and G1-B tests must remain green. Any schema conflict, nondeterminism, invalid output invariant, source mutation, target collision, or need to change the frozen contract stops the gate. A failed atomic write leaves only a private temporary artifact and never replaces a final cache.
