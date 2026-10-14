# G3-A LOSO Rank256 Preprocessing Specification

## Scope and frozen inputs

G3-A transforms, but never re-extracts, the 6,600 by 807 frozen G2-B1 night-level feature matrix. The only input is the verified `g2b1_full_v1` private cache with inventory fingerprint `1f914cdfe88857a5bc2b5a9a9773ab0c7e87fa9ca81354015bf7393d670732f3`. The Frozen Manifest and Spec SHA256 values are respectively `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a` and `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a`.

G3-A does not read age, labels, diagnoses, follow-up outcomes, EDFs, or source physiological NPZ files. It does not alter G2 masks or support counts, drop records, fill missing values, train models, or modify the 807-feature contract.

## Data interface

`load_features(cache_root, manifest)` reads the private anonymous index and all 6,600 G2-B1 caches with `allow_pickle=False`. It returns float64 values `(6600,807)`, bool source-valid masks `(6600,807)`, the frozen feature IDs, anonymous cache names, site IDs, and a deterministic fingerprint covering the G2 run identity, inventory and every cache file. Each cache must identify schema `g2b1_full_v1`, the frozen Manifest/Spec, and the exact 807 IDs. Valid values are finite; invalid values are NaN.

The loaded valid/missing counts for each site and feature must equal the committed G2-B1 aggregate CSV exactly.

## LOSO contract

The three holdouts are I0002, I0006 and S0001. The selected training site is the largest of the two non-holdout sites; the remaining site is recorded as unused.

| Holdout | Train | Train N | Test N | Unused |
|---|---|---:|---:|---|
| I0002 | S0001 | 5139 | 319 | I0006 |
| I0006 | S0001 | 5139 | 1142 | I0002 |
| S0001 | I0006 | 1142 | 5139 | I0002 |

Indexes preserve G2-B1 source order, are unique, and never overlap within a fold. No second training site is added.

## Percentile transform

For a value `x` and the valid reference multiset of size `n`, let `L(x)` be the count strictly below `x` and `E(x)` the count equal to `x`. The percentile is

`u(x) = (L(x) + 0.5 E(x)) / n`.

The formula applies to every feature, including binary and constant columns. Ties receive identical values; a constant or singleton reference maps its represented value to 0.5. NaNs and invalid-mask positions never enter a reference. Infinite valid inputs are rejected.

If a reference column has no valid values, rank-valid is false and output is NaN even when the source value is valid. Below-reference values map to zero and above-reference values map to one. Equality with an endpoint still uses the `L/E` formula.

Transductive mode ranks each selected train and holdout site against its own complete unlabeled site cohort. Inductive mode fits only the selected training site and applies that immutable reference to both train and holdout. Training percentiles and masks must be exactly equal across protocols. Holdout data never enter the inductive reference.

Source-valid and rank-valid masks are stored separately and are never conflated.

## Rank256 quantization

For valid percentiles in `[0,1]`:

`k = floor(255 u + 0.5)` and `q = k / 255`.

`k` is int16 in `[0,255]`; missing positions use the separate sentinel `-1`. `q` is float64 and missing positions are NaN. Half-up behavior maps 0.5 to `128/255`. Maximum error is `1/510` plus floating comparison tolerance. Quantization must be monotone, deterministic and cannot add dimensions or noise.

## Five views

All ranking and quantization occurs once in 807-dimensional unique-feature space. `get_views` then uses `(view_id, view_position, unique_feature_id)` from the Frozen Manifest to produce V1=113, V2=129, V3=349, V4=113 and V5=123. Repeated Demo5 positions map to the same unique values and masks; views are never ranked independently.

## Audit contract

Private raw audits retain per-site/per-feature valid, missing, unique, tie, zero and raw distribution statistics. Rank/quantization audits cover every fold, protocol, role and feature: reference size, percentile and grid cardinality, occupied bins, quantization errors, added ties, collapsed levels, endpoint counts, ranges, reference-unavailable state and inductive overflow counts.

Public output contains only feature-level aggregates. It excludes record identifiers, paths, ages, labels and low-sample raw value ranges. Programmatic checks enforce percentile range, finite/NaN rules, tie equality, monotonicity, order invariance, grid membership, error bound, source/rank mask separation and view consistency.

## Private artifacts and recovery

The only output root is the approved private G3-A directory. It contains immutable data/split manifests, private raw audit, and six fold/protocol units. Each unit stores row indexes, source/rank masks, train/test percentiles, int levels, float quantized matrices, mapped views, reference distributions, audit rows, metadata and an atomically last completion marker.

Unit construction occurs in a sibling `.building` directory and is atomically renamed only after complete write/read verification. Existing valid units are skipped. Existing invalid units or incomplete build directories stop execution; nothing is deleted or overwritten. Run identity freezes input fingerprint, Manifest/Spec, code SHA256 values, LOSO mapping, formula, 256 levels, Python and NumPy.

CLI modes are explicit: `--preflight-only`, `--smoke`, `--build`, and `--verify-only`. Smoke builds only both protocols for holdout I0002. Build safely completes all six units. Verify-only independently validates all artifacts. No training option exists.

## Gate boundary

G3-A may report protocol differences and missingness but may not choose a model, alter the transform, merge the unused site, evaluate outcomes, or begin G3-B/G4/G5.
