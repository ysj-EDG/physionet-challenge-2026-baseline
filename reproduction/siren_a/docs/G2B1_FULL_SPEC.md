# G2-B1 Full Night-level Cache Specification

## Scope

G2-B1 materializes the frozen SIREN-inspired Branch A feature contract for all
6,600 `timegrid_v2.1.0` / `validity_v0.2` NPZ records. Every record produces
807 unique features and 827 five-view positions. This gate does not filter
patients, alter feature definitions, impute, normalize, rank by site, read
labels, or train models.

## Frozen inputs

- NPZ root: the audited 6,600-record v2.1 cache only.
- Sites: I0002=319, I0006=1142, S0001=5139.
- Frozen Manifest SHA256:
  `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a`.
- Frozen Spec SHA256:
  `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a`.
- Frozen computation API: `compute_night_features(record, manifest)`.
- G2-B0-R private CAISR audit: exactly 70 annotation-missing records, six
  readable records with fewer than 20 valid stages, and 76 records with fewer
  than 20 aligned valid stages when missing sources are included.

No CAISR EDF is reopened in G2-B1. The private audit's record mapping is used
only to cross-check frozen NPZ metadata and output semantics.

## G2-B1 cache contract

The schema identifier is `g2b1_full_v1`. Each cache contains:

- 807 `unique_feature_ids`, values (`float64`), masks (`bool`), and support
  counts (`int64`).
- V1=113, V2=129, V3=349, V4=113, V5=123 values/masks/support arrays.
- Scalar provenance: source NPZ SHA256, frozen Manifest and Spec SHA256,
  hashes of `g2a_features.py`, `g2a_io.py`, `g2b0_batch_safety.py`,
  `g2b1_cache_contract.py`, and `g2b1_full_run.py`, Python and NumPy versions,
  and cache schema version.

Historical Pilot-only provenance is forbidden. A G2-B0 cache cannot validate
as G2-B1 merely by changing a schema string.

All valid values are finite and all invalid values are NaN. Pooled families
(EEG, Spatial, HRV, EMG, Resp) require valid=>support>=5 and
invalid=>support<5. CAISR20 and Demo5 require valid=>support=1 and
invalid=>support=0. Unique-to-view values, masks, and support are exact.

## Inventory and anonymous naming

The input inventory is sorted by site and relative path, contains exactly
6,600 unique regular `.npz` files, and rejects symlinks, unexpected layouts,
duplicates, and source/output overlap. Stable cache names are ordinal and
anonymous (`cache_000000.npz` through `cache_006599.npz`). The private index
is the sole mapping between cache name, site, record key, and source path.

The run identity freezes input root, inventory fingerprint, expected counts,
Manifest/Spec hashes, code hashes, Python/NumPy versions, and schema version.
An existing output root may be resumed only when this identity is identical.

## Execution modes

- `--preflight-only`: validate Git-independent data prerequisites, inventory,
  fingerprints, CAISR audit, references, capacity, and output compatibility;
  never create record caches.
- `--build`: require a new output root, publish run identity/inventory/index,
  then compute all records sequentially.
- `--resume`: require an identity-compatible output root; validate and skip
  complete caches, compute missing caches, and stop on any invalid cache.
- `--verify-only`: require a complete compatible output root and independently
  validate all caches and source hashes without feature recomputation.

No mode is implicit. Full writing requires explicit `--build` or `--resume`.

## Publication and recovery

Each record is loaded once, hashed, pooled once, assembled into Views, and
published through the existing same-filesystem temporary-file, fsync,
round-trip, hard-link no-clobber mechanism. A published cache is immediately
validated. Checkpoint files are immutable, aggregate-only progress markers
written every 100 completed records. Temporary files never count as complete.

Resume actions are `COMPUTE`, `SKIP_VERIFIED`, or `STOP_INVALID`. No invalid
cache is deleted, overwritten, or silently skipped. A record failure stops new
calculation while preserving already verified outputs.

## CAISR closure

All 6,600 records are retained. For the 70 audit-mapped missing annotations,
EEG648, Spatial36, and CAISR20 must all be invalid; HRV, EMG, Resp, and Demo
remain governed independently. For the six readable short-stage records,
every stage-conditioned column follows its own support>=5 rule; no record-wide
20-epoch rule is introduced.

## Full verification and QC

Verify-only checks 6,600 caches, 5,326,200 unique positions, and 5,458,200 View
positions; all dimensions, dtypes, finite/NaN rules, support semantics,
provenance, source hashes, and mappings must pass. It emits private aggregate
run/verification summaries and a public feature-only CSV with one row per
site/Overall and unique feature ID. No record identity or patient value enters
Git.

QC reports family and per-feature valid/missing totals, rates, record-level
valid-fraction distributions, support percentiles, and lawful-zero counts.
`caisr_sleep_mean_stage_entropy` coverage is reported without changing or
dropping the feature; zero coverage triggers `ENTROPY_COVERAGE_ALERT`.

## Gate decision

Submission requires 18/18 Golden and 2/2 accepted Pilot exact regression,
6,600 successful builds, 6,600 independent validations, CAISR 70/6/76 closure,
unchanged historical caches and frozen files, complete tests, and clean Git
scope. G3 and all model/preprocessing work remain prohibited.
