# FeatureLab v2 Family Cache Schema

Status: implemented for the Task 1–3 synthetic contract only. No real-record cache has been produced or validated in this phase.

## Scope and path

The final cache unit is exactly one source record, one feature family, and one Recipe:

`<cache_root>/<family_id>/<recipe_id>/<record_key>.npz`

`family_id` is a safe registry identifier, `recipe_id` is a lowercase 64-character SHA256 digest, and `record_key` has the form `rec_` plus a lowercase 64-character SHA256 digest. Requested-family count and execution order are not path or identity material.

The cache contains only final family features and provenance. It does not contain raw signals, preprocessed overnight waveforms, labels, TTE, predictions, scores, or model artifacts.

## NPZ keys

The schema version is `featurelab_family_cache_v1`. Every final file contains exactly these non-object arrays; missing or extra keys are invalid.

| Key | Shape | dtype | Meaning |
|---|---:|---|---|
| `cache_schema` | scalar | Unicode string | Cache schema version. |
| `completion_status` | scalar | Unicode string | Must be `COMPLETE`. |
| `family_id` | scalar | Unicode string | Feature-family identifier. |
| `family_version` | scalar | Unicode string | Family algorithm version. |
| `recipe_id` | scalar | Unicode string | SHA256 of `recipe_material_json`. |
| `recipe_material_json` | scalar | Unicode string | Canonical numerical Recipe material. |
| `recipe_provenance_json` | scalar | Unicode string | Canonical provenance, including Git commit; provenance is not Recipe material. |
| `source_identity_json` | scalar | Unicode string | Canonical request-independent record identity. |
| `source_provenance_json` | scalar | Unicode string | Resolved source path, mtime in nanoseconds, and mode; payload provenance only, never cache-match identity. |
| `source_identity_digest` | scalar | Unicode string | SHA256 of SourceIdentity material. |
| `family_input_identity_json` | scalar | Unicode string | Canonical family-specific input observation and contract identity. |
| `family_input_identity_digest` | scalar | Unicode string | SHA256 of FamilyInputIdentity material. |
| `processing_parameters_json` | scalar | Unicode string | Canonical resolved family configuration. |
| `feature_ids` | `(F,)` | Unicode string | Exact ordered family feature IDs. |
| `values` | `(F,)` | `float64` | Finite where valid and NaN where invalid. |
| `valid_mask` | `(F,)` | `bool` | Per-feature validity. |
| `support_count` | `(F,)` | `int64` | Nonnegative statistic-specific support. |
| `missing_reason` | `(F,)` | Unicode string | Nonempty for every invalid feature. |
| `feature_units` | `(F,)` | Unicode string | Ordered feature units. |
| `payload_sha256` | scalar | Unicode string | Typed logical digest of every other key. |

All JSON uses sorted keys, compact separators, UTF-8 semantics, and no object arrays. `recipe_material_json`, `source_identity_json`, and `family_input_identity_json` are internally checked against their respective digest fields. Resolved processing parameters must equal the Recipe's processing parameters.

## Logical payload digest

`payload_sha256` excludes only itself. For every remaining key in lexical key order, the digest receives four length-prefixed byte sequences:

1. UTF-8 key name.
2. NumPy `dtype.str` encoded as ASCII.
3. Canonical JSON array shape encoded as ASCII.
4. C-contiguous array bytes.

Each component is prefixed by its unsigned eight-byte big-endian length. Consequently, a change in key, dtype, shape, value bytes, ordered feature IDs, identity, Recipe, provenance, or completion marker changes the digest.

## Identity and validation

`SourceIdentity` is derived once from the source file and is independent of the requested family set, family order, or union of required fields. Its digest covers the canonical record namespace and identifiers, safe record key, source byte size and SHA256, extraction schema, and validity schema. The resolved source path and selected stat values are serialized separately in `source_provenance_json`; they remain payload provenance and are not cache-match identity.

`FamilyInputIdentity` separately covers the SourceIdentity digest, the selected family's `FamilyInputContract` digest, and the actual shapes/dtypes of that family's required fields. This separation guarantees that the same source, family, and Recipe can be reused when run alone or within a multi-family request.

Validation requires the exact key set, schema, completion marker, scalar/vector dtypes, equal feature-vector lengths, unique ordered feature IDs, finite-valid/NaN-invalid semantics, nonnegative support, invalid-value reasons, internal identity digests, Recipe digest, processing parameters, and logical payload digest.

## Resume decisions

- `BUILD_NEW`: the final path does not exist. Orphan temporary files do not count as final caches.
- `SKIP_VERIFIED`: the final file passes complete content validation and matches family, family version, Recipe, feature order, SourceIdentity, and FamilyInputIdentity.
- `STOP_INVALID`: a final path exists but is corrupt, incomplete, mismatched, wrongly ordered, or otherwise invalid.

An invalid final file is never overwritten, removed, renamed, or repaired automatically.

## Atomic publication and recovery

The writer creates one uniquely named temporary file in the final directory, writes and fsyncs it, reloads it with `allow_pickle=False`, and performs full validation. It then publishes with a same-filesystem hard-link operation, which fails if the final path already exists. After publication it validates the final file and fsyncs the directory.

The writer removes only its own temporary file. Pre-existing orphan temporary files are ignored and preserved. If another process wins publication, the losing runner validates the winner: a matching winner becomes `SKIP_VERIFIED` with reason `CONCURRENT_WINNER`; an invalid winner becomes family-local `STOP_INVALID`. No concurrent path is overwritten.

## Error isolation

A record-level source or canonical identity failure returns `STOP_RECORD` before any family extractor runs. A missing or malformed family-only input returns `REQUIRES_INPUT_REVIEW` for that family and creates no cache. A family computation or publication exception returns `FAILED` for that family and creates no completed result. An invalid existing cache returns family-local `STOP_INVALID`. Input-complete sibling families continue in all family-local cases.
