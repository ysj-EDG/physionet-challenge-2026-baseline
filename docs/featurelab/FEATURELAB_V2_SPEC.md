# FeatureLab v2 Design Specification

Status: proposed for implementation review

Source commit: `7b3e5f8ac63096199969c75eb99a646d1be2cae8`

Target branch: `research/featurelab-extractors-v2`

## 1. Purpose and boundaries

FeatureLab v2 is an isolated, NPZ-derived extension layer for the frozen SIREN Branch A feature pipeline. Its smallest recomputation unit is one record, one feature family, and one recipe version. It must not modify the frozen `timegrid_v2.1.0` source NPZ files, the 807-feature night cache, the five production views, model code, training artifacts, or LOSO results.

Batch 1 implements six independently cached candidate families totaling 637 columns. Batch 2 produces only a preprocessing-difference audit for 553 proposed columns. No Batch 2 raw-signal algorithm may be implemented until its unresolved preprocessing semantics have been reviewed.

The implementation must never read disease labels, `y`, time-to-event data, diagnoses, model scores, predictions, or site-level outcome summaries. Record selection for smoke testing is based only on source schema and modality/stage support.

## 2. Verified source contract

The only Batch 1 source root is:

`/database/home/gaohaojie/workspace/python-example-2026/validity_v0_2_full_6600/workspace/result_bundle/npz`

Every source is opened with `numpy.load(..., allow_pickle=False)`. The loader requires:

- `extraction_version == timegrid_v2.1.0`.
- `validity_schema_version == validity_v0.2`.
- `X_seq.shape == (T, 483)`.
- `X_ecg.shape[1:] == (12,)` and `x_static.shape == (196,)`.
- All family-specific stage, time, channel, and validity arrays with their exact verified shapes.
- `epoch_start_sec` and `epoch_duration_sec` consistent with `T`; adjacency additionally requires a 30-second difference in `epoch_start_sec`.

The loader exposes an allowlisted `SourceRecord` and deliberately omits `y`. The source identity includes the canonical resolved source path, record key, file size, file SHA256, source schemas, required field names, shapes, and dtypes. A family declares its required keys so unrelated source fields do not enter its recipe.

The record key is deterministic, cross-site unique, and safe for use as one filename component. The loader first verifies that NPZ `site_id` and `record_id` agree with the source's `<site_id>/<record_id>.npz` relative location. It then canonicalizes the exact tuple `{"namespace":"timegrid_v2.1.0","site_id":<site_id>,"record_id":<record_id>}` as sorted compact UTF-8 JSON and sets `record_key` to `rec_` followed by the full lowercase SHA256 hex digest. Raw identifiers never appear in the cache filename. An inventory and the cache validator both maintain the canonical identity tuple behind each key; the same key associated with a different tuple or different source identity is `STOP_INVALID`, never an overwrite or automatic rename. Duplicate paths, duplicate canonical tuples with different files, unsafe/noncanonical keys, and path-versus-NPZ identity disagreement are hard errors before calculation.

The frozen column slices remain unchanged:

- EEG54: `X_seq[:, 0:54]`.
- Coherence360: `X_seq[:, 54:414]`.
- BSR18: `X_seq[:, 414:432]`.
- EMG24: `X_seq[:, 432:456]`.
- Resp14: `X_seq[:, 456:470]`.
- CAISR time series: `X_seq[:, 470:483]`.
- HRV11: `X_ecg[:, 0:11]`.
- Demographic10 plus CAISR186: `x_static`.

## 3. Extension architecture

The implementation lives under `per_epoch_features/extensions/` and remains independent of `TimegridV2Extractor.extract_all()`.

- `contracts.py` defines immutable feature definitions, source identity, recipe identity, `FeatureResult`, and validation errors.
- `sources.py` validates and loads an existing NPZ once per record without reading labels.
- `registry.py` contains an explicit ordered registry of family ID, version, default configuration, extractor callable, required source fields, and numerical code dependencies.
- `recipe.py` canonicalizes recipe material and derives a SHA256 recipe ID.
- `cache.py` validates, reads, and atomically publishes one family result.
- `runner.py` evaluates requested families only, sharing the in-memory source record while keeping outputs independent.
- `common.py` contains stage masks, linear quantiles, time-grid adjacency, stable serialization, and repeated aggregation helpers.
- Six family modules contain pure numerical functions: `eeg_tail.py`, `eeg_temporal.py`, `eeg_coherence_stage.py`, `caisr_stage.py`, `emg_stage.py`, and `resp_stage.py`.

The public family interface is:

```python
extract_family(source: SourceRecord, config: Mapping[str, object]) -> FeatureResult
```

`FeatureResult` contains equal-length, one-dimensional arrays or tuples for `feature_ids`, `values`, `valid_mask`, `support_count`, `missing_reason`, and `feature_units`, plus scalar `family_id`, `family_version`, `source_identity`, and `processing_parameters`. Validation requires globally unique feature IDs, finite values exactly where valid, `NaN` exactly where invalid, nonnegative integer support, and a nonempty reason for each invalid value. Valid numerical zero is retained.

Feature ordering is constructed from explicit tuples, never unordered mappings.

## 4. Recipe and cache contract

The private default cache root is:

`/database/home/gaohaojie/workspace/feature-cache-extension`

The path layout is:

`<root>/<family_id>/<recipe_id>/<record_key>.npz`

Each recipe ID is the SHA256 of canonical JSON containing only inputs that can change that family's numerical result:

- Family ID and algorithm version.
- Formula identifiers and all numerical parameters.
- Required source fields, column definitions, and validity rules.
- Relevant preprocessing semantics inherited from the source.
- SHA256 fingerprints of the family extractor and explicitly declared transitive numerical code units. At minimum this covers the exact NPZ source-adapter callable that materializes family inputs, the stage/time-grid validity callables used by the family, shared aggregation callables, family-specific numerical helpers, and explicitly referenced numerical constants.
- Versions of NumPy and any third-party numerical package used by that family.

The declarative recipe material also embeds the source-adapter contract version, exact required source fields and columns, stage-code mapping, stage scopes, adjacency definition, validity expressions, minimum supports, missingness rules, and normalized processing parameters. Each dependency is named as a module-qualified symbol or explicit constant. Its function/class definition is canonicalized from the Python AST without comments, docstrings, or source-location metadata before hashing; every numerical helper it calls must be separately declared. The registry rejects an unresolved symbol, an undeclared numerical helper detected by its dependency audit, or a missing constant fingerprint. This ensures changes to source adaptation, stage validity, or shared calculations invalidate every affected family without making unrelated functions in the same file part of the recipe.

The repository Git commit is recorded as provenance but is not recipe material. Cache serialization, atomic-publication code, CLI/reporting code, documentation, tests, other families, training code, and unrelated repository changes cannot invalidate a family recipe unless they are explicitly declared as numerical dependencies. Tests must prove both directions: changing a declared numerical dependency changes the recipe, while changing an unrelated file does not.

Cache payloads use non-object NumPy arrays and canonical JSON strings. They include cache schema, completion status, family and recipe identities, ordered feature IDs, all `FeatureResult` fields, source identity, processing parameters, code provenance, runtime versions, and a logical payload SHA256. The payload digest covers typed array shapes/dtypes/bytes and canonical metadata, excluding only the digest field itself.

Resume decisions have exactly three outcomes:

- `BUILD_NEW`: the final path is absent.
- `SKIP_VERIFIED`: schema, completion marker, family, recipe, feature order, source identity, invariants, and payload hash all match.
- `STOP_INVALID`: a final path exists but any check fails.

Writers create a same-directory temporary file, flush and fsync it, reload and validate it, and publish with a no-clobber atomic operation. An existing final file is never replaced, removed, or repaired automatically. Temporary files never count as complete caches. A runner requested for one missing family computes only that family.

No raw or preprocessed overnight waveform is cached in Batch 1.

## 5. Batch 1 feature definitions

### 5.1 EEG tail quantiles: 288 columns

For each lead in `(F3, F4, C3, C4, O1, O2)`, band in `(Delta, Theta, Alpha, Sigma)`, stage scope in `(Wake, REM, N2, N3, NREM, All-stage)`, and quantile in `(Q88, Q95)`, calculate relative power as:

`logPbandPbeta + logPbeta - logP0`

The per-lead source offsets are band ratios Delta=4, Theta=3, Alpha=2, Sigma=1, beta=7, and P0=6 within each nine-column EEG block. Validity requires valid stage membership, lead availability, `eeg_clean_subsegment_count[:, lead] > 0`, and finite source values. Each output requires at least 20 observations and uses `numpy.quantile(..., method="linear")`.

### 5.2 EEG overnight temporal organization: 96 columns

For the same 24 lead-band series, use only valid sleep epochs with stages N1, N2, N3, or REM and the EEG quality rule above. At least 20 observations are required to establish the reference distribution.

- `range80 = Q90 - Q10`, using linear quantiles.
- `drift` splits the complete raw record positions into three contiguous index ranges, then selects valid sleep epochs inside the first and last thirds. Each side requires 20 observations. It is last-third mean minus first-third mean.
- `tail_hour_concentration` sets a per-record reference threshold to sleep Q88 and uses strict `value > Q88`. It scans every 120-position window on the original 30-second grid. A window participates only with at least 96 valid sleep observations; the output is the maximum high-value count divided by that window's valid observation count.
- `tail_entry_rate_per_hour` counts false-to-true transitions only across adjacent raw positions where both observations are valid sleep epochs and `epoch_start_sec` differs by exactly 30 seconds. Wake, invalid data, and time gaps break adjacency. The denominator is the number of legal pairs times 30 seconds, converted to hours. A constant series therefore has zero entries.

The support field records the statistic-specific observation or legal-pair count.

### 5.3 Stage-specific EEG coherence: 90 columns

Use pairs F3-F4, C3-C4, and O1-O2, whose verified pair positions are 0, 9, and 14 in the 15-pair block. For Delta, Theta, Alpha, Sigma, and Beta, use the first five mean-coherence columns within each 24-column pair block. Aggregate Mean and IQR for REM, N2, and N3.

Validity requires both physical leads available, `eeg_common_clean_subsegment_count[t] > 0`, valid target stage, and a finite stored value. The common-clean count is the exact support used by the frozen extractor to calculate all pair spectra, so finite upstream zero alone is not treated as validity. Each aggregate requires five epochs.

### 5.4 CAISR stage dynamics: 35 columns

Stage order is fixed as `(N3=1, N2=2, N1=3, REM=4, Wake=5)`.

The first 25 outputs are a source-major 5 by 5 transition matrix. A transition is counted only when the two original positions are consecutive, their starts differ by 30 seconds, and both stages are valid. Each source row is normalized by that source stage's legal outgoing transitions. A row with no outgoing support is five invalid `NaN` values.

The final ten outputs are P50 and P95 bout durations for each stage. They reuse the existing CAISR186 values in `x_static`: median bout indices 44 through 48 and P95 indices 69 through 73, in upstream `(Wake, N1, N2, N3, REM)` order. No CAISR EDF is read. Validity is restored by requiring `annotation_status == available` and at least one corresponding bout reconstructed from the original valid stage grid. Invalid or unknown epochs terminate a bout. Support is the number of stage bouts.

### 5.5 Stage-specific EMG: 72 columns

For Chin, left leg, and right leg, select within each eight-column channel block offsets 0 (`log_rms_norm`), 3 (`tonic_fraction`), 4 (`burst_rate_per_min`), and 6 (`phasic_mini_epoch_fraction`). Aggregate Mean and IQR for Wake, REM, and NREM.

Validity requires target-stage validity, channel availability, `emg_preprocessing_success`, `emg_epoch_success[t, channel]`, and finite source value. Each aggregate requires five epochs. Existing thresholds and whole-night baseline are not recomputed.

### 5.6 Stage-specific Resp: 56 columns

For every existing Resp14 column, aggregate Mean and IQR for REM and NREM. Validity requires target-stage validity, `resp_feature_valid[t, column]`, and a finite value. Channel-specific absence is already encoded in the per-feature mask. Each aggregate requires five epochs.

## 6. Manifest and future view mapping

`audit/featurelab_batch1_manifest.csv` has exactly one row per candidate column and records all fields required by the task: ID, family, modality, proposed view, source field and column, stage/channel scope, meaning, formula, parameters, unit, support, validity, recipe version, reference, and implementation status.

Statuses are limited to `COMPUTED_FROM_EXISTING_NPZ`, `REUSED_FROM_EXISTING_STATIC`, `NEEDS_NEW_RAW_SIGNAL`, and `REQUIRES_INPUT_REVIEW`. Batch 1 rows may be marked implemented only after their source semantics and tests pass. The manifest is checked against all 807 frozen unique IDs; exact duplicates are reported and excluded from the claimed new count.

The proposed, non-training view placement is:

- V1: 48 Wake tail columns.
- V2: 48 REM tail columns.
- V3: 144 N2, N3, and NREM tail columns.
- V4: 48 All-stage tail plus 96 temporal columns.
- V5: 90 coherence, 35 CAISR, 72 EMG, and 56 Resp columns.

No new five-view training manifest or model input matrix is generated.

## 7. Batch 2 preprocessing audit only

`docs/featurelab/BATCH2_PREPROCESSING_REVIEW.md` compares current local behavior with each candidate algorithm for EEG spectral 288, EEG complexity 192, ECG/fragmentation 20, CAISR event intervals 9, respiration 24, and SpO2 20.

For every family it records raw channels, current loader and preprocessing, proposed preprocessing, sample rate, filters, windows, validity, thresholds, reuse decision, necessary EDF access, reusable intermediate cache, cost, overlap, unresolved parameters, expected dimensions, and `READY_TO_IMPLEMENT` or `REQUIRES_REVIEW`.

The ID565 repository is an algorithmic reference only. In particular, its temporal example compresses valid sleep epochs before drift and one-hour calculations, while FeatureLab retains the original grid. Its complexity and coherence examples concatenate separated stage epochs and use preprocessing/window choices that are not automatically equivalent to the local pipeline. Those differences must be explicit; no complete external extractor, training logic, labels, TabFM, CORAL, or prediction code is copied.

## 8. Testing and smoke protocol

Tests use Python's standard `unittest` because the specified environment does not contain `pytest` and dependencies must not be installed automatically.

Synthetic tests cover:

1. Q88/Q95 equality with NumPy linear quantiles.
2. Drift over uncompressed raw positions.
3. No transitions across gaps.
4. Exactly 120 raw positions for a one-hour window.
5. Missing positions are not compressed.
6. No high-state entries for a constant series.
7. Low support yields `NaN`, false validity, and a reason.
8. Valid physiological zero remains valid.
9. Transition-row normalization and missing rows.
10. CAISR code/order and bout-static mapping, including a synthetic aligned stage grid with invalid gaps whose directly calculated per-stage bout P50/P95 values must equal the values read through the exact `x_static` static indices and stage reorder.
11. Independent EMG and Resp masks.
12. Source NPZ SHA256 and stat metadata do not change.

Cache tests use isolated temporary roots:

- Repeating the same configuration yields six `SKIP_VERIFIED` decisions and no extractor calls.
- Changing one EEG-tail parameter creates only a new EEG-tail recipe; every other recipe and cache remains reusable.
- Removing one family result from a disposable test copy rebuilds only that family.

The real smoke selects one record from the verified NPZ inventory using only schema, stage support, and modality availability. It reads no label and no EDF. Selection first uses existing aggregate/audit evidence where possible and otherwise inspects only a bounded number of NPZ metadata records. The source file is SHA256-snapshotted before and after. The smoke performs one build, one cache readback, and one resume.

All real single-patient feature values, Q88 thresholds, Q95 values, drift values, concentration values, entry rates, transition probabilities, stage statistics, record identity mappings, and per-column support details remain only in the private cache/report root and are never committed or copied into public reports. Tracked reports may contain aggregate dimension/valid/missing counts, elapsed time, byte counts, resume decisions, source immutability results, and explicitly labeled synthetic numerical examples only. “Anonymous” real-patient numerical examples are not considered public-safe.

## 9. Failure behavior and deliverables

Malformed source schemas and invalid existing cache files are hard stops for the affected record/family. Expected data insufficiency is per-column missingness, not a software error or record deletion. A family that cannot recover validity semantics is reported as `REQUIRES_INPUT_REVIEW`; other independent families continue.

Tracked deliverables are extension source, standard-library tests, this specification, cache schema documentation, Batch 1 aggregate report, Batch 2 review, and the candidate manifest. EDF, NPZ, patient values, demographics, model artifacts, predictions, identity maps, and temporary files are never committed.

After verification, only explicit tracked files are staged. The branch is committed and pushed, and local/remote SHA equality is checked. No code is copied or merged back to `research/siren-a-stagepool-v1`.
