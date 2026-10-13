# G1-B Feature Contract Review Report

## Gate result

`G1B_R_STATUS = SUBMITTED_FOR_REVIEW`

The 827-row Branch A contract is frozen as a specification and submitted for human review. This status does not authorize G2.

## Evidence reviewed

- G1-B-R Git precondition: branch `research/siren-a-stagepool-v1`, starting HEAD `8eaaf610df70fe7c663b73e8ffc75a8f4a097527`, clean worktree, and matching GitHub remote ref.
- Existing G1-A audit evidence: 6,600 unique v2.1 records, site counts 319/1,142/5,139, extraction `timegrid_v2.1.0`, validity `validity_v0.2`, and dimensions X_seq=483, X_ecg=12, x_static=196.
- Source review: demographic helpers, EEG/EMG/Resp/HRV extractors, time-grid metadata construction, CAISR algorithmic extractor, typed feature rules, CAISR20 manifest, and probability/source-validity metadata.
- Fixed read-only verification sample: first/middle/last sorted NPZ from each site (nine records total), without recording identifiers or patient-level values. All sampled records matched versions, dimensions, and validity-array axes. The sample contained both valid numeric zeros and invalid zero fills, confirming that metadata—not value equality to zero—must govern validity.

No complete 6,600-record scan was rerun for G1-B; the approved prior aggregate audit is the cohort evidence. No EDF or diagnosis/follow-up field was read.

## G1-B-R source reproductions

- Case A used the actual algorithmic extractor and the `_stage_probabilities` method compiled directly from the v2.1 source. Five valid stage epochs with no posterior channels produced `stage_probability_channels_present=[False]*5`, no valid probability rows, and `x_static[87]=0`. One present channel also produced an entropy near zero after the upstream zero-fill path. Neither output is valid under the corrected contract.
- With all five finite deterministic posterior channels present, all probability rows were valid and the algorithm returned mathematically near-zero entropy. This remains a valid value.
- Case B reproduced the missing-annotation metadata injection. After `__extraction_mode__`, `__sampling_frequencies__`, and a length-five all-NaN `__aligned_stage_code__` were inserted, `algo_data` was nonempty and the 186-value algorithmic block was not all zero: two values were nonzero and recording time was 150 seconds. Therefore no all-zero block assumption is used.
- Importing the complete extractor through the package was blocked by an unrelated installed NeuroKit module using Python 3.10 union syntax under Python 3.9. The reproduction isolated the dependency-free algorithmic module and compiled the exact v2.1 probability method from its AST; no environment was changed.
- The fixed first/middle/last sample from each site (nine records total) had correct `(T,5)` axes, five present channels, and source lengths equal to `T`. All rows were finite and therefore marked `stage_probability_valid`, but every sampled record contained out-of-range probability values up to 9; the extractor clips these values. Consequently 0/9 sampled records satisfy the conservative entropy-source contract. This is aggregate-only evidence, not a full-cohort scan, and it does not alter any stored value.

## Frozen decisions

- Demo5 is Male, BMI, White, Black, Asian with explicit Other/Unknown, Others, Unavailable, and BMI-zero semantics.
- EEG648 uses six model stage groups, independent lead pooling, minimum five valid epochs, and linear-interpolation IQR.
- Spatial36 retains the approved 24 asymmetry and 12 anterior-posterior formulas with same-epoch intersections and minimum five epochs.
- CAISR20 retains the submitted indices/order and uses source-validity metadata. No all-zero algorithmic-block sentinel is part of formal validity; individual zeros are not missing by themselves.
- `caisr_sleep_mean_stage_entropy` additionally requires complete five-channel probability provenance, exact v2.1 matrix/vector axes, source lengths equal to the stored epoch grid, finite bounded values, and positive row sums. Missing or partial posterior inputs make only this feature missing; a fully sourced zero entropy remains valid.
- HRV/EMG/Resp use their own validity metadata; pooled Mean/IQR requires five valid observations. HRV observations are windows, while EMG/Resp observations are epochs.
- Missing output is raw NaN with `valid_mask=False`; no patient is excluded.

## Manifest integrity

| View | Rows |
|---|---:|
| V1 | 113 |
| V2 | 129 |
| V3 | 349 |
| V4 | 113 |
| V5 | 123 |
| Total | 827 |

Unique feature IDs: 807. The 25 Demo5 view rows reuse five IDs and identical semantics. All view positions are contiguous and zero-based. The original G1-A Draft Manifest remains unchanged.

## Missingness and lawful zero findings

The extractor writes zeros for several failed/unavailable inputs, but legitimate physiology can also equal zero. Therefore EEG, HRV, EMG, Resp, and CAISR values are never classified solely by `value == 0`. BMI zero remains an explicit demographic sentinel. CAISR validity is metadata-driven; even a complete all-zero CAISR block is only a consistency observation and is not a validity predicate.

## Verification boundary

Synthetic tests cover Demo5, stage sets, the five-observation boundary, NumPy linear IQR, spatial formulas and intersections, modality-specific masks, CAISR zero semantics, absent/partial/complete probability inputs, legitimate zero entropy, missing-annotation metadata injection, exclusions, and 827/807 view integrity. Existing G1-A tests are also required to pass before commit.

No unresolved critical source-index or missingness semantics remain for this contract. Fold-specific imputation, scaling, and site transformation are outside G1-B and remain for G3.
Partial probability input is not ambiguous in v2.1: the five-channel presence vector, row-validity vector, stored matrix, and serialized source lengths jointly expose it. Missing/partial channels, truncated ranges, nonfinite values, all-zero rows, or out-of-range values make `x_static[87]` missing under this conservative contract.

- Patient-level/raw data committed: **No**
- NIGHT_LEVEL_CACHE: **NOT GENERATED**
- MODEL_TRAINING: **NOT STARTED**
- G2: **NOT STARTED**
