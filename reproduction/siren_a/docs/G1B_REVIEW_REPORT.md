# G1-B Feature Contract Review Report

## Gate result

`G1B_STATUS = SUBMITTED_FOR_REVIEW`

The 827-row Branch A contract is frozen as a specification and submitted for human review. This status does not authorize G2.

## Evidence reviewed

- Git precondition: branch `research/siren-a-stagepool-v1`, starting HEAD `3860491be55857fe278d4f0a3a1914c913c54629`, clean worktree, and matching GitHub remote ref.
- Existing G1-A audit evidence: 6,600 unique v2.1 records, site counts 319/1,142/5,139, extraction `timegrid_v2.1.0`, validity `validity_v0.2`, and dimensions X_seq=483, X_ecg=12, x_static=196.
- Source review: demographic helpers, EEG/EMG/Resp/HRV extractors, time-grid metadata construction, CAISR algorithmic extractor, typed feature rules, CAISR20 manifest, and the existing whole-block CAISR sentinel handling.
- Fixed read-only verification sample: first/middle/last sorted NPZ from each site (nine records total), without recording identifiers or patient-level values. All sampled records matched versions, dimensions, and validity-array axes. The sample contained both valid numeric zeros and invalid zero fills, confirming that metadata—not value equality to zero—must govern validity.

No complete 6,600-record scan was rerun for G1-B; the approved prior aggregate audit is the cohort evidence. No EDF or diagnosis/follow-up field was read.

## Frozen decisions

- Demo5 is Male, BMI, White, Black, Asian with explicit Other/Unknown, Others, Unavailable, and BMI-zero semantics.
- EEG648 uses six model stage groups, independent lead pooling, minimum five valid epochs, and linear-interpolation IQR.
- Spatial36 retains the approved 24 asymmetry and 12 anterior-posterior formulas with same-epoch intersections and minimum five epochs.
- CAISR20 retains the submitted indices/order and uses source-validity metadata plus the documented all-zero algorithmic block sentinel. Individual zeros are not missing by themselves.
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

The extractor writes zeros for several failed/unavailable blocks, but legitimate physiology can also equal zero. Therefore EEG, HRV, EMG, Resp, and event-derived CAISR values are never classified solely by `value == 0`. BMI zero and the exact all-zero `x_static[10:196]` CAISR block are exceptions because their upstream implementations explicitly define those sentinels.

## Verification boundary

Synthetic tests cover Demo5, stage sets, the five-observation boundary, NumPy linear IQR, spatial formulas and intersections, modality-specific masks, CAISR zero semantics, exclusions, and 827/807 view integrity. Existing G1-A tests are also required to pass before commit.

No unresolved critical source-index or missingness semantics remain for this contract. Fold-specific imputation, scaling, and site transformation are outside G1-B and remain for G3.

- Patient-level/raw data committed: **No**
- NIGHT_LEVEL_CACHE: **NOT GENERATED**
- MODEL_TRAINING: **NOT STARTED**
- G2: **NOT STARTED**
