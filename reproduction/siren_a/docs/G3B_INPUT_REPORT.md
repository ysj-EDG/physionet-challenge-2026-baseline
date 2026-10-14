# G3-B Model Input Adapter Report

`G3B_STATUS = SUBMITTED_FOR_REVIEW`

## Frozen inputs and identity join

- Branch: `research/siren-a-stagepool-v1`; parent: `c87e4db01dbd3445334dbc9918efe48733238841`.
- G2-B1 cache fingerprint: `cf88b4e87f8b524258c685db696c278198dae1e6a10bd39fb63e893b2df031df`.
- G3-A artifact set: 52 files, 417,435,113 bytes, aggregate SHA256 `4a0f3956432030e4fa2fb8b51d15aa8fa5269f990d7795cca40d580970920a64`.
- G3-A-V evidence: six of six units report PASS; completion hashes and verifier code hashes were rechecked.
- Frozen Manifest SHA256: `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a`.
- Frozen Spec SHA256: `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a`.
- DATA2 demographics SHA256: `b9009598589846e84462636bfcaa290ed1d8666c89b1bc1ad1a9eda26b43ff7f`.
- Exact composite-key join: 6,600/6,600 caches matched; duplicate keys 0; unmatched records 0; site mismatches 0.
- Site counts closed at I0002=319, I0006=1,142, and S0001=5,139. CSV row order was not used for target alignment.
- `BDSPPatientID` was present for all records. Site-scoped duplicate patients and Train/Test patient overlaps were both 0.

Only selected training rows are parsed into labels and ages. `FoldInputs` contains no test label, test age, patient identifier, record path, or unused-site target array. Age and labels are metadata fields, never model columns.

## LOSO labels and age groups

| Holdout | Train site | Train N | Positive | Negative | Finite age | 5-year groups | Single-class groups | Groups with both classes |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| I0002 | S0001 | 5,139 | 334 | 4,805 | 5,139 | 9 | 0 | 9 |
| I0006 | S0001 | 5,139 | 334 | 4,805 | 5,139 | 9 | 0 | 9 |
| S0001 | I0006 | 1,142 | 112 | 1,030 | 1,142 | 9 | 0 | 9 |

All parsed training ages were finite and ranged from 50 to 90 years. Groups use `floor(age / 5)`. No group was dropped, merged, or supplemented. LambdaRank query ordering and single-class policy remain explicitly deferred to G4.

## Model-input adapters

All three folds, both normalization protocols, five frozen views, and three adapter types were checked: 90/90 combinations passed. View widths remained V1=113, V2=129, V3=349, V4=113, and V5=123. Transductive and Inductive training arrays, masks, row order, labels, ages, and feature IDs matched exactly within each fold; test arrays were allowed to differ by protocol.

- LR: the `full_train_preview` state uses each current training matrix column's finite median. A training-subset all-missing column would use `128/255`; no full-training preview view had an all-missing column. All preview LR outputs had zero remaining NaNs. Observed Rank256 values were unchanged. G4 must refit a separate state on each actual 70% seed subset.
- HGB: float64 shape and frozen feature order were preserved; native NaNs remained unchanged; no scaling or missing indicator was added.
- LambdaRank: the same native-NaN passthrough contract was verified. Labels and 5-year training age groups are separate metadata; Age, SiteID, and labels never enter X.

Training-source missing positions by frozen training site were identical across protocols. For S0001 training they were V1=14,013, V2=41,737, V3=81,027, V4=13,437, V5=15,059. For I0006 training they were V1=862, V2=10,404, V3=27,832, V4=844, V5=1,636. LR preview reduced these to zero through train-only imputation; HGB and LambdaRank retained the original counts.

The low-coverage stage-entropy feature remains present in V5. No feature was selected, removed, rescaled, or appended.

## Isolation, provenance, and recovery

Every compact adapter state records the adapter type, feature-order SHA, fit-row SHA, finite counts, fill values, all-missing flags, and fit scope. Every combination check additionally records exact train/test index fingerprints and G3-A unit source fingerprints. The root identity records G2-B1, G3-A, G3-A-V, Manifest, Spec, demographics, metadata join, code, Python, and NumPy fingerprints.

Private canonical output: `/database/home/gaohaojie/workspace/python-example-2026-siren-a-g3b-private/model_inputs_v1`.

The canonical output contains 192 files and approximately 4.3 MiB; it references rather than duplicates G3-A Rank256 matrices. Completion is published last with file SHA256 closure. `verify-only` independently rebuilt all targets, ages, age groups, adapter states, masks, and checks. Repeated build verified and skipped the completed output. One initial failed write and two superseded successful code-fingerprint versions were retained under distinct local names as audit evidence; none was deleted or presented as canonical.

The final build completed in 115.77 s wall time with peak RSS 199,800 KiB and zero swap. No-clobber, incomplete-build rejection, exact public-summary resume, and source-drift rejection were tested.

## Validation

- New G3-B tests: 21/21 passed, with recorded RED→GREEN cycles for all three modules and recovery fixes.
- Historical unittest regression (G1-B, G2-A, G2-B0, G2-B0-R, G2-B1, G3-A, G3-A-V, and G3-B): 142/142 passed.
- G1-A direct regression: 15/15 passed.
- Combined executed regression checks: 157 passed, 0 failed.
- Python syntax compilation passed.
- Public aggregate CSV contains exactly 90 rows and only fold/protocol/view/model-level statistics.
- No estimator package is imported or instantiated; no `.fit()` call, model training, scoring, standardization, feature selection, or site-rank recomputation was performed.

## Gate state

- `G3B_IDENTITY_JOIN = PASS`
- `G3B_TRAIN_LABELS = PASS`
- `G3B_AGE_GROUPS = PASS`
- `G3B_LR_ADAPTER = PASS`
- `G3B_HGB_ADAPTER = PASS`
- `G3B_LAMBDARANK_ADAPTER = PASS`
- `G3B_TRAIN_TEST_ISOLATION = PASS`
- `G3B_90_INPUT_COMBINATIONS = VERIFIED`
- `G3A_RANK256 = UNCHANGED`
- `G3AV_VERIFICATION = UNCHANGED`
- `G4_MODEL_TRAINING = NOT STARTED`
- `G5_RANK_ENSEMBLE = NOT STARTED`

G3-B prepares only deterministic model inputs and preview adapter states. Model hyperparameters, class weighting, 70% sampling, LambdaRank query policy, 15 seeds, and ensemble rules remain unfrozen for G4/G5 review.
