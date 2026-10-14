# G3-B Model Input Adapter Specification

## Scope

G3-B creates a label-safe, model-independent input layer over frozen G3-A Rank256 artifacts. It connects the audited DATA2 demographics source to 6,600 anonymous G2-B1 records through the exact SiteID, BidsFolder, and SessionID composite key, exposes labels and age only for selected LOSO training rows, adapts five frozen views for LR, HGB, and LambdaRank, and never trains a model.

Inputs are read-only: G2-B1 night_cache_v1, G3-A rank256_v1, G3-A-V verification_v1, Frozen Manifest and Spec, and DATA2 demographics.csv. Existing tracked code and all G1/G2/G3 artifacts remain unchanged.

## Identity and label contract

The join is independent of CSV row order. Each private-index relative-path stem must equal BidsFolder + _ses- + SessionID, and SiteID must match separately. All 6,600 composite keys, cache names, and source paths must be unique, with no unmatched rows or site conflicts.

Cognitive_Impairment accepts case-insensitive True/False, 1/0, T/F, and Yes/No only. Empty, missing, or unknown values are errors. Age must parse to a finite value in the explicit plausibility interval 0–130 years and is never included in model features; the audited DATA2 values are 50–90 years. BDSPPatientID, when present, is checked in a SiteID-scoped namespace for repeated studies and train/test overlap; no patient identity enters public output.

The private metadata index may retain raw identity records only for auditing. load_train_targets receives frozen training indices and returns y_train and age_train. No API returns y_test, age_test, unused-site targets, or a global 6,600-label vector.

## Fold input contract

load_fold(holdout_site, protocol, view) loads existing G3-A indices, Rank256 view values, and rank-valid masks. FoldInputs contains X_train, X_test, train_valid_mask, test_valid_mask, y_train, age_train, train_age_groups, feature_ids, train_indices, test_indices, frozen fold identifiers, and source fingerprints. It contains no test labels, test age, patient identifiers, source paths, or outcome fields.

Frozen folds are I0002 from S0001, I0006 from S0001, and S0001 from I0006. Protocols are transductive and inductive. Views remain V1=113, V2=129, V3=349, V4=113, and V5=123. Train matrices, labels, ages, groups, indexes, masks, and ordering must be identical across protocols for a fold; test matrices may differ.

Feature IDs come only from the Frozen Manifest. Age, labels, SiteID, patient IDs, BidsFolder, SessionID, follow-up fields, and diagnoses are not model columns.

## Adapter contract

fit_adapter supports lr, hgb, and lambdarank. For lr, each finite training-subset column uses its median. A column with no finite selected values uses 128/255. State records finite counts, fill values, all-missing flags, fit scope, fit-index fingerprint, and feature-ID fingerprint. transform_adapter applies the frozen state to train or test; it never fits on transform data, adds no indicator, performs no scaling, and preserves observed values exactly.

For hgb and lambdarank, state is a provenance-bearing passthrough declaration. Transform preserves float64 matrices and native NaNs exactly, adds no columns, and performs no normalization or imputation.

The full_train_preview adapter is only a G3-B interface check. G4 must call fit_adapter on each actual 70 percent seed subset and create a distinct state.

## Age groups

make_age_groups computes floor(age / 5) as int64. The pipeline audits group sizes and class counts, single-class groups, groups smaller than two, and groups lacking both classes. It does not drop, merge, or supplement groups. Query-group ordering and single-class training policy remain for G4.

## Source integrity and recovery

Preflight validates G2-B1 identity, G3-A run identity and six completion files, G3-A-V completion and code fingerprints, the reported 52-file G3-A aggregate fingerprint, Frozen Manifest/Spec SHA256, and demographics content SHA256. Any drift stops execution.

Private outputs record required source, code, metadata, index, feature-order, adapter, and training-subset fingerprints plus Python and NumPy versions. Outputs are atomic, no-clobber, completion-last, and independently verifiable. G3-B references G3-A arrays rather than duplicating all normalized matrices.

## CLI and privacy

The module launcher supports only --preflight-only, --smoke, --build, and --verify-only. Smoke checks 30 I0002 combinations; build checks 90 combinations. Neither mode imports or calls model estimators.

Private output lives only under the approved external model_inputs_v1 root. Public CSV and report contain fold/protocol/view/model-level aggregates only. Patient identifiers, record IDs, paths, row-level ages, labels, and features are prohibited.

## Gate

Submission requires an exact 6,600-row join, legal training labels and ages, zero train/test patient overlap, verified fold isolation, 90 valid combinations, LR subset-fit semantics, native-NaN passthrough for HGB/LambdaRank, recoverable private evidence, full regression, unchanged G2/G3/G3-A-V inputs, and no model training.
