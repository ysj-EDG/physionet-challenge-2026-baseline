# G2-B0-R CAISR EDF Audit Report

## Gate result

`G2B0R_STATUS = NEEDS_REVIEW`

`CAISR_CENSUS_STATUS = COMPLETE`

The 6,600-record CAISR EDF census, frozen-NPZ stage metadata cross-check, Resume
hardening, and annotation-missing correction completed. Review is required
because neither the CAISR `<20` count nor the no-stage-source-inclusive count
matches the 18 records reported by SIREN; the paper's exact `scorable` criterion
cannot be established from the current sources.

## Environment and sources

- Branch: `research/siren-a-stagepool-v1`
- Execution parent/local/remote SHA before this commit:
  `8831833c64d4450685bb3eb32110dfdeed2f7226`
- Parent of that execution SHA:
  `ed5707531bfc0c3cc404b33baafae9627106c4b0`
- Input cohort: data2 training records only; I0002=319, I0006=1142,
  S0001=5139, total=6600
- Frozen NPZ: `timegrid_v2.1.0`, validity schema `validity_v0.2`
- Frozen Manifest SHA256:
  `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a`
- Frozen Spec SHA256:
  `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a`
- edfio: 0.4.11, lazy stage-channel reads
- Execution: single worker, 100-record atomic checkpoints; 6,600 records
  completed in approximately 7 min 23 s

Only the three approved identity columns were consumed from demographics. No
expert/human annotation source, diagnosis, follow-up field, age, hidden data,
or model label was accessed. EDF paths were constructed exactly; no fuzzy file
search or annotation fallback was used.

## CAISR file census

| Site | Total | EDF Missing | EDF Present | Unreadable | No stage_caisr | Raw valid <20 | Raw valid >=20 | Aligned valid <20 | Aligned valid >=20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 319 | 2 | 317 | 0 | 0 | 0 | 317 | 0 | 317 |
| I0006 | 1142 | 4 | 1138 | 0 | 0 | 0 | 1138 | 0 | 1138 |
| S0001 | 5139 | 64 | 5075 | 0 | 0 | 6 | 5069 | 6 | 5069 |
| Overall | 6600 | 70 | 6530 | 0 | 0 | 6 | 6524 | 6 | 6524 |

`EDF Missing/Present` use all records as denominator. Raw and aligned threshold
columns use only `FILE_PRESENT_STAGE_READABLE` records (317, 1138, 5075; total
6530); missing/unreadable/no-stage files are not silently treated as zero valid
epochs. All three site-level mutually exclusive file-state totals close exactly
to the corresponding cohort totals.

All 6,530 present EDF files were readable and contained exactly one normalized
`stage_caisr` channel. Every channel had sampling frequency 1/30 Hz; there were
no frequency warnings. Every readable record contained finite codes outside
1–5, totaling 39,180 invalid/unknown epochs; these were excluded exactly as
specified and were not rounded or remapped.

## Valid-epoch distributions

### Raw CAISR EDF

| Site | 0 | 1–4 | 5–19 | >=20 | P05 | P25 | Median | P75 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 0 | 0 | 0 | 317 | 750.0 | 858.0 | 911.0 | 992.0 | 1081.6 |
| I0006 | 0 | 0 | 0 | 1138 | 740.0 | 787.0 | 835.0 | 880.0 | 972.0 |
| S0001 | 0 | 1 | 5 | 5069 | 766.0 | 853.0 | 917.0 | 972.0 | 1038.0 |
| Overall | 0 | 1 | 5 | 6524 | 754.45 | 836.0 | 901.0 | 964.0 | 1036.0 |

The six low-count readable records had exact valid counts 2 (one record), 7
(two), 10 (two), and 15 (one), all at S0001.

### Aligned to frozen v2.1 grid

The aligned table is identical to the raw table above. There were zero records
with raw/aligned valid-count differences, zero records with raw>=20 but
aligned<20, and zero readable records whose raw stage length differed from T.

For the existing NPZ `stage_valid` field across all records, including records
without an EDF source, the `<20` counts are I0002=2, I0006=4, S0001=70,
Overall=76. These comprise 70 exact-path EDF-missing records with zero valid
aligned stages plus the six readable low-count records. They are reported for
comparison only; no record was removed.

## EDF versus frozen NPZ consistency

| Check | Matches | Denominator | Mismatches |
|---|---:|---:|---:|
| `annotation_status` | 6600 | 6600 | 0 |
| NaN-aware `stage_code_aligned` | 6600 | 6600 | 0 |
| `stage_valid` mask | 6600 | 6600 | 0 |
| `stage_valid.sum()` / aligned count | 6600 | 6600 | 0 |

The 70 missing EDFs correspond exactly to NPZ `annotation_status=missing`, an
all-NaN aligned stage vector, and an all-false stage-valid mask. The 6,530
readable EDFs correspond exactly to `annotation_status=available`. There were
no source-snapshot mismatches, unreadable files, or indeterminate comparisons.

## Comparison with the SIREN 18-record exclusion

SIREN reports 18 records with fewer than 20 scorable 30-second epochs:
I0002=2, I0006=0, S0001=16. The current results are:

- CAISR raw valid stage `<20`: 0, 0, 6 (total 6).
- CAISR aligned valid stage `<20` among readable stage EDFs: 0, 0, 6.
- Exact-path CAISR EDF missing: 2, 4, 64 (total 70).
- Frozen NPZ `stage_valid<20`, including missing-stage records: 2, 4, 70
  (total 76).

Thus neither CAISR interpretation reproduces 18 or its site distribution.
Possible reasons include a different definition of `scorable`, a different raw
annotation/data snapshot, treatment of missing algorithmic annotations, or a
different source/time-grid scope. Count coincidence at I0002 is insufficient
to establish implementation equivalence. The paper's exact exclusion cannot
be reconstructed from the authorized inputs, so no 6582-record cohort is
created and no exclusion rule is changed.

## Resume hardening

The validator now checks exact `float64` value, `bool` mask, and `int64` support
dtypes and exact shapes before indexing. Invalid dtype/shape inputs return an
invalid result and `STOP_INVALID`, rather than leaking an `IndexError`.

Manifest-family support rules are now enforced even if unique and View arrays
are altered consistently:

- EEG648, Spatial36, HRV22, EMG48, Resp28: valid=>support>=5 and
  invalid=>support<5.
- CAISR20 and Demo5: valid=>support=1 and invalid=>support=0.

Tests passed for float/int masks, shape errors, negative support, valid pooled
support=0, invalid pooled support=5, valid CAISR support=0, invalid Demo support=1,
synchronized unique/View tampering, corrupted NPZ, provenance mismatch,
`STOP_INVALID`, and normal `SKIP_VERIFIED`.

## Annotation-missing semantics

The boundary selector no longer requires stage-conditioned EEG output when
`annotation_status=missing`. In the upstream-faithful synthetic condition
(stage code all NaN, stage-valid all false), valid outputs were:

| Family | Valid / Total | Meaning |
|---|---:|---|
| EEG648 | 0 / 648 | Missing because no stage source |
| Spatial36 | 0 / 36 | Missing because no stage source |
| CAISR20 | 0 / 20 | Missing under frozen source rules |
| HRV22 | 22 / 22 | Independently governed |
| EMG48 | 48 / 48 | Independently governed |
| Resp28 | 28 / 28 | Independently governed |
| Demo5 | 5 / 5 | Independently governed |

The record remains present as an 807-feature output. This test distinguishes
EEG signal presence from calculability of stage-conditioned EEG features and
does not use an expert-stage fallback.

## Recovery, regression, and source protection

- The completed 66 audit chunks replayed successfully without overwriting;
  all 6,600 rows and aggregate results were identical.
- Standard-library/unittest suites: 58/58 passed, including G1-B, G2-A,
  G2-B0, new audit/Resume/pilot tests, and the Golden exact regression.
- G1-A's 15 fixture-free pytest-style functions were invoked directly because
  pytest is not installed and dependencies were not changed: 15/15 passed.
- Effective final regression total: 73/73 passed.
- Golden regression: 18/18 caches matched exactly for unique/View values,
  NaN positions, masks, support counts, shapes, and dtypes.
- Golden file SHA256 snapshot: 20/20 files unchanged before/after.
- Accepted Pilot file SHA256 snapshot: 11/11 files unchanged before/after.
- Source size/mtime verification: NPZ 6600/6600 and CAISR EDF existence/
  size/mtime state 6600/6600 unchanged after audit.
- Frozen Manifest and Spec hashes remained unchanged.

Detailed identities, mappings, file errors, checkpoints, and SHA evidence are
kept only in the dedicated private audit root. No patient-level identity,
filename, path, stage sequence, feature vector, age, or label is included in
this Git artifact.

## Gate decision

The census and bounded code corrections are technically complete. The only
unresolved semantic issue is the non-equivalence between the current CAISR
valid-stage counts and SIREN's 18-record `scorable` exclusion. This requires
human review and does not block the integrity of the 6,600-record audit itself,
but it prevents automatic G2-B1 release.

`CAISR_6600_EDF_AUDIT = COMPLETED`

`G2B0_RESUME_CORRECTION = COMPLETED`

`G2A_GOLDEN_CACHE = UNCHANGED`

`G2B0_PILOT_CACHE = UNCHANGED`

`G2B1_FULL_CACHE = NOT GENERATED`

`G3_PREPROCESSING = NOT STARTED`

`SITE_RANK_NORMALIZATION = NOT STARTED`

`MODEL_TRAINING = NOT STARTED`
