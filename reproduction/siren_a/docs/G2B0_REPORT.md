# G2-B0 Full-Run Readiness and Targeted Pilot Report

## Status

`G2B0_STATUS = SUBMITTED_FOR_REVIEW`

This gate tested bounded execution readiness only. It did not generate the full 6600-record cache, change the frozen feature contract, impute or normalize features, use outcomes, or train a model.

## A. Git and frozen inputs

- Branch: `research/siren-a-stagepool-v1`
- Pre-commit local and remote HEAD: `ed5707531bfc0c3cc404b33baafae9627106c4b0`
- Commit parent for this gate: `ed5707531bfc0c3cc404b33baafae9627106c4b0`
- Frozen Manifest SHA256: `7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a`
- Frozen Spec SHA256: `f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a`
- Manifest dimensions: 807 unique features and 827 view positions; V1=113, V2=129, V3=349, V4=113, V5=123.
- The frozen files were re-hashed after implementation and were unchanged.
- The worktree changes were restricted to the ten files approved for G2-B0.

The final commit SHA and remote equality are recorded in the terminal delivery because a document cannot contain the SHA of the commit that contains itself.

## B. Golden regression

- Golden records checked: 18, six from each site.
- Exact matches: 18/18.
- Unique values, NaN positions, boolean masks, integer support counts, dtype, and shape all matched exactly.
- Every position in all five views matched exactly.
- All 18 stored source hashes matched the current read-only sources.
- Golden cache SHA256 snapshots taken before and after execution were identical to the current files.
- No Golden file or source record was rewritten.

## C. Optimization

The baseline was loaded directly from commit `ed5707531bfc0c3cc404b33baafae9627106c4b0`. Three fixed Golden sources, one per site, were used in the same Python process after warm-up, with five timed repetitions per source.

| Metric | Frozen baseline | Optimized |
|---|---:|---:|
| Median runtime per record | 0.7376 s | 0.1120 s |
| Optimized/baseline runtime | — | 0.1518 |
| Full `X_seq` float64 conversions | 843 | 1 |
| Median traced peak memory | 7.31 MB | 3.86 MB |
| Optimized/baseline traced peak | — | 0.5278 |

The observed speed-up was approximately 6.59 times. Record lengths in the comparison were 1026, 748, and 921 epochs. The optimization did not change any Golden output.

## D. Cache safety and recovery

- Publishing uses a same-filesystem temporary file followed by atomic hard-link publication; it does not use an existence check followed by replacement.
- The temporary file and final file are fully validated for keys, dtype, shape, finite/NaN semantics, view mapping, and provenance.
- Existing targets are rejected without modification.
- A real two-writer race produced exactly one winner.
- A simulated pre-publication failure left neither a final file nor a temporary residue.
- Resume actions are limited to `COMPUTE`, `SKIP_VERIFIED`, and `STOP_INVALID`.
- Both completed Pilot caches returned `SKIP_VERIFIED`.
- Corrupt content, wrong shape, wrong view mapping, and every changed source, contract, code, environment, or schema fingerprint returned `STOP_INVALID` in tests.
- Unexpected cache keys, including a label-like test key, are rejected rather than silently accepted for Resume.
- A forced final-file verification failure removed only the writer's newly linked, unverified target and left no final or temporary residue.

Two bounded attempts stopped on overly broad real-boundary selection predicates. Their private evidence was preserved without deletion or overwrite. RED-to-GREEN regression tests then required an independently computable EEG output for no-HRV and annotation-missing cases, and required available algorithmic annotation for the probability-incomplete case. The successful run used a new no-clobber subdirectory; the earlier attempts are not accepted Pilot outputs.
Intermediate successful retries were also retained but superseded after code review strengthened post-link cleanup and exact-key Resume validation. Only the final stricter-code run is treated as the accepted Pilot.


## E. Targeted Pilot

- Candidate seed: 20261013.
- Distinct source records read: 80, including the 18 Golden sources.
- Total source read calls: 82; repeated calls were replay checks.
- New Pilot records: 2.
- Site distribution: I0002=2, I0006=0, S0001=0.
- Golden plus accepted new Pilot records: 20, within the 30-record limit.
- Accepted caches: 2/2 passed write, close, safe reload, source hash, exact recomputation, view mapping, and Resume validation.
- Dimensions: 807 unique features and 827 view positions for every accepted cache.

Real boundary coverage:

| Boundary | Result |
|---|---|
| Race Unavailable | Covered |
| Non-9 probability outside [0,1] at a valid stage | Covered |
| No valid HRV with independently computable EEG | NOT COVERED BY REAL PILOT |
| Missing algorithmic annotation with independently computable EEG | NOT COVERED BY REAL PILOT |
| Incomplete or length-mismatched stage probabilities with available annotation | NOT COVERED BY REAL PILOT |
| Probability anomaly at an invalid stage beyond value 9 | NOT COVERED BY REAL PILOT |

The four required feature-local behaviors were covered synthetically. The bounded search was not enlarged when suitable real cases were absent.

Aggregate validity in the two accepted records:

| Family | Unique columns | Valid / total positions |
|---|---:|---:|
| EEG stage spectral | 648 | 1080 / 1296 |
| Spatial asymmetry | 24 | 36 / 48 |
| Spatial anterior-posterior gradient | 12 | 16 / 24 |
| CAISR20 | 20 | 38 / 40 |
| HRV11 pooled | 22 | 44 / 44 |
| EMG24 pooled | 48 | 96 / 96 |
| Resp14 pooled | 28 | 56 / 56 |
| Demo5 | 5 | 6 / 10 |

Missing values remained NaN with false masks. No record was dropped and no missing value was filled.

## F. Entropy probability QC

The 18 Golden records reproduced the prior public total of 540 values equal to 9. All 540 were at `stage_valid=False`; each of the five probability channels contributed 108.

The 28 additional non-9 out-of-range values were:

- all exactly `1.0000594854354858`;
- all greater than 1 and less than 9;
- all in the Wake probability channel;
- all at `stage_valid=True`;
- concentrated in one of the six I0006 Golden records;
- associated with source lengths that matched the epoch axis;
- not negative, non-finite, or greater than 9.

Source tracing shows that the v2.1 time-grid layer copies the raw annotation probability channel without clipping, while the upstream static entropy implementation clips probabilities before normalization. Therefore these values are raw source probability output, not the value-9 missing encoding. The producer-level reason for the small fixed overshoot is not established in this gate. The frozen strict entropy rule correctly rejects the record without clipping, recomputing entropy, changing `x_static[87]`, or invalidating the other CAISR features.

Strict entropy source validity was 0/18 Golden records.

`ENTROPY_COVERAGE_ALERT = TRIGGERED`

This is a coverage warning for human review, not a pooling calculation failure.

## G. Full inventory preflight

- Total source files: 6600.
- I0002: 319; I0006: 1142; S0001: 5139.
- Duplicate record keys: 0.
- Symlinks: 0.
- Unexpected extensions or directory depth: 0.
- Input and output roots were distinct.
- The private inventory contains relative names and file-system metadata for later review.
- Only names and file-system metadata were enumerated for all 6600 files; their contents were not scanned.

## H. Validation and gate decision

- G1-A: 15/15 function tests passed using a standard-library runner because pytest is not installed.
- G1-B and G2-A/G2-B0: 46/46 unittest tests passed.
- Total: 61/61 tests passed.
- Python syntax checks passed for all seven modified or added code/test modules.
- One initial package-style G1-A invocation failed because that historical file uses a local import; rerunning it from its own directory with the explicit function runner passed all 15 tests. No environment was changed.
- No runtime or memory regression was observed.
- No source record, Golden cache, frozen contract, historical report, or upstream extractor was modified.

The implementation is technically ready for G2-B1 review. Human review is still required for the entropy coverage alert, the raw Wake-probability overshoot, and the real boundary categories not found within the fixed 80-record budget.

`G2A_GOLDEN_CACHE = UNCHANGED`

`G2B0_PILOT_CACHE = GENERATED`

`G2B1_FULL_CACHE = NOT GENERATED`

`G3_PREPROCESSING = NOT STARTED`

`SITE_RANK_NORMALIZATION = NOT STARTED`

`MODEL_TRAINING = NOT STARTED`
