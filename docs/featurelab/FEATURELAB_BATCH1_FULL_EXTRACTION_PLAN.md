# FeatureLab Batch 1 Full Extraction Plan

```text
FULL_6600_STATUS = NOT_STARTED
FULL_EXTRACTION_AUTHORIZATION = REQUIRED
BATCH1_RECIPE_STATE = FROZEN
BATCH2_STATE = AUDIT_ONLY
TRAINING_AND_SCORING = OUT_OF_SCOPE
```

This document is an execution and acceptance plan. It does not authorize or start extraction. The full command must not be run until a later explicit approval creates the private authorization file described below.

## Frozen scope

The only computations are the six approved NPZ-derived Batch 1 families. They produce 637 ordered candidate columns per input file and do not read EDF, rerun the legacy extractor, construct five-view inputs, or access labels, TTE, predictions, or scores.

| family | columns | frozen Recipe ID |
|---|---:|---|
| eeg_tail | 288 | `17f79dd79b79c4cc26ab0477b6c936e28cfb31161e9c0bb86ff44a21fa496054` |
| eeg_temporal | 96 | `5dc03f81224b71d166ec736f0517575e982debecab46f06c85ae2c0a1446f8bd` |
| eeg_coherence_stage | 90 | `c2c6949634746176d99c204792e5f860384530fee1a35ce54d5f4e41eb676a39` |
| caisr_stage | 35 | `6529a32282e29ec44ef5af0290379c2d11f3d70527504a513b5121e6069ed7f3` |
| emg_stage | 72 | `29cd008bf82a47a6fa823c400d8bca4b5e71f6a7c68a8b7a7eb7682da74ec91b` |
| resp_stage | 56 | `3c1c051e7091cfeac73160a112f048208156cbdac1b5f4e33e24e4ce1825dd35` |

Any Recipe change requires a new review. A full-run operator must not alter parameters, formulas, feature order, validity semantics, stage mapping, or minimum support to reduce missingness.

## Source inventory and private destinations

The deterministic path-only inventory currently contains exactly 6,600 files:

| site | files |
|---|---:|
| I0002 | 319 |
| I0006 | 1,142 |
| S0001 | 5,139 |
| **total** | **6,600** |

- Frozen source: `/database/home/gaohaojie/workspace/python-example-2026/validity_v0_2_full_6600/workspace/result_bundle/npz`
- Family cache root: `/database/home/gaohaojie/workspace/feature-cache-extension`
- Proposed private full-run checkpoint: `/database/home/gaohaojie/workspace/feature-cache-extension/full_batch/checkpoint.json`
- Proposed private authorization file: `/database/home/gaohaojie/workspace/feature-cache-extension/full_batch/FULL_EXTRACTION_APPROVED.txt`

The checkpoint contains the private site/file mapping and per-family statuses. It must never be added to Git. Final family files retain the existing layout `<cache root>/<family>/<Recipe>/<safe key>.npz`; there is no second cache format.

## Batch execution contract

`extensions.batch_runner` separates path inventory from NPZ decoding. It requires an explicit positive maximum, worker count, family selection, execution mode, expected full inventory size, and—for full mode—a private file whose complete trimmed content is exactly `FEATURELAB_FULL_EXTRACTION_APPROVED`.

For each selected source it calls `run_record()` once with the six-family tuple. That source is therefore loaded once per request rather than once per family. The underlying runner independently decides `BUILD_NEW`, `SKIP_VERIFIED`, or `STOP_INVALID` for every family and never overwrites a final file.

The private checkpoint is atomically replaced after each completed source. It preserves `COMPLETE`, `PARTIAL`, `STOP_RECORD`, and all five family statuses. A resumed command conservatively revalidates completed source/cache identities; verified families skip numerical extraction, while missing family files build. An invalid final file remains `STOP_INVALID` and is never repaired automatically.

### Gate 4F frozen full-run request

The first 6,600-record full run is machine-checked against all of the following values before any source is opened:

- the ordered family tuple is exactly `eeg_tail`, `eeg_temporal`, `eeg_coherence_stage`, `caisr_stage`, `emg_stage`, and `resp_stage`;
- every family uses its registered `FamilySpec` defaults, with no configuration or extractor override;
- the ordered manifest contains exactly 637 unique Feature IDs and has frozen canonical digest `92f071616f961256ac993a68624ef8f44f262a0a6738011e4bb0c4ec405d8039`;
- the six live Recipe IDs and dimensions equal the frozen table above;
- `workers=1`, `max_records=6600`, and `expected_inventory_size=6600`;
- the inventory site totals are exactly I0002=319, I0006=1,142, and S0001=5,139; and
- the authorization path is exactly `/database/home/gaohaojie/workspace/feature-cache-extension/full_batch/FULL_EXTRACTION_APPROVED.txt`, and an actual run additionally requires its exact token.

Any mismatch refuses startup; the runner never changes its configuration to match observed data. Full mode has a run-level safety stop, whereas Pilot mode retains its approved record-isolation behavior. In Full mode an abnormal record is written completely to the private checkpoint before a stop becomes active, and no subsequent record is scheduled. Because the first Full run is fixed to one worker, there can be no already-submitted background records after the stop boundary.

The checkpoint records the stop reason, completed-record count, abnormal status, safe-resume flag, repeated-error signatures, and stop history. A nonrecoverable active stop cannot be resumed by simply rerunning the command. A resource stop is marked safe to retry only after resources again pass the startup guard. The command-line exit status is nonzero for every safety stop and for a Full run that has not passed final acceptance.

## Resource estimate and worker recommendation

The cross-site Pilot used one real worker over six files, of which five required all six new family builds and one reused all six caches.

| metric | observation or estimate |
|---|---:|
| full source bytes | 9,304,850,942 bytes (8.67 GiB) |
| source file size min / median / max | 13,657 / 1,459,918.5 / 3,023,070 bytes |
| new-build mean / median | 2.506 / 2.492 s per source |
| one-worker idealized wall estimate | 4.60 h |
| planning wall range | 4.5–7 h |
| Pilot peak RSS | 356,176 KiB (about 348 MiB) |
| projected family-cache payload | about 531 MB (506 MiB) |
| required free capacity at Full startup | at least 2 GiB |
| required remaining capacity during Full run | at least 1 GiB |
| Gate 4F peak-RSS stop threshold | 1,048,576 KiB (1 GiB) |

The timing model has only five newly built real sources and must be treated as low-confidence. Resume validation, filesystem contention, source heterogeneity, and exception handling can increase wall time. Source immutability checking and source adaptation read each compressed file multiple times, so total read traffic can exceed 35 GiB even though the source inventory itself is 8.67 GiB.

Recommended initial full-run setting: `workers=1`. It is the only real-data concurrency level measured, completes within a practical planning window, limits simultaneous decompression memory, and avoids saturating a shared filesystem. Synthetic no-clobber competition passed for two workers, but that is not evidence of real I/O or memory scaling. Raising the full run above one worker requires a separate monitored authorization or a later evidence-based adjustment; it must not be inferred from the ideal two-worker estimate.

Disk capacity is checked once before scheduling and once after each completed Full-mode record using a filesystem free-space query; no cache-tree scan is added. Full startup stops below 2 GiB, and an active run stops below 1 GiB after first checkpointing the just-completed record. Peak process RSS is sampled on the same bounded schedule. The 1 GiB RSS ceiling is deliberately about three times the Pilot peak; crossing it is an abnormal-growth safety stop, not a request to add workers or delete existing caches.

## Read-only preflight before authorization

1. Confirm the FeatureLab branch and approved SHA, a clean worktree, and unchanged Branch A.
2. Confirm the source inventory remains exactly 6,600 with the stated site totals.
3. Confirm all six current Recipe IDs equal the frozen table above.
4. Confirm at least 2 GiB free at the cache destination.
5. Confirm the private checkpoint destination is outside Git and any existing checkpoint matches the inventory/request digest.
6. Execute the `--preflight-only` command below. It reads inventory paths, the registry/Recipe contract, filesystem capacity, process RSS, and an existing checkpoint if present. It never decodes a source NPZ, calls an extractor, writes a cache, or creates/modifies a checkpoint.

```bash
cd /database/home/gaohaojie/workspace/python-example-2026-featurelab
PYTHONPATH=per_epoch_features PYTHONNOUSERSITE=1 \
  /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python \
  -m extensions.batch_runner \
  --source-root /database/home/gaohaojie/workspace/python-example-2026/validity_v0_2_full_6600/workspace/result_bundle/npz \
  --cache-root /database/home/gaohaojie/workspace/feature-cache-extension \
  --checkpoint /database/home/gaohaojie/workspace/feature-cache-extension/full_batch/checkpoint.json \
  --max-records 6600 \
  --workers 1 \
  --execution-mode full \
  --expected-inventory-size 6600 \
  --full-authorization-file /database/home/gaohaojie/workspace/feature-cache-extension/full_batch/FULL_EXTRACTION_APPROVED.txt \
  --preflight-only
```

Before authorization, an absent authorization file is reported as `authorization_present=false`, `authorization_valid=false`, and `execution_permitted=false`; this is the expected safe result, not permission to run. A valid read-only contract preflight exits zero. Contract, inventory, Recipe, checkpoint, disk, or RSS failures exit nonzero.

## Preflight after future authorization

After explicit approval creates the private authorization file:

1. Repeat the read-only preflight and require `execution_permitted=true`.
2. Execute the Full-mode dry run below. It must report 6,600 selected, zero executed, and no cache or checkpoint mutation.

```bash
cd /database/home/gaohaojie/workspace/python-example-2026-featurelab
PYTHONPATH=per_epoch_features PYTHONNOUSERSITE=1 \
  /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python \
  -m extensions.batch_runner \
  --source-root /database/home/gaohaojie/workspace/python-example-2026/validity_v0_2_full_6600/workspace/result_bundle/npz \
  --cache-root /database/home/gaohaojie/workspace/feature-cache-extension \
  --checkpoint /database/home/gaohaojie/workspace/feature-cache-extension/full_batch/checkpoint.json \
  --max-records 6600 \
  --workers 1 \
  --execution-mode full \
  --expected-inventory-size 6600 \
  --full-authorization-file /database/home/gaohaojie/workspace/feature-cache-extension/full_batch/FULL_EXTRACTION_APPROVED.txt \
  --dry-run
```

The authorization file is deliberately not created by this plan or by `--preflight-only`.

## Explicit future launch and resume command — do not execute in this phase

Only after a new approval, remove `--dry-run` and run exactly:

```bash
cd /database/home/gaohaojie/workspace/python-example-2026-featurelab
PYTHONPATH=per_epoch_features PYTHONNOUSERSITE=1 \
  /database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python \
  -m extensions.batch_runner \
  --source-root /database/home/gaohaojie/workspace/python-example-2026/validity_v0_2_full_6600/workspace/result_bundle/npz \
  --cache-root /database/home/gaohaojie/workspace/feature-cache-extension \
  --checkpoint /database/home/gaohaojie/workspace/feature-cache-extension/full_batch/checkpoint.json \
  --max-records 6600 \
  --workers 1 \
  --execution-mode full \
  --expected-inventory-size 6600 \
  --full-authorization-file /database/home/gaohaojie/workspace/feature-cache-extension/full_batch/FULL_EXTRACTION_APPROVED.txt
```

The resume command is identical. Do not remove the checkpoint or any final family file after interruption. Reuse is based on validated SourceIdentity, FamilyInputIdentity, Recipe, feature order, and payload digest—not merely checkpoint text.

## Monitoring and review queue

- Treat the private checkpoint as the progress ledger. Its request section records attempts, source status, family status/reason, elapsed seconds, cache bytes, source immutability, and readback status after each source.
- Derive ETA from completed new-build elapsed times, excluding all-cache skips, and report a range rather than a single precise completion time.
- Keep aggregate counters for all record and family statuses by site. Keep the private identity mapping and detailed reasons outside Git.
- Queue `REQUIRES_INPUT_REVIEW`, `FAILED`, `STOP_INVALID`, and `STOP_RECORD` for explicit review. Legal per-column NaN and frozen missing reasons are not record failures.
- Cache-size monitoring must distinguish newly published bytes from total verified bytes, because prior Pilot/Smoke files will skip.
- Never use label, outcome, prediction, model score, or feature value to reprioritize the queue.

## Stop conditions

Full mode applies the following stop policy after atomically checkpointing the abnormal record:

Immediate run-level stops are:

1. any final family cache reports `STOP_INVALID`;
2. cache strict readback or full payload comparison is false;
3. the source NPZ SHA256/stat changes, or either the read-before or read-after source snapshot cannot be established;
4. the live Recipe, ordered Feature IDs, dimensions, or registered defaults differ from the frozen contract;
5. startup free space is below 2 GiB, runtime free space is below 1 GiB, or peak RSS exceeds 1 GiB.

Systemic run-level stops occur on the second distinct record within one site carrying the same source-contract incompatibility, or on the second distinct record within one site and family carrying the same family-input contract incompatibility. Repeated-error counters use record keys so a resumed observation of the same record does not falsely count twice.

An isolated ordinary computation failure, `REQUIRES_INPUT_REVIEW`, or non-systemic `STOP_RECORD` is recorded and isolated. A record containing any failed family is `PARTIAL`, never `COMPLETE`. Legal low-support NaN values remain valid family outputs under their per-feature missingness rules and never trigger a run stop.

Pilot mode retains its prior bounded behavior: one record can be isolated while the Pilot queue continues. Neither mode converts `STOP_INVALID` to `BUILD_NEW`, deletes a bad cache, relaxes validity checks, or changes feature formulas.

## Final full-cohort acceptance criteria

After an authorized run, a separate read-only audit must verify:

- exactly 6,600 inventory entries with no duplicate path or safe identity collision;
- exactly 39,600 family outcomes accounted for across six frozen Recipes;
- each existing final cache passes schema, complete status, logical digest, Recipe, source/family-input identity, ordered Feature ID, validity/NaN, support, and missing-reason validation;
- dimensions remain 288, 96, 90, 35, 72, and 56, totaling 637 per complete source;
- all `PARTIAL` and non-success family outcomes have explicit private reasons and no fabricated values;
- source immutability is true for every attempted source;
- per-site status and missing-reason distributions are reviewed for systematic divergence;
- CAISR `STATIC_BOUT_SEMANTIC_MISMATCH` remains invalid and is never hidden by tolerance changes;
- no EDF, old cache, five-view manifest, model, prediction, Branch A file, or Batch 2 implementation was modified;
- Git contains code and aggregate documentation only, never private inventory/checkpoint or patient-level output.

Passing these criteria would authorize review of downstream migration; it would not itself authorize model training, LOSO, Rank256, or five-view changes.
