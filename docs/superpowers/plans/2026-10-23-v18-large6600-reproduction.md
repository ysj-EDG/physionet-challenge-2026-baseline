# Matcha V18 Large6600 Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the fixed Matcha V18 Large6600 reproduction with persistent, resumable artifacts entirely in the target workspace.

**Architecture:** Call the existing repository entrypoints in separate stages with explicit workspace paths. Use `tmux` for every encoder training/export job and keep `/database2` read-only. Make only the minimal embedding-cache fingerprint change permitted by the design.

**Tech Stack:** Python 3.10, `sleepfm_env`, PyTorch/CUDA, HDF5, Parquet, pandas, tmux, Git.

**Spec:** `docs/superpowers/specs/2026-10-23-v18-large6600-design.md`

## Global Constraints

- Source commit is `7f2eae2fe4f170c87a21943b24ea4e217eabf67f`.
- `/database2/physionet2026_kaggle/data2` is read-only input only.
- All artifacts stay below `/database/home/gaohaojie/workspace/repd/Matcha`.
- Every encoder training/export task uses a dedicated `tmux` session.
- Do not alter model, loss, sampling, labels, features, or scoring logic.
- Only the embedding shard checkpoint-fingerprint cache guard may be added.

## Review Focus

- A stale shard from another encoder checkpoint must never be reused; test with a mismatched HDF5 fingerprint.
- An interrupted E1 run must resume optimizer, teacher, epoch, and RNG state through the existing checkpoint path.
- Missing human/CAISR annotations must remain represented by upstream masks and not be fabricated.
- Non-contiguous eligible epochs must remain indexed by the upstream `window_index` and masks.
- A failed stage must leave prior persistent artifacts intact and produce a clear rerun command.

### Task 1: Fixed workspace, source, environment, and data manifest

**Files:**
- Create: `artifacts/v18_large6600/{prepared,encoder,frozen,loso,final_model,smoke,scratch,logs,reports}`
- Create: `artifacts/v18_large6600/reports/source_environment.json`
- Create: `artifacts/v18_large6600/prepared/data_validation.json`

- [ ] Verify branch, tag resolution, source commit, requirements, Dockerfile, Python, CUDA, and GPU mapping.
- [ ] Validate the Large data root, record count, site counts, labels, and source file presence without writing under `/database2`.
- [ ] Save the exact command and read-only validation results.

### Task 2: Minimal cache safety patch and focused regression test

**Files:**
- Modify: `export_full_embeddings.py` in `export_shard`
- Test: existing import/compile checks plus a focused temporary-directory shard fingerprint check

- [ ] Write a failing check showing an existing shard with a different checkpoint hash is not accepted as complete.
- [ ] Run the check and observe the expected failure.
- [ ] Add only the checkpoint-attribute comparison before the existing-file early return.
- [ ] Re-run the focused check and the repository compile/import checks.
- [ ] Commit this single cache-preservation change on `reproduction/v18-large6600`.

### Task 3: Real-data smoke in isolated artifacts

**Files:**
- Create: `artifacts/v18_large6600/smoke/*`
- Create: `artifacts/v18_large6600/logs/*`

- [ ] Build a roughly 48-record real-data preparation cache using the existing `prepare_training_data` path.
- [ ] Run smoke E1 in a dedicated tmux session with `TEST_MODE`, CUDA, and the upstream training script.
- [ ] Run EMA-teacher export in a second dedicated tmux session and validate HDF5/Parquet reload.
- [ ] Run existing downstream training, residual/runtime reload, and official interface smoke checks.
- [ ] Clear only the smoke test-mode environment for all subsequent commands.

### Task 4: Formal preparation and E1 checkpoint

**Files:**
- Create: `artifacts/v18_large6600/prepared/*`
- Create: `artifacts/v18_large6600/encoder/*`

- [ ] Prepare all eligible Large records with existing preprocessing and annotation alignment.
- [ ] Record successful/failed records, site counts, label counts, cache sizes, and missing modalities.
- [ ] Run the formal 15-epoch E1 job in a dedicated tmux session on the verified idle GPU with the V18 constants.
- [ ] Verify the epoch-15 checkpoint contains the EMA teacher, optimizer/RNG state, finite parameters, and matching manifest fingerprint.

### Task 5: Permanent frozen embedding export

**Files:**
- Create: `artifacts/v18_large6600/frozen/e1_seed20260804_epoch15/embeddings/*`
- Create: `artifacts/v18_large6600/frozen/e1_seed20260804_epoch15/metadata.json`

- [ ] Export teacher-state 192-D 30-second representations in a dedicated tmux session directly to the permanent workspace path.
- [ ] Verify shard fingerprints, masks, window indices, logits, manifest paths, finite values, record/site counts, and total windows.
- [ ] Mark the embedding cache immutable for downstream stages.

### Task 6: Cache-only LOSO and residual evaluation

**Files:**
- Create: `artifacts/v18_large6600/loso/*`

- [ ] Run the existing V18-compatible LOSO functions/entrypoint from the permanent embedding manifest for sites I0002, I0006, and S0001 and seeds 20260806/07/08.
- [ ] Preserve inner split, age-conditioned epoch selection, same-site pairwise loss, natural/site-balanced sampling, and all fold-local fitting rules.
- [ ] Save raw logits, probabilities, ensemble logits, OOF rows, fold metrics, and residual-adjusted outputs.
- [ ] Compute official AC-AUROC/AUROC/AUPRC metrics and macro/worst-site summaries.

### Task 7: Full-data final model and official runtime validation

**Files:**
- Create: `artifacts/v18_large6600/final_model/*`
- Create: `artifacts/v18_large6600/reports/REPRODUCTION_REPORT.md`

- [ ] Train final sequence seeds with fixed 2/4/1 epochs from the permanent embedding cache, using one tmux session per training job.
- [ ] Fit residual artifacts on all eligible training records and assemble the existing runtime package.
- [ ] Reload E1, all sequence checkpoints, residual JSON, and `team_code.py` runtime.
- [ ] Run official `run_model.py` inference on an independently listed real-record subset spanning all three sites.
- [ ] Write the report with actual results, differences, exact rerun commands, and status/续跑 command.
