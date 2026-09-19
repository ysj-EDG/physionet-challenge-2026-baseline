# Workspace cleanup

## Baseline

- Standalone baseline: `baseline.py` at baseline freeze commit `6d3f5bb`.
- Results: `baseline_results/`.
- Historical regression oracle: `data2_caisr20_results/`.
- Canonical DATA2 LOSO Macro / Worst AC: 0.6801306350 / 0.6166620962.

## Active

- CAISR20 baseline and its `evaluate_model.py`, `feature_scaling.py`, and
  `feat_input/feature_rules_v1.json` dependencies.
- `per_epoch_features/`, `extract_data2_timegrid_v2.py`,
  `channel_table.csv`, and `helper_code.py` as the frozen extractor reference.
- Official Challenge entry points and dependencies: `team_code.py`,
  `train_model.py`, `run_model.py`, `train_lstm.py`, static/pooled logistic
  modules, requirements, Dockerfile, and README.
- Local DATA2 cache `npz_data2/` remains ignored and unchanged.

## Archived

The P0-P6, legacy LSTM, DATA2 LSTM, 1103 bridge, site-domain, and
Stage/Event temporal-probe trees were removed from the active tree. They remain
recoverable from Git history and the cleanup recovery tag.

## Deleted local

- Old 1103 NPZ caches and smoke cache.
- Checkpoints, bytecode, repeated logs, failed runs, and temporary PID/timing
  files.
- Duplicate P5/H100 and interrupted P6000 result directories.

## Deferred

- `eeg_qc/` and ignored `docs/` were retained for review.
- The uncommitted `output/data2_nextstep/` audit, its runner, and minimal
  low-cost dependencies/cache were retained because they are not yet protected
  by Git history.
- The local official split directory remains ignored and retained.

## Recovery

- Tag: `pre-hierarchical-cleanup-20260919`
- Tag commit: `754ba8687efcdab15495fa0e0e20bcd6978afcd2`

## Verification

- CAISR20 baseline historical regression: PASS; metric differences are zero.
- Baseline Macro / Worst AC: 0.6801306350 / 0.6166620962.
- Baseline and extractor compilation: PASS.
- Archived timegrid_v2 tests: PASS (4/4).
