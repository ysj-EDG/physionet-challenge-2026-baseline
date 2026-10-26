# H100 Checkpoint Backup Design

## Intent

Protect the formal H100 E1 run from losing completed checkpoints when a remote
run fails before medex executes its final `rsync_back`. The backup mechanism is
an independent local monitor and must not change Matcha training behavior.

## Goals

- Discover the active medex task for the current Matcha workspace and
  `run_h100_e1.py` through medex's running-task registry.
- Poll the remote H100 run directory while the task is active.
- Pull each completed `checkpoint_epoch_XX.pt` once it exists, plus the current
  `checkpoint_last.pt` and `history.json`.
- Write downloads through temporary local files and atomically rename them so a
  local checkpoint is never mistaken for a complete file after interruption.
- Keep a local backup log and remain safe to run in tmux alongside medex.

## Non-goals and constraints

- Do not modify the encoder, data pipeline, training loop, or model outputs.
- Do not upload or duplicate the prepared cache.
- Do not store credentials in the script; reuse the installed medex connection
  configuration and its SSH password handling.
- Do not delete remote or local checkpoints.
- The monitor is best-effort: the normal medex `rsync_back` remains the final
  complete-result transfer.

## Approaches

1. Independent local monitor using medex's connection and running-task APIs
   (selected). This leaves training untouched, can pull each checkpoint during
   the run, and can be restarted independently.
2. Add remote checkpoint-copy logic to `run_h100_e1.py`. This would create
   extra remote copies and alter the execution wrapper, so it is unnecessary
   for the requested protection.
3. Only expand `rsync_back`. This cannot protect against a run that fails or is
   interrupted before medex reaches its final transfer step.

## Design

The monitor will be a standalone script under the Large6600 artifact workspace,
with a configurable local destination, poll interval, remote task selectors,
and maximum idle polls. It will import the installed medex runner only for
configuration, running-task discovery, SSH command construction, and password
handling. It will never print the password or embed it in a command log.

For each poll it will:

1. Select the exact active task matching the Matcha workspace, `eeg_env`, and
   `run_h100_e1.py`; fail closed if there are zero or multiple matches.
2. List only final checkpoint/history files beneath the remote E1 output
   directory. Temporary files such as `*.pt.tmp` are ignored.
3. Compare remote size with the local backup and transfer new/changed files to
   `<name>.partial`, then atomically replace the destination after a successful
   transfer and size check.
4. Record each successful transfer and continue until the medex task is gone;
   after task disappearance, perform one final discovery/transfer pass and exit
   without removing any backup.

## Verification

- Unit tests cover exact task selection, rejection of ambiguous/missing tasks,
  filtering of temporary checkpoint files, no-op behavior for an already
  complete local file, and atomic replacement after a successful transfer.
- A dry-run against the current H100 task will verify that the monitor resolves
  suffix `1ca5ba` and the remote run directory without downloading the cache.
- The live monitor will be started in its own tmux session and its log will show
  one transfer per newly completed checkpoint.
