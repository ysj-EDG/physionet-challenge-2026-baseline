# H100 Checkpoint Backup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent local monitor that repeatedly pulls completed H100 E1 checkpoints while the medex run is active.

**Architecture:** A standalone Python monitor will use the installed medex runner to discover the exact active task and construct authenticated SSH/rsync commands. It will parse final remote files, transfer each file to a local partial path, verify size, and atomically rename it into the Large6600 artifact backup directory. It will never modify the training process.

**Tech Stack:** Python 3.10, standard library, installed `medex-run` module, SSH/rsync, unittest.

**Spec:** `docs/superpowers/specs/2026-10-25-h100-checkpoint-backup-design.md`

## Global Constraints

- Do not modify Matcha encoder, data pipeline, training loop, or model outputs.
- Do not upload or duplicate the prepared cache.
- Do not store credentials in the monitor; reuse medex configuration and password handling.
- Do not delete remote or local checkpoints.
- Run the monitor itself in tmux.

## Review Focus

- Multiple matching medex tasks must fail closed rather than backing up the wrong run; test in Task 1.
- Temporary `.pt.tmp` files must never be pulled; test in Task 1.
- A partial local download must not be treated as a valid checkpoint; test in Task 2.
- Repeated polling must not redownload an unchanged complete file; test in Task 2.
- Loss of the active remote task after a completed checkpoint must allow one final pull pass; verify in Task 3.

### Task 1: Pure discovery and remote listing helpers

**Files:**
- Create: `/database/home/gaohaojie/workspace/repd/Matcha/artifacts/v18_large6600/test_h100_checkpoint_monitor.py`
- Create: `/database/home/gaohaojie/workspace/repd/Matcha/artifacts/v18_large6600/h100_checkpoint_monitor.py`

**Interfaces:**
- Produces `select_active_task(tasks, workspace, script, env) -> dict`.
- Produces `parse_remote_listing(text) -> list[RemoteFile]`.
- Produces `is_final_checkpoint_name(name) -> bool`.

- [ ] **Step 1: Write failing tests** for exact task selection, ambiguous/missing task rejection, temporary-file filtering, and parsing of remote `size<TAB>mtime<TAB>name` lines.
- [ ] **Step 2: Run the focused unittest module and verify it fails because the monitor helpers do not exist.**
- [ ] **Step 3: Implement the minimal pure helpers in `h100_checkpoint_monitor.py`.**
- [ ] **Step 4: Run the focused unittest module and verify it passes.**

### Task 2: Safe atomic checkpoint transfer

**Files:**
- Modify: `/database/home/gaohaojie/workspace/repd/Matcha/artifacts/v18_large6600/h100_checkpoint_monitor.py`
- Modify: `/database/home/gaohaojie/workspace/repd/Matcha/artifacts/v18_large6600/test_h100_checkpoint_monitor.py`

**Interfaces:**
- Produces `install_downloaded_file(partial_path, destination, expected_size) -> None`.
- Produces `should_transfer(remote_file, destination) -> bool`.
- Produces `sync_remote_file(remote, remote_path, destination, expected_size, rsync_command_factory) -> bool`.

- [ ] **Step 1: Add failing tests** for incomplete local files, unchanged-file no-op, and successful atomic replacement from a partial file.
- [ ] **Step 2: Run the focused unittest module and verify the new tests fail.**
- [ ] **Step 3: Implement the minimal size-check and atomic replacement behavior; use the injected command factory so tests do not require H100 access.**
- [ ] **Step 4: Run the focused unittest module and verify it passes.**

### Task 3: Live medex polling loop and tmux launch

**Files:**
- Modify: `/database/home/gaohaojie/workspace/repd/Matcha/artifacts/v18_large6600/h100_checkpoint_monitor.py`
- Modify: `/database/home/gaohaojie/workspace/repd/Matcha/artifacts/v18_large6600/test_h100_checkpoint_monitor.py`
- Create: `/database/home/gaohaojie/workspace/repd/Matcha/artifacts/v18_large6600/h100_checkpoint_backup.log`

**Interfaces:**
- CLI accepts task selectors, destination, poll interval, and idle-poll limit.
- Uses medex `load_config`, `load_running_tasks`, `make_remote`, `ssh_capture_script`, `rsync_pull_base`, and `with_sshpass` without printing secrets.

- [ ] **Step 1: Add failing tests** for final-pass behavior when task discovery returns no task after a prior successful observation and for logging each successful new checkpoint.
- [ ] **Step 2: Run the focused unittest module and verify the new tests fail.**
- [ ] **Step 3: Implement the polling loop, remote path construction, final pass, and CLI.**
- [ ] **Step 4: Run the focused unittest module and the existing repository tests.**
- [ ] **Step 5: Run a live dry-run/status check against the current H100 task, then launch the monitor in tmux `matcha_h100_checkpoint_backup`.**
- [ ] **Step 6: Verify the monitor log and local destination contain the already completed H100 checkpoints without touching the prepared cache.**
