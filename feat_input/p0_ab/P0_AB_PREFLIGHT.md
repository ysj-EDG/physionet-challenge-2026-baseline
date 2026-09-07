# P0 legacy_clip vs typed_v1 launch preflight

**Final gate: BLOCKED**

Generated: `2026-09-07T13:19:30.718200+00:00`
Frozen Git HEAD: `0e6e7087e17af5df3f9438d440f45b2d47b06cf7`
Parent baseline: `a4fe0b4dd914856993e67870801c9eb35b5311b2`

## Scope freeze

The worktree started at the audited commit with only the pre-existing untracked `eeg_qc/` directory. No reset was performed and no user file was overwritten. Audit artifacts added later are intentionally untracked.

The parent-to-head production diff was classified as input preprocessing plumbing only. AST/content checks confirm that model architecture and initialization, collate/ECG offset 10, batch size, optimizer/LR, balanced sampler, pos_weight, gradient clipping, scheduler, early stopping, checkpoint selection, calibration, and threshold selection are unchanged.

Frozen rule file SHA256: `b8624092b12e5a2800d7627413d6d4c791fec8c7fb77552a007e1d43342c3c90`. `build_feature_rules_v1.py` was not run, no common z clipping was added, and per-column rules were not changed.

## Raw input identity

Cache: `/database/home/gaohaojie/workspace/python-example-2026/npz_new`
Split: `/database/home/gaohaojie/workspace/python-example-2026/split`
Records hashed: `1103`
Ordered aggregate NPZ content SHA256: `9b3d473ea2ec5810715859db099ff7da930b59d7f33cb6277d47ee22002e81ad`

| split | records | manifest SHA256 | ordered ID SHA256 | fallback | label counts |
|---|---:|---|---|---:|---|
| train | 733 | `5d5be75ff4d66684fdc7f5600cf82dfaceca22cee2b3f2c7f66bcdc2c2c0261e` | `c1d9a4702980b139b2969bef688a6346cf9251542c57c22e090fbf0edb7e8773` | 0 | `{"0": 680, "1": 53}` |
| val | 158 | `9ee4e400fe389a9cad09aba9c523a93462f0edc341f413af08c48a8f74d27bcd` | `0a65f9801187a986062f56585bb8f1a5c3f8772f7c002e35d3129348a1394df0` | 0 | `{"0": 146, "1": 12}` |
| test | 158 | `bfc237e3ba13d2f19ce69e9a6eff36c0ffa8c403bb874a6935aa5cea47935c31` | `1b2a76b5893155cbb9fc0d0c4e8603e57c468b2a7050c5bc1171d92796c41150` | 0 | `{"-1": 158}` |
| external | 54 | `52c10b1b21f0f21dc23ba787dde3bb91e3424c9a3eeadf9d3b5c0c242c5748de` | `96b795184128b439cc1e77d2647a106648d6a8b94ab33c94fc5b1c96185f4e99` | 0 | `{"-1": 54}` |

Every listed NPZ has an individual byte-content SHA256 in `input_fingerprints.json`; mask content, label, raw age, sequence lengths, and ordered membership are recorded there as well. A and B point to these exact same files. Missing caches cause an immediate BLOCKED result and never trigger extraction or cache substitution.

Train/val labels are required to be exactly binary. Test/external `-1` values, if present, are explicitly not valid truth labels and must be excluded from later metric computation.

## fallback_sequence boundary

No training fallback_sequence records exist; the known fit/offline fallback omission does not affect the current typed_v1 fit. No preprocessing code was changed.

The known offline-validator omission remains documented but is not triggered by this frozen dataset. Because the training count is zero, no scaler code or old report was changed.

## Pairing checks

- PASS — `manifest_files_present`
- PASS — `no_missing_cache`
- PASS — `no_unexpected_cache`
- PASS — `no_npz_read_errors`
- PASS — `expected_total_records_1103`
- PASS — `expected_split_counts_733_158_158_54`
- PASS — `no_within_split_duplicates`
- PASS — `no_cross_split_overlap`
- PASS — `train_val_labels_binary`
- PASS — `train_fallback_absent`
- PASS — `test_external_minus_one_excluded_from_future_truth_metrics`
- PASS — `git_head_is_0e6e708`
- PASS — `git_parent_is_a4fe0b4`
- PASS — `production_worktree_matches_head`
- PASS — `frozen_constants_unchanged`
- PASS — `frozen_functions_and_model_unchanged`
- PASS — `training_strategy_nodes_unchanged`
- PASS — `rules_file_is_committed_version`
- FAIL — `training_entry_does_not_score_minus_one_external`
- PASS — `existing_15_tests_pass`
- PASS — `committed_seed7_scaler_state_matches_frozen_inputs`
- PASS — `fit_preserves_all_global_rng_states`
- PASS — `transform_preserves_all_global_rng_states`
- PASS — `legacy_matches_a4fe0b4_on_all_1103_records`
- PASS — `typed_all_1103_outputs_float32_finite`
- PASS — `same_seed_initial_model_weights_identical`
- PASS — `first_batch_diagnostic_sampling_identical`
- PASS — `first_two_epoch_sampling_identical`
- PASS — `real_device_forward_backward_pass`
- PASS — `smoke_initial_weights_identical`
- PASS — `typed_dataset_equals_inference_path`
- PASS — `raw_metric_age_preserved`

The checks ran in a standalone preflight process. They did not update optimizer weights or save a model. Formal runs start in separate fresh processes and do not execute additional sampling diagnostics.

## Launch

`run_p0_ab.sh` first re-hashes Git HEAD, production code, all manifests, the frozen rules, and all 1103 NPZ files. It refuses existing result directories, then runs A followed by B on the same visible GPU with seed 7. Only `LSTM_INPUT_PREPROCESSING` and `LSTM_MODEL_DIR` differ.

**BLOCKED: failed gates: `training_entry_does_not_score_minus_one_external`. See `pairing_checks.json` for evidence.**
