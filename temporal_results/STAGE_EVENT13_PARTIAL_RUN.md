# Stage/Event13 probe archive

The DATA2 Stage/Event13 temporal study was archived before workspace cleanup. The run used commit `c2dbe4f` on a Medex H100 snapshot and completed only the I0002 holdout orderless and shuffled arms.

## Preserved results

| Arm | Holdout | AC-AUROC | AUROC | AUPRC | Best epoch | Validation site-macro AC |
|---|---|---:|---:|---:|---:|---:|
| O | I0002 | 0.721044 | 0.733304 | 0.361576 | N/A | N/A |
| S | I0002 | 0.747145 | 0.756303 | 0.419648 | 8 | 0.661744 |

The shuffled arm ran for 13 epochs and stopped under the frozen early-stopping rule. The remote process then failed while writing fold provenance because the Medex snapshot intentionally omitted `.git`. It did not start the real-order arm or the remaining LOSO folds, so these values are incomplete diagnostics and not a completed T1 result.

The full T0 audit is preserved in `t0_stage_event_audit.json`: all 6,600 records passed exact reconstruction of the 11 hard-stage-derived CAISR features, with no sampling-frequency errors. The temporal unit suite passed 10 tests before launch. H100 runtime profiling and the later provenance fix remain available in Git history.

Checkpoints, redundant inner-split manifests, failed-run binaries, and repeated logs are intentionally not archived here.
