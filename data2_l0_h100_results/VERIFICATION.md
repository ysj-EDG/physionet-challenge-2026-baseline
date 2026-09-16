# DATA2-L0 independent result verification

The repository's official age-conditioned AUROC implementation (`gap=2`) was rerun from each saved raw-logit CSV. All values matched `aggregate_metrics.json` to floating-point precision.

| Holdout | Checkpoint | N | Positive | Eligible pairs | AC-AUROC |
|---|---|---:|---:|---:|---:|
| I0002 | best | 319 | 52 | 1886 | 0.660657476140 |
| I0002 | epoch6 | 319 | 52 | 1886 | 0.653234358431 |
| I0006 | best | 1142 | 112 | 14493 | 0.516594217898 |
| I0006 | epoch6 | 1142 | 112 | 14493 | 0.531497964535 |
| S0001 | best | 5139 | 334 | 182330 | 0.438666154774 |
| S0001 | epoch6 | 5139 | 334 | 182330 | 0.502363845774 |

Checkpoint checks:

- Six expected checkpoints exist: best and epoch6 for each of three sites.
- Each contains a nonempty 12-tensor `state_dict`.
- Best checkpoint epochs are 1, 1, and 19; diagnostic checkpoints are epoch 6.
- Each stores `legacy_clip`, global class-balanced sampling, empirical `pos_weight`, seed 7, and `outer_holdout_used_during_training=false`.
- Raw-logit record counts exactly match 319, 1142, and 5139.
