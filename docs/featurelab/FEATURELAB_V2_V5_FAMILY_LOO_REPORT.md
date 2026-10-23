# FeatureLab v2 V5 Family Leave-One-Out Report

## Status

This report covers the four pre-specified V5 family leave-one-out analyses.
It contains aggregate results only. Patient/record identifiers, labels, ages,
raw predictions, bootstrap multiplicities, and private filesystem locations are
excluded. The private model, prediction, checkpoint, and diagnostic units are
kept outside Git.

| Contract | Result |
|---|---:|
| LOO configurations | 4 / 4 |
| Configuration widths | 286, 341, 304, 320 |
| New model units | 360 / 360 |
| New prediction units | 1,080 / 1,080 |
| Frozen sample/query matches | 30 / 30 per configuration |
| Checkpoint replay maximum absolute difference | 0.0 |
| Frozen reference entries | 5,400 |
| Fused groups | 24 |
| Bootstrap replicates | 2,000 per comparison |
| Invalid bootstrap replicates | 0 in all published rows |

The four configurations are selected from the audited V5 view manifest by
feature identity and unique-column index. No dimension-based fallback is used.
The 123-column old807 V5 prefix and all repeated-feature identities are
preserved; only the named family block is removed.

## Configuration contract

| Configuration | Removed family | Width |
|---|---|---:|
| `LOO_NO_COHERENCE` | EEG coherence stage | 286 |
| `LOO_NO_CAISR` | CAISR stage | 341 |
| `LOO_NO_EMG` | EMG stage | 304 |
| `LOO_NO_RESP` | Resp stage | 320 |

All new models use the frozen V5 rank-256 fold arrays, the approved model
factories, and seeds 42--56. Existing V1--V4 models, E0/E2 references, feature
extraction, candidate-pool construction, Rank256 construction, feature
selection, and fusion-weight tuning were not rerun.

## Aggregate outputs

The following files are the complete public aggregate outputs:

- `audit/featurelab_v2_v5_family_loo_site_scores.csv`: 24 site/protocol rows.
- `audit/featurelab_v2_v5_family_loo_protocol_summary.csv`: 8 protocol rows.
- `audit/featurelab_v2_v5_family_loo_seed_stability.csv`: 96 seed-scope rows.
- `audit/featurelab_v2_v5_family_loo_bootstrap.csv`: 96 paired-bootstrap rows.

The site file reports counts and official aggregate metrics per configuration,
protocol, and holdout site. The protocol file reports macro, worst-site, and
small-site aggregates together with deltas against the frozen E0 and E2
references. The seed file reports only seed-level counts and distribution
summaries. The bootstrap file reports fixed-seed interval summaries only.

All four files passed row-count and privacy-field checks. The bootstrap seed is
`20261023`; paired LOO/baseline multiplicities are generated privately and are
not serialized publicly.

## Reproducibility and protection checks

- Existing frozen seed-stability, bootstrap, and view-replacement public files
  retained their pre-existing SHA256 digests.
- All 360 model units and 1,080 prediction units were complete before target
  evaluation and aggregate scoring.
- A repeated cache read uses the completed private units without refitting or
  re-predicting.
- No EDF, raw waveform, candidate pool, Rank256 unit, or Branch A artifact was
  modified by this analysis.
- The full FeatureLab extensions unittest suite passed (211 tests).

## Scope boundary

This is a fixed family-ablation diagnostic, not a replacement model input
manifest. It does not authorize a new experiment, feature migration, full
re-extraction, training, or LOSO run.
