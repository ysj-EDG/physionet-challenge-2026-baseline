# CORE_NO_CAISR20 structural ablation report

Status: completed for the authorized V5-only route comparison.  This report
contains route/site aggregates only.  Record-level scores, prediction vectors,
sample identities, model metadata and bootstrap arrays remain private under
`/database/home/gaohaojie/workspace/feature-cache-extension/core_no_caisr20_v1`.

## Execution contract

- Deleted exactly 20 canonical historical CAISR20 IDs from CORE V5: 285 → 265.
- V1–V4 were read-only; 90 new V5 models were fitted (2 train sites × 15
  seeds × 3 model types).
- 135 new V5 predictions were produced for the three holdouts using the
  inductive protocol.
- Final CORE_NO_CAISR20 width was 225 (180 frozen V1–V4 + 45 new V5).
- The structural feature-position accounting is 969 positions (113/129/349/113 frozen V1–V4 plus 265 retained V5) and 949 unique Feature IDs.
- Rank256 rebuilds, raw EDF reads, candidate-pool rebuilds and transductive
  predictions: 0.
- `DEDUP_FUSION_50_50` was computed as the exact 0.5/0.5 prediction-level
  arithmetic fusion with no post-fusion rerank.

## Route-level AC summary

The public CSV `audit/featurelab_core_no_caisr20_summary.csv` contains all
five route summaries and four registered comparisons for six scopes.  The key
macro and worst-site values are:

| route | Macro AC | Worst-site AC |
|---|---:|---:|
| CAISR20 | 0.667716 | 0.611644 |
| CORE | 0.681923 | 0.670915 |
| CORE_NO_CAISR20 | 0.679122 | 0.669303 |
| CURRENT_FUSION | 0.701761 | 0.660292 |
| DEDUP_FUSION | 0.702903 | 0.661849 |

Primary differences:

- `CORE_NO_CAISR20 − CORE`: Macro −0.002801; Worst-site −0.001612.
- `DEDUP_FUSION − CURRENT_FUSION`: Macro +0.001141; Worst-site +0.001558.

Descriptive differences are available in the same summary for
`DEDUP_FUSION − CORE` and `DEDUP_FUSION − CAISR20`.

## Stability and bootstrap

`audit/featurelab_core_no_caisr20_seed_stability.csv` contains 12 rows (two
primary comparisons × six scopes), each based on all 15 seeds.  The bootstrap
file contains the same 12 rows with seed `20261023`, 2000 shared record-level
multiplicity replicates per site, and zero invalid replicates.

For the primary macro comparisons, the bootstrap intervals were:

- `CORE_NO_CAISR20 − CORE`: −0.007231 to +0.001646.
- `DEDUP_FUSION − CURRENT_FUSION`: −0.001949 to +0.004252.

These are descriptive route diagnostics, not a claim of independent raw-signal
re-extraction.  The fixed parent routes and current-fusion artifacts were
validated read-only; current fusion replay matched its frozen score vectors.

## Artifact and privacy checks

- New subset manifest: `audit/featurelab_core_no_caisr20_manifest.json`.
- Public site rows: 15 (= five routes × three holdouts).
- Public summary rows: 54 (= 30 route summaries + 24 comparison rows).
- Public seed rows: 12; public bootstrap rows: 12.
- No patient identifiers, record-level feature values, sample identities or
  private paths are written to the public report/CSV files.
- Branch A, old candidate pools, CORE, CAISR20 and current fusion public/private
  artifacts were not modified.

No next ablation, weight search, model training or migration was started.
