# CORE_NO_CAISR20 Structural Ablation

This route is a structural, route-level ablation of the frozen `CORE_COH_EMG`
V5 view.  Exactly the twenty historical CAISR20 feature IDs are removed from
the ordered 285-column V5 manifest; the retained V5 has 265 columns.  V1--V4
remain byte-for-byte frozen.  The final route therefore has 180 frozen V1--V4
prediction columns and 45 newly fitted V5 columns (15 seeds × 3 models), and
uses the same within-holdout mean-tie-rank rule over all 225 columns.

Only the new V5 units are fitted.  The three holdouts use the frozen inductive
protocol and the existing site sample identities.  No candidate-pool rebuild,
Rank256 rebuild, raw-signal extraction, transductive prediction, or parent
route rewrite is part of this ablation.

`DEDUP_FUSION_50_50` is defined after route finalization as

```text
0.5 * CORE_NO_CAISR20_FINAL + 0.5 * CAISR20_FINAL
```

where the vectors are paired on the exact frozen test-record order and are not
post-fusion reranked.

The public artifacts contain only route/site aggregates, seed summaries and
bootstrap summaries.  Record-level vectors, identities and model metadata are
private under `core_no_caisr20_v1`.

## Frozen contracts

| item | contract |
|---|---|
| V5 old/new width | 285 → 265 |
| deleted IDs | canonical CAISR20 20-ID prefix, by ID and order |
| final columns | 180 frozen V1–V4 + 45 new V5 = 225 |
| models | S0001/I0006 × seeds 42–56 × lr/hgb/lambdarank = 90 |
| predictions | 3 holdouts × 15 seeds × 3 models = 135 inductive units |
| bootstrap | seed 20261023, 2000 shared record multiplicities per site |

The run does not claim raw-source independence: the route uses frozen Rank256
inputs.  Parent CORE, CAISR20 and current-fusion artifacts are read-only.
