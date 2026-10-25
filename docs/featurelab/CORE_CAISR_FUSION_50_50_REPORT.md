# CORE + CAISR20 — Fixed 50/50 Fusion Report

## Status

`FIXED_ROUTE_FUSION_STATUS = COMPLETE`

The authorized fixed route was evaluated once from frozen parent prediction payloads. No model was refit, no prediction was recomputed, no feature was re-extracted, and no parent artifact was modified. The first publish created the three private final units; the strict repeat verified all three as `SKIP_VERIFIED`.

Verification status: the 11 focused fusion tests passed; the frozen G1B contract suite passed (13/13). The bounded full extensions suite was started once with a 120-second limit and did not complete before the limit (`EXTENSIONS_TEST = NOT_COMPLETED`); completed tests in its log showed no failure.

## Parent replay and identity

| Gate | Result |
|---|---|
| Parent final index/order identity | 3/3 |
| Parent seed index/order/seed identity | 45/45 |
| Parent replay maximum absolute AC difference | 0.0 (required ≤ 1e-12) |
| Arithmetic exactness | fixed float64 `(CORE + CAISR20) / 2`; no post-fusion rank |
| Fusion final groups | 3/3 |
| Bootstrap | 2000 replicates, seed 20261023, record-level paired multiplicities |

Frozen parent public SHA256 values were recorded before the run and verified unchanged afterward: `audit/caisr20_core_protocol_v2_site_scores.csv` `b58601171fc8626b899d8a9d9968c15290d93893df79720a97b985a2e9adf605`; `audit/caisr20_core_protocol_v2_summary.csv` `d70db233584dc7defea0e8d129438193a01655990bc166016ff0ba0a3d38e901`; `audit/caisr20_core_protocol_v2_seed_stability.csv` `b043c84180235c48f88adad771d245c7bb9b8ab2858a406b1a6c567953d38dd3`; `audit/caisr20_core_protocol_v2_bootstrap.csv` `350ba24b799ee22aca84f9491cbd839d2ef44beccce819bd320086736c189fb`; `audit/caisr20_core_protocol_v2_score_correlation.csv` `929061f1dda7fc50baefc9efb0cada8eb1a5761a673524f0eb3b5e31f740703b`; `audit/featurelab_v2_core_coh_emg_site_scores.csv` `a51c5666a82afcef0d807cb2db20f054a7a8148ed97df5cee57c2af8f9195a1c`; `audit/featurelab_v2_core_coh_emg_protocol_summary.csv` `1326983570c3b129ab3b784fc1a4cb4b20824ff8c07bedebece1ae2bdc059b88`; `audit/featurelab_v2_core_coh_emg_seed_stability.csv` `2d1e988b79255cdad6456a8ead3f641bbfdf25adad923ebf7881affd0f075c22`; `audit/featurelab_v2_core_coh_emg_bootstrap.csv` `0d5ac2cf41c96c3f8061f3a0b3a3d8079492ba3b8112e046739c37c3765a31eb`.

## Final site scores

All rows are aggregate site metrics; record-level score vectors remain private.

| Route | I0002 AC | I0006 AC | S0001 AC | Macro AC |
|---|---:|---:|---:|---:|
| CAISR20 | 0.7343584305 | 0.6571448285 | 0.6116437229 | 0.6677156607 |
| CORE | 0.6961823966 | 0.6786724626 | 0.6709153732 | 0.6819234108 |
| FUSION_50_50 | 0.7407211029 | 0.7042710274 | 0.6602917786 | 0.7017613030 |

The fixed fusion deltas (route aggregate first) are:

| Scope | Fusion − CORE | Fusion − CAISR20 |
|---|---:|---:|
| I0002 | +0.0445387063 | +0.0063626723 |
| I0006 | +0.0255985648 | +0.0471261989 |
| S0001 | −0.0106235946 | +0.0486480557 |
| Macro | +0.0198378922 | +0.0340456423 |
| Worst-site | −0.0106235946 | +0.0486480557 |
| Small-site mean | +0.0350686355 | +0.0267444356 |

Worst-site is `min(route site AC)` before subtraction. The fusion improves the parent Macro values in this frozen replay, while its S0001 score remains below CORE; this is a diagnostic result, not an automatic deployment decision.

## Seed stability

The public seed file has 12 aggregate rows (two fixed comparisons × six scopes), each with `n_seeds=15`, sign counts, mean, median, q25, q75, min, and max. Key results:

- Fusion−CORE Macro: mean +0.0247480; 15/15 positive.
- Fusion−CORE Worst-site: mean +0.0010958; 8 positive, 7 negative.
- Fusion−CAISR20 Macro: mean +0.0342221; 15/15 positive.

Seed diagnostics do not replace the final route point estimate.

## Paired bootstrap

The public bootstrap file has 12 rows. Every row has `valid=2000`, `invalid=0`, and the same bootstrap seed. Selected aggregate results:

- Fusion−CORE Macro: observed +0.0198379; 95% interval [−0.0055134, +0.0446383].
- Fusion−CORE Worst-site: observed −0.0106236; 95% interval [−0.0301703, +0.0615989].
- Fusion−CAISR20 Macro: observed +0.0340456; 95% interval [+0.0158850, +0.0523264].
- Fusion−CAISR20 Worst-site: observed +0.0486481; 95% interval [+0.0223242, +0.0649548].

The same site-specific multiplicity matrix is used for all three routes in each site.

## Privacy and boundaries

Public Git artifacts contain only aggregate values, route/site counts, and distribution summaries across seeds/bootstrap replicates. Patient-level scores, record IDs, labels, ages, raw prediction arrays, and private paths are excluded. Detailed final and bootstrap arrays are retained only in the private cache root.

`MODEL_REFIT = NOT_STARTED`; `PREDICTION_RECOMPUTE = NOT_STARTED`; `FEATURE_REEXTRACTION = NOT_STARTED`; `WEIGHT_SEARCH = CLOSED`; `POST_FUSION_RERANK = NOT_PERFORMED`; `RANK256_REBUILD = NOT_STARTED`; `BRANCH_A_MIGRATION = NOT_STARTED`; `BATCH2 = AUDIT_ONLY`. No automatic candidate selection follows this report.
