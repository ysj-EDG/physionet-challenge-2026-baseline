# FeatureLab v2 CORE_COH_EMG Report

## Status

The fixed CORE_COH_EMG run completed with 90/90 models and 270/270 predictions. A strict second execution returned zero model builds and zero prediction builds while validating the completed units. The public files below contain aggregates only.

| Contract | Result |
|---|---:|
| CORE width | 285 |
| old807 / coherence / EMG | 123 / 90 / 72 |
| Removed added blocks | CAISR 35 / Resp 56 |
| Rank256 rebuilds | 0 |
| Sample/query identity | 30/30 |
| New models / predictions | 90/90 / 270/270 |
| Checkpoint replay max absolute difference | 0.0 |
| Fixed fused groups | 6/6, 225 columns each |
| CORE references | 1350/1350 |
| Bootstrap | 2000 per row, invalid 0 |

## CORE aggregate result

| Protocol | Macro AC | Worst-site AC | Small-site mean AC | Δ vs E2 | Δ vs E0 |
|---|---:|---:|---:|---:|---:|
| Transductive | 0.670707 | 0.658273 | 0.676923 | +0.001696 | +0.008314 |
| Inductive | 0.681923 | 0.670915 | 0.687427 | +0.004230 | +0.010945 |

The six site-level rows are in `audit/featurelab_v2_core_coh_emg_site_scores.csv`.

## CAISR x Resp 2x2

The public 12-row file `audit/featurelab_v2_core_caisr_resp_2x2.csv` uses the fixed mapping CORE = (0,0), LOO_NO_RESP = (+CAISR), LOO_NO_CAISR = (+Resp), and E2 = (+CAISR,+Resp). The interaction is the post-hoc descriptive contrast `E2 - LOO_NO_RESP - LOO_NO_CAISR + CORE`; both equivalent conditional-effect expressions were checked within 1e-12.

At Macro scope, the interaction is -0.000591 (transductive) and +0.000796 (inductive). Conditional effects vary by site and protocol; this is not a causal interaction claim.

## Stability and bootstrap

`audit/featurelab_v2_core_coh_emg_seed_stability.csv` has 60 aggregate rows: four CORE contrasts plus the interaction across six scopes and two protocols, each paired over 15 seeds. `audit/featurelab_v2_core_coh_emg_bootstrap.csv` has 60 rows: four contrasts plus the interaction over the same scopes/protocols. Every row has 2000 valid replicates and uses the fixed seed 20261023.

For Macro scope, CORE--E2 is +0.001696 (transductive) and +0.004230 (inductive); CORE--E0 is +0.008314 and +0.010945. The bootstrap intervals and site-specific values must be read from the aggregate CSV rather than interpreted as external-test significance.

## Protection checks and scope boundary

The frozen E0/E2/LOO/view public inputs remained unchanged. No EDF, feature extraction, candidate pool, Rank256 unit, Branch A file, or prior experiment output was modified. No new family combination, Batch2 implementation, feature selection, fusion-weight tuning, or Branch A migration was started.
