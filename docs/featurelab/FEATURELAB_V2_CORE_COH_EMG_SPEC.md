# FeatureLab v2 CORE_COH_EMG Specification

## Scope

`CORE_COH_EMG` is the single pre-registered post-hoc V5 configuration in this run. It selects the frozen V5 rows by `source_block`, retaining `old807` (123), `eeg_coherence_stage` (90), and `emg_stage` (72), for 285 ordered columns. Only the added `caisr_stage` (35) and `resp_stage` (56) blocks are removed; historical CAISR/respiration columns inside old807 remain.

The selection is represented by `audit/featurelab_v2_core_coh_emg_manifest.json`, whose feature and unique-column hashes bind it to the frozen 1464-row view manifest. No feature extraction, candidate-pool rebuild, Rank256 rebuild, EDF read, or existing experiment retraining is part of this configuration.

## Training and fusion contract

The frozen Rank256 V5 units are projected by exact row identity. Values, NaN placement, masks, row order, and feature order are checked against the full V5 arrays. Seeds are 42--56, training sites are S0001 and I0006, and the three frozen model families are LR, HGB, and LambdaRank. LR median adapters are fit on each selected seed subset; HGB and LambdaRank retain native NaN handling.

The run produces 90 V5 model units and 270 holdout prediction units. Existing sample/query identities are checked for all 30 train-site/seed pairs. Models are checkpoint-reloaded and prediction resume validates payload digests without refitting or re-predicting.

After all new units are complete, each of six holdout/protocol groups is fused from 180 frozen old V1--V4 columns and 45 CORE V5 columns. Each raw column is average-tie percentile-ranked within its holdout queue, then the 225 ranks are equally averaged. No E2 or LOO V5 prediction is mixed into the CORE fusion.

## Fixed comparisons

The four fixed cells are: CORE (no added CAISR/Resp), `LOO_NO_RESP` (CORE + CAISR), `LOO_NO_CAISR` (CORE + Resp), and E2 (CORE + both). The only reported contrasts are CORE--E2, CORE--E0, CORE--LOO_NO_CAISR, and CORE--LOO_NO_RESP. The CAISR and Resp conditional effects and the descriptive 2x2 interaction are calculated algebraically and checked to 1e-12.

Seed stability uses paired seeds 42--56 and six scopes. Record-level paired bootstrap uses seed 20261023, 2000 replicates, and one shared multiplicity draw per site/protocol for all four cells and the interaction. Invalid replicates remain invalid and are never encoded as zero.

## Privacy and boundaries

Public outputs contain only aggregate site/protocol metrics, seed summaries, and bootstrap summaries. Patient/record identities, labels, ages, raw predictions, bootstrap multiplicities, and private paths remain outside Git. This is an exploratory fixed-combination diagnostic, not a deployment or feature-selection decision. Batch2 remains audit-only; no migration to Branch A is authorized.
