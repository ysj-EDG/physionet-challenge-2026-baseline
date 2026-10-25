# CORE + CAISR20 — Fixed 50/50 Prediction-Level Fusion

## Scope and freeze

This document defines the single authorized post-hoc route diagnostic `CORE_CAISR_FUSION_50_50` on branch `research/featurelab-extractors-v2`. It consumes frozen prediction payloads only. It does not train, refit, extract features, rebuild Rank256, recalibrate, search weights, or modify Branch A.

The parent routes are `CORE_COH_EMG_INDUCTIVE` and `CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL`. Parent files and private payloads are read-only. The implementation was based on commit `de08a87c0a4affaabb3a6072d7e0978a7c367309`; Branch A remains at `7b3e5f8ac63096199969c75eb99a646d1be2cae8`.

## Formula

For each holdout record, after each parent has applied its own frozen within-route rank fusion:

```text
FUSION_FINAL = 0.5 * CORE_FINAL + 0.5 * CAISR_FINAL
```

The result is float64 and is passed directly to the official AC-AUROC (age gap 2), AUROC, and AUPRC scorer. No post-fusion rank, percentile, clipping, quantization, normalization, standardization, calibration, or thresholding is permitted. The weights are pre-registered and symmetric; site-, seed-, model-, or data-dependent alternatives are outside this task.

## Identity gates

Before scoring, each of the three holdouts must have exact parent index arrays: same length, integer dtype, no duplicates, identical order, and finite scores. The final gate is 3/3. The seed gate is 15 seeds (42–56) × 3 holdouts = 45/45. No sorting or fuzzy record matching is allowed.

Parent final metrics are replayed from the frozen vectors before fusion and must differ from the public parent AC values by at most `1e-12`.

## Seed diagnostic

For each seed and holdout, CORE uses the frozen 5-view × 3-model (15-column) average-tie rank mean. CAISR uses that seed's raw decision function converted to an average-tie percentile rank. Seed fusion is the same arithmetic 0.5/0.5 operation with no second rank transform. Seed results are stability diagnostics only; they are not substituted for the final 225-column CORE route.

## Paired bootstrap

The bootstrap uses record-level multiplicities, seed `20261023`, and 2000 replicates. Within each site, one multiplicity matrix is shared by CORE, CAISR20, and FUSION. Route aggregates are formed first; comparisons are then `scope(FUSION) − scope(parent)`. In particular, Worst-site is `min(FUSION site AC) − min(parent site AC)`, never the minimum of site-level deltas.

## Artifacts

Aggregate, non-patient-level artifacts are written under `audit/`:

- `core_caisr_fusion_50_50_manifest.json`
- `core_caisr_fusion_50_50_site_scores.csv`
- `core_caisr_fusion_50_50_summary.csv`
- `core_caisr_fusion_50_50_seed_stability.csv`
- `core_caisr_fusion_50_50_bootstrap.csv`

Record-level score vectors and bootstrap arrays are private only:
`/database/home/gaohaojie/workspace/feature-cache-extension/core_caisr_fusion_50_50_v1`. Each final holdout unit is atomically published with a payload digest and no-clobber verification. Public files contain only aggregate site metrics and diagnostic summaries; no record identifiers, patient scores, labels, ages, or per-record support are published.

## Explicit non-goals

No candidate is selected for deployment. No additional weight, stacking, meta-classifier, calibration, threshold search, feature search, Batch2 implementation, model training, full extraction, Rank256 rebuild, or Branch A migration is started by this analysis.
