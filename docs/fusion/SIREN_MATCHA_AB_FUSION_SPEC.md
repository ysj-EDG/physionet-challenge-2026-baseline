# SIREN + Matcha B fusion specification

This branch performs prediction-level fusion of two frozen parent outputs only.

- A: `DEDUP_FUSION = 0.5 * CORE_NO_CAISR20 + 0.5 * CAISR20`, from the frozen `research/featurelab-extractors-v2` ref `626ec1e9f178ae5922db09fab79955a6182e0f67`.
- B: historical Matcha M2 inner-1-to-4 LOSO, preserving the native raw-logit mean of seeds `20260806`, `20260807`, and `20260808`; E1 identity is recorded by SHA-256 in the public manifest.
- The join is exact on `(site, record identity)` and retains the frozen A order for all 6,600 rows.
- Each holdout uses the existing SIREN average-tie percentile-rank helper: `rA` and `rB`.
- Primary: `AB_SIREN_60_40 = 0.6*rA + 0.4*rB`. Control: `AB_EQUAL_50_50 = 0.5*rA + 0.5*rB`. No calibration, reranking, threshold optimization, or weight search.
- Metrics are official AC-AUROC, AUROC, AUPRC, record count, positive count, and legal age-matched pair count.
- Bootstrap is paired at record level, uses the same multiplicities for all routes within each site, `n=2000`, seed `20261023`, and reports both observed and bootstrap deltas.

Public artifacts contain aggregate metrics and hashes only. Record-level vectors, ages, labels, scores, and identity fields are kept outside Git under the private artifact root.
