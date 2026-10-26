# SIREN + Matcha B prediction-level fusion report

This report uses only two frozen prediction branches. No model, feature, embedding, or raw-signal computation was run.

## Gates

- A parent replay: PASS; max AC difference: `0`
- B parent replay: PASS; historical M2 inner-1-to-4 Macro AC: `0.71850270512405634`
- Exact identity bridge and A-order canonicalization: PASS; 6,600 rows retained.
- Rank metric invariance for A and B: PASS; no post-fusion rerank.
- Parent mutation check: PASS.

## Primary route summary

| Route | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site | Small-site mean |
|---|---:|---:|---:|---:|---:|---:|
| A_FINAL | 0.742311770944 | 0.704547022701 | 0.661849393956 | 0.702902729200 | 0.661849393956 | 0.723429396822 |
| B_MATCHA_M2 | 0.748144220573 | 0.739322431519 | 0.668041463281 | 0.718502705124 | 0.668041463281 | 0.743733326046 |
| AB_SIREN_60_40 | 0.759278897137 | 0.740805906300 | 0.682901881204 | 0.727662228214 | 0.682901881204 | 0.750042401718 |
| AB_EQUAL_50_50 | 0.758218451750 | 0.745773821845 | 0.684056381287 | 0.729349551627 | 0.684056381287 | 0.751996136797 |

## A/B correlation

| Site | Spearman(A,B) | Spearman(rA,rB) | Pearson(rA,rB) |
|---|---:|---:|---:|
| I0002 | 0.676483915361 | 0.676483915361 | 0.676483915361 |
| I0006 | 0.638771975460 | 0.638771975460 | 0.638771975460 |
| S0001 | 0.503913645299 | 0.503913645299 | 0.503913645299 |

## Pair complementarity

| Site | Legal pairs | A-only correct | B-only correct | Both correct | Both wrong | B-correct/A-wrong fraction | A-correct/B-wrong fraction |
|---|---:|---:|---:|---:|---:|---:|---:|
| I0002 | 1886 | 203 | 214 | 1197 | 272 | 0.113468 | 0.107635 |
| I0006 | 14493 | 1473 | 1977 | 8738 | 2305 | 0.136411 | 0.101635 |
| S0001 | 182330 | 26332 | 27461 | 94343 | 34194 | 0.150612 | 0.144419 |

## Paired bootstrap

| Comparison | Scope | Observed delta | 95% CI | P(delta>0) |
|---|---|---:|---:|---:|
| AB60_MINUS_A | I0002 | 0.01696713 | [-0.01824824, 0.05496394] | 0.8125 |
| AB60_MINUS_A | I0006 | 0.03625888 | [0.01360182, 0.05841575] | 0.9990 |
| AB60_MINUS_A | S0001 | 0.02105249 | [0.00650649, 0.03454227] | 0.9980 |
| AB60_MINUS_A | Macro | 0.02475950 | [0.00965990, 0.03981375] | 0.9995 |
| AB60_MINUS_A | Worst-site | 0.01696713 | [-0.01824824, 0.02857360] | 0.8100 |
| AB60_MINUS_A | Small-site mean | 0.02661300 | [0.00483388, 0.04847440] | 0.9930 |
| AB60_MINUS_B | I0002 | 0.01113468 | [-0.04822430, 0.06242521] | 0.6395 |
| AB60_MINUS_B | I0006 | 0.00148347 | [-0.02975982, 0.03136498] | 0.5395 |
| AB60_MINUS_B | S0001 | 0.01486042 | [-0.00614510, 0.03767172] | 0.9080 |
| AB60_MINUS_B | Macro | 0.00915952 | [-0.01379861, 0.03023019] | 0.7870 |
| AB60_MINUS_B | Worst-site | 0.00148347 | [-0.04861638, 0.01737283] | 0.3170 |
| AB60_MINUS_B | Small-site mean | 0.00630908 | [-0.02663062, 0.03500829] | 0.6490 |
| AB50_MINUS_A | I0002 | 0.01590668 | [-0.02951667, 0.06275060] | 0.7615 |
| AB50_MINUS_A | I0006 | 0.04122680 | [0.01342432, 0.06966275] | 0.9990 |
| AB50_MINUS_A | S0001 | 0.02220699 | [0.00348936, 0.03951036] | 0.9915 |
| AB50_MINUS_A | Macro | 0.02644682 | [0.00748850, 0.04516527] | 0.9960 |
| AB50_MINUS_A | Worst-site | 0.01590668 | [-0.02951667, 0.03108629] | 0.7540 |
| AB50_MINUS_A | Small-site mean | 0.02856674 | [0.00148436, 0.05689097] | 0.9815 |
| AB50_MINUS_B | I0002 | 0.01007423 | [-0.03935612, 0.05252261] | 0.6590 |
| AB50_MINUS_B | I0006 | 0.00645139 | [-0.01792874, 0.03170399] | 0.6985 |
| AB50_MINUS_B | S0001 | 0.01601492 | [-0.00169352, 0.03530595] | 0.9615 |
| AB50_MINUS_B | Macro | 0.01084685 | [-0.00787249, 0.02834484] | 0.8815 |
| AB50_MINUS_B | Worst-site | 0.00645139 | [-0.03935612, 0.01812724] | 0.4475 |
| AB50_MINUS_B | Small-site mean | 0.00826281 | [-0.01799859, 0.03221893] | 0.7365 |

The primary route is `AB_SIREN_60_40`; `AB_EQUAL_50_50` is a fixed control, not a weight search.

Record-level vectors and identity fields are stored only in the private artifact root and are not included in this branch.
