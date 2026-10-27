# SIREN + Matcha B Cross-Fold Weight Selection v1

This CPU-only analysis consumes only the previous frozen patient-level rank payloads. No model, feature, embedding, prediction, reranking, or calibration computation was run.

## Frozen-input gates

- Previous payload SHA256 and manifest: PASS.
- A, B, AB60 (`wB=0.4`), and AB50 (`wB=0.5`) replay: PASS at maximum AC error `<= 1e-12`.
- Target-fold labels excluded from observed and nested-bootstrap weight selection: PASS.
- Candidate grid fixed before evaluation: `0.2, 0.3, 0.4, 0.5, 0.6, 0.7`.

## Cross-fold selected weights

| Target | Selected wB | Selection folds | Selection score | Target AC |
|---|---:|---|---:|---:|
| I0002 | 0.6 | I0006 + S0001 | 0.715776966092 | 0.760339342524 |
| I0006 | 0.6 | I0002 + S0001 | 0.721972994906 | 0.747947284896 |
| S0001 | 0.7 | I0002 + I0006 | 0.754880532885 | 0.682013382329 |

## Primary route summary

| Route | I0002 | I0006 | S0001 | Macro | Worst | Small-site mean |
|---|---:|---:|---:|---:|---:|---:|
| A_FINAL | 0.742311770944 | 0.704547022701 | 0.661849393956 | 0.702902729200 | 0.661849393956 | 0.723429396822 |
| B_MATCHA_M2 | 0.748144220573 | 0.739322431519 | 0.668041463281 | 0.718502705124 | 0.668041463281 | 0.743733326046 |
| AB60 | 0.759278897137 | 0.740805906300 | 0.682901881204 | 0.727662228214 | 0.682901881204 | 0.750042401718 |
| AB50 | 0.758218451750 | 0.745773821845 | 0.684056381287 | 0.729349551627 | 0.684056381287 | 0.751996136797 |
| AB_CROSSFOLD | 0.760339342524 | 0.747947284896 | 0.682013382329 | 0.730100003250 | 0.682013382329 | 0.754143313710 |

## Selection stability

| Target | wB | Selected count | Fraction | Observed selected |
|---|---:|---:|---:|---:|
| I0002 | 0.2 | 5 | 0.002500 | False |
| I0002 | 0.3 | 2 | 0.001000 | False |
| I0002 | 0.4 | 132 | 0.066000 | False |
| I0002 | 0.5 | 515 | 0.257500 | False |
| I0002 | 0.6 | 646 | 0.323000 | True |
| I0002 | 0.7 | 700 | 0.350000 | False |
| I0006 | 0.2 | 154 | 0.077000 | False |
| I0006 | 0.3 | 345 | 0.172500 | False |
| I0006 | 0.4 | 299 | 0.149500 | False |
| I0006 | 0.5 | 146 | 0.073000 | False |
| I0006 | 0.6 | 313 | 0.156500 | True |
| I0006 | 0.7 | 743 | 0.371500 | False |
| S0001 | 0.2 | 100 | 0.050000 | False |
| S0001 | 0.3 | 138 | 0.069000 | False |
| S0001 | 0.4 | 206 | 0.103000 | False |
| S0001 | 0.5 | 144 | 0.072000 | False |
| S0001 | 0.6 | 356 | 0.178000 | False |
| S0001 | 0.7 | 1056 | 0.528000 | True |

## Nested bootstrap comparisons

The primary uncertainty analysis is `NESTED_CROSSFOLD_BOOTSTRAP`; `FIXED_SELECTED_BOOTSTRAP` is auxiliary and keeps the observed selected weights fixed.

| Comparison | Scope | Observed delta | 95% CI | P(delta>0) |
|---|---|---:|---:|---:|
| AB_CROSSFOLD_MINUS_A | I0002 | 0.01802757 | [-0.03503763, 0.07500183] | 0.7565 |
| AB_CROSSFOLD_MINUS_A | I0006 | 0.04340026 | [0.01041107, 0.07463121] | 0.9940 |
| AB_CROSSFOLD_MINUS_A | S0001 | 0.02016399 | [-0.00262513, 0.04225745] | 0.9605 |
| AB_CROSSFOLD_MINUS_A | Macro | 0.02719727 | [0.00218414, 0.05201073] | 0.9835 |
| AB_CROSSFOLD_MINUS_A | Worst-site | 0.01802757 | [-0.03503763, 0.03484018] | 0.7175 |
| AB_CROSSFOLD_MINUS_A | Small-site mean | 0.03071392 | [-0.00379133, 0.06698840] | 0.9530 |
| AB_CROSSFOLD_MINUS_B | I0002 | 0.01219512 | [-0.03007069, 0.04878910] | 0.7330 |
| AB_CROSSFOLD_MINUS_B | I0006 | 0.00862485 | [-0.03137822, 0.02775273] | 0.6745 |
| AB_CROSSFOLD_MINUS_B | S0001 | 0.01397192 | [-0.00469422, 0.03101641] | 0.9555 |
| AB_CROSSFOLD_MINUS_B | Macro | 0.01159730 | [-0.00436663, 0.02588068] | 0.9120 |
| AB_CROSSFOLD_MINUS_B | Worst-site | 0.00862485 | [-0.03916295, 0.01541039] | 0.4200 |
| AB_CROSSFOLD_MINUS_B | Small-site mean | 0.01040999 | [-0.01402719, 0.02937352] | 0.7575 |
| AB_CROSSFOLD_MINUS_AB60 | I0002 | 0.00106045 | [-0.02393044, 0.02901774] | 0.4895 |
| AB_CROSSFOLD_MINUS_AB60 | I0006 | 0.00714138 | [-0.01512758, 0.02151338] | 0.4965 |
| AB_CROSSFOLD_MINUS_AB60 | S0001 | -0.00088850 | [-0.01242245, 0.00932768] | 0.3865 |
| AB_CROSSFOLD_MINUS_AB60 | Macro | 0.00243778 | [-0.01138879, 0.01497310] | 0.4975 |
| AB_CROSSFOLD_MINUS_AB60 | Worst-site | -0.00088850 | [-0.02410585, 0.00668916] | 0.1990 |
| AB_CROSSFOLD_MINUS_AB60 | Small-site mean | 0.00410091 | [-0.01622387, 0.02183988] | 0.5155 |
| AB_CROSSFOLD_MINUS_AB50 | I0002 | 0.00212089 | [-0.01405287, 0.02036100] | 0.4515 |
| AB_CROSSFOLD_MINUS_AB50 | I0006 | 0.00217346 | [-0.02300101, 0.01125800] | 0.3750 |
| AB_CROSSFOLD_MINUS_AB50 | S0001 | -0.00204300 | [-0.01177461, 0.00509483] | 0.2785 |
| AB_CROSSFOLD_MINUS_AB50 | Macro | 0.00075045 | [-0.01106935, 0.00946972] | 0.3825 |
| AB_CROSSFOLD_MINUS_AB50 | Worst-site | -0.00204300 | [-0.02316824, 0.00304564] | 0.1250 |
| AB_CROSSFOLD_MINUS_AB50 | Small-site mean | 0.00214718 | [-0.01449479, 0.01398528] | 0.4310 |

## Interpretation

- Selected weights: I0002=`0.6`, I0006=`0.6`, S0001=`0.7`; they are not identical.
- Bootstrap selection is dispersed: the largest candidate frequency is `0.528`; I0002, I0006, and S0001 have modal weights 0.7, 0.7, and 0.7, respectively, but no target is concentrated near 1.0.
- AB_CROSSFOLD Macro AC is `0.730100003250`, above AB60 by `0.002437775036` and above AB50 by `0.000750451622`; its Worst-site AC `0.682013382329` is below AB60 `0.682901881204` and AB50 `0.684056381287`.
- The nested-bootstrap Macro intervals for AB_CROSSFOLD minus AB60, AB50, and B cross zero, so the point-estimate gains do not establish a reliable improvement.
- Recommendation: retain the simpler fixed `AB50` route as the current formal fusion rule. Cross-fold selection is not promoted because its selected weights differ by target, selection frequencies are dispersed, and its apparent Macro gain is not supported as a clear nested-bootstrap improvement.
