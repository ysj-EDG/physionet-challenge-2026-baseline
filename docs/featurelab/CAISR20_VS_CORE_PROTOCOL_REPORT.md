# CAISR20 vs CORE — Gate Report

## Status

`ROUTE_LEVEL_COMPARISON_STATUS = BLOCKED_SOURCE_VALUE_BRIDGE`

The historical regression bridge passed, the DATA2 inventory passed, and all
30 frozen CORE sample identities matched. The route stopped before fitting a
new model because the raw CAISR source validity positions do not match the
immutable old-V5 source. This is a source-contract difference, not a model
failure; no new route score is reported.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| Historical CAISR20 seed-7 bridge | PASS | Maximum absolute decision-score difference `4.440892098500626e-16`; no convergence warnings |
| Historical archived Macro AC | PASS | `0.6801306349958572` |
| DATA2 inventory | PASS | 6600 records; I0002=319, I0006=1142, S0001=5139; positives=498 |
| CAISR20 manifest | PASS | 20 unique IDs; zero demographic columns; frozen `x_static` indices |
| CORE sample identity bridge | PASS | 30/30 selected local/global index artifacts |
| Old-V5 raw value bridge | PASS on finite overlap | 0 finite-value mismatches across all 20 features |
| Old-V5 validity/NaN bridge | FAIL | 8059 validity-position mismatches |
| New CAISR20 models | NOT STARTED | Safety stop before model fitting |
| New predictions/fusion/metrics | NOT STARTED | Safety stop before prediction generation |

The largest validity discrepancy is
`caisr_sleep_mean_stage_entropy` (old-V5 valid=7, DATA2 finite=6600). The
other 19 features also contain nonzero validity-position differences. Because
the missingness contract changes, treating this as an exact or silently
equivalent source would be unsafe.

## Frozen historical facts

The archived pure CAISR20 route is distinct from compact30 (demo10 + CAISR20).
The archived pure route site AC values are I0002=`0.7460233297985154`,
I0006=`0.6777064789898571`, S0001=`0.6166620961991992`, with Macro AC
`0.6801306349958572`. Compact30 was not retrained.

The frozen CORE inductive comparator remains Macro AC
`0.6819234107996947`; no difference is claimed because the new CAISR route did
not reach formal training and scoring.

## Privacy and immutability

This report contains only aggregate gate counts and frozen contract metadata.
It contains no patient identifiers, record-level values, raw predictions,
private paths, labels, or bootstrap multiplicities. Existing CORE, candidate
pool, old CAISR20 archive, Rank256, and Branch A artifacts were not modified.

## Required next decision

A follow-up review must decide whether to provide a documented source-validity
bridge (with an explicitly approved semantic contract) or to keep this route
blocked. No compact30 substitution, missingness relaxation, feature rewrite,
or new extraction is authorized by this report.
