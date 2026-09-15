# NEW1103 timegrid_v2 bridge QC

- Exact coverage: 1103 NEW NPZ files matched one-to-one to the frozen P5 manifest.
- Patient IDs, labels, sites, and raw ages match for all records.
- Full streaming validation passed for 483/12/196 dimensions, metadata alignment, and finite core arrays.
- Extraction version is uniformly `timegrid_v2.0.0`.
- Extraction Git SHA values: `69e2aecef497f879b6a9e9c1953474191b8e6d91, 9eb1b8dc93e2d5905ae544a462f2dce71b1e1fe1`; exact source hashes are in `extraction_versions.json`.
- Recovery summary: 902 existing + 201 generated + 0 failed.

## Site counts

| site | n | patients | positives | negatives | eligible_ac_pairs_gap2 |
|---|---|---|---|---|---|
| I0002 | 54 | 54 | 8 | 46 | 48 |
| I0006 | 192 | 192 | 20 | 172 | 441 |
| S0001 | 857 | 857 | 56 | 801 | 5124 |
