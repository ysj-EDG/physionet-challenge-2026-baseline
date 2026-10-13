# G2-A Implementation Plan

Each task uses RED → minimal GREEN → focused regression.

1. **NPZ loader and schema validation** — test version, shape, missing field, JSON safety, and absence of label access; implement in `g2a_io.py`.
2. **Frozen Manifest parser** — test SHA lock, required columns, dimensions, unique IDs, positions, and malformed rejection.
3. **EEG and Spatial pooling** — independently test stage masks, lead masks, NREM, 4/5 support, linear IQR, six/twelve indices, and same-epoch operations; implement in `g2a_features.py`.
4. **CAISR, HRV, EMG, Resp, and Demo5** — test family-specific validity, legal zeros, entropy provenance, windows versus epochs, and demographic sentinels.
5. **Unique pool and five views** — test 807/827 order recovery, repeated Demo5 identity, NaN/mask/support invariants, and no all-valid/drop condition.
6. **Cache writer/reader** — test atomic publication, no overwrite, `allow_pickle=False` round-trip, fingerprints, and source immutability.
7. **Deterministic selector** — test seed stability, six-per-site base sample, uniqueness, 30-record hard cap, and no full execution option.
8. **Real smoke and QC** — enumerate 6600 names, select 18 records, read sequentially, write private caches, replay, aggregate family and entropy QC, and preserve anonymous public output.
9. **Report and Git** — generate aggregate report, run G1-A/G1-B/G2-A tests and syntax checks, verify Frozen/source hashes and file scope, explicitly stage eight approved files, commit, push, and stop.

Self-check: this plan changes no G1 file, adds no feature formula, has no label path, has no full-cache mode, caps content reads at 80 and outputs at 30, and writes real record artifacts only outside Git.
