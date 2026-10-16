# SIREN Branch A v1 Code Snapshot Report

## Status

`V1_CODE_SNAPSHOT_STATUS = COMPLETE`

- Source branch: `research/siren-a-stagepool-v1`
- Source commit: `b2675c90db7df4bf9f1d5b309be56a1fd1573f73`
- Snapshot path: `/database/home/gaohaojie/workspace/python-example-2026-siren-a-v1-code`
- Exported source files: 125
- Exported source bytes: 1,882,638
- Manifest SHA256: `bbda39347974c9bff8830f1d1407e085ee4cb2b6e8c1b2030e458d89715ca592`

## Contents and verification

The snapshot contains 67 source files, 30 tests, 11 frozen specifications, 15 documentation files,
one configuration file and one license file. It preserves the repository-relative paths for the
G1–G5-C Branch A implementation, per-epoch extraction source, official root entry points and their
repository-local Python imports, the frozen feature contract, run specifications, channel rules and
dependency/source notices.

Every exported file was read from the fixed Git commit rather than copied from the mutable working
tree. The exporter verified each byte sequence against the corresponding Git object, checked all
Python files with `ast.parse`, and confirmed that the snapshot contains no symbolic links or files
hard-linked to the current worktree. A second invocation returned `VERIFY_AND_REUSE` without changing
the snapshot.

The manifest is published at
`reproduction/siren_a/audit/v1_code_snapshot_manifest.csv`; the snapshot-local provenance files are
under `_snapshot/`.

## Exclusions and portability boundary

No EDF, NPZ, NPY, pickle, joblib, checkpoint, model weight, patient table, private identity map,
prediction, cache, log or temporary build artifact was copied. Third-party packages were recorded by
version but their site-packages were not copied.

This is a `CODE_SNAPSHOT`, not a data backup or a validated portable package. Original absolute paths,
external data/cache/model dependencies and historical provenance checks remain unchanged. No migration
or portability refactor was attempted, and subsequent research continues in the main experiment
worktree.

`V1_CODE_SNAPSHOT_INCLUDES_DATA = NO`
`V1_PORTABILITY_REFACTOR = NOT_STARTED`
