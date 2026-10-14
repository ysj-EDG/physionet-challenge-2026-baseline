"""G3-B label-safe model-input orchestration (never trains a model)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import shutil
import sys
import tempfile
import time
from io import StringIO
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .adapters import AdapterState, check_adapter, fit_adapter, transform_adapter
from .input_data import EXPECTED_FOLDS, VIEW_SIZES, check_cross_protocol_train, load_fold
from .label_join import audit_age_groups, check_identity_join, check_patient_overlap, load_metadata_index


SCHEMA = "g3b_inputs_v1"
MANIFEST_SHA256 = "7aef4e0462936ad0fbb3ee504a71062b0f50d82de5c54e1d70df2d13691e455a"
SPEC_SHA256 = "f9f6ae6b7fd3126569914dad9c643c17ee9fa56354c460c694f46060cb174d9a"
G2_CACHE_FINGERPRINT = "cf88b4e87f8b524258c685db696c278198dae1e6a10bd39fb63e893b2df031df"
G3_ARTIFACT_FINGERPRINT = "4a0f3956432030e4fa2fb8b51d15aa8fa5269f990d7795cca40d580970920a64"
G3_ARTIFACT_FILES = 52
G3_ARTIFACT_BYTES = 417435113
PROTOCOLS = ("transductive", "inductive")
MODELS = ("lr", "hgb", "lambdarank")
VIEWS = tuple(VIEW_SIZES)
PRIVATE_TOKENS = ("bidsfolder", "sessionid", "bdsppatientid", "record_id", "patient_id", "/database/")


@dataclass(frozen=True)
class G3BConfig:
    repo: Path
    g2_root: Path
    g3_root: Path
    g3av_root: Path
    demographics_csv: Path
    private_root: Path
    manifest_path: Path
    spec_path: Path
    public_csv: Path
    public_report: Path


def default_config() -> G3BConfig:
    repo = Path(__file__).resolve().parents[3]
    return G3BConfig(
        repo=repo,
        g2_root=Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g2b1-private/night_cache_v1"),
        g3_root=Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g3a-private/rank256_v1"),
        g3av_root=Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g3av-private/verification_v1"),
        demographics_csv=Path("/database2/physionet2026_kaggle/data2/demographics.csv"),
        private_root=Path("/database/home/gaohaojie/workspace/python-example-2026-siren-a-g3b-private/model_inputs_v1"),
        manifest_path=repo / "reproduction/siren_a/docs/G1B_FEATURE_MANIFEST_FROZEN.csv",
        spec_path=repo / "reproduction/siren_a/docs/G1B_FEATURE_SPEC_FROZEN.md",
        public_csv=repo / "reproduction/siren_a/audit/g3b_input_summary.csv",
        public_report=repo / "reproduction/siren_a/docs/G3B_INPUT_REPORT.md",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def array_sha256(array: np.ndarray) -> str:
    value = np.asarray(array)
    digest = hashlib.sha256(); digest.update(str(value.shape).encode("ascii")); digest.update(value.dtype.str.encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def atomic_json_no_clobber(path: Path, payload: object) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name); handle.write(canonical_json(payload)); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError("JSON round-trip verification failed")
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_npz_no_clobber(path: Path, **arrays: np.ndarray) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name); np.savez(handle, **arrays); handle.flush(); os.fsync(handle.fileno())
        with np.load(temporary, allow_pickle=False) as archive:
            if set(archive.files) != set(arrays):
                raise ValueError("NPZ round-trip keys failed")
            for key, expected in arrays.items():
                observed = np.asarray(archive[key]); expected_array = np.asarray(expected)
                if observed.shape != expected_array.shape or observed.dtype != expected_array.dtype:
                    raise ValueError(f"NPZ round-trip schema failed: {key}")
                equal = (np.array_equal(observed, expected_array, equal_nan=True)
                         if observed.dtype.kind in "fc" else np.array_equal(observed, expected_array))
                if not equal:
                    raise ValueError(f"NPZ round-trip failed: {key}")
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_directory_no_clobber(building: Path, target: Path) -> None:
    """Publish a same-filesystem tree without ever replacing an existing target."""

    building = Path(building); target = Path(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if not building.is_dir() or building.is_symlink():
        raise ValueError("building directory unavailable")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(target)
    try:
        directories = sorted((p for p in building.rglob("*") if p.is_dir()), key=lambda p: len(p.parts))
        for directory in directories:
            (target / directory.relative_to(building)).mkdir()
        files = sorted((p for p in building.rglob("*") if p.is_file()), key=lambda p: (p.name == "complete.json", p.as_posix()))
        for source in files:
            destination = target / source.relative_to(building)
            os.link(source, destination)
        for directory in sorted((p for p in target.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
            _fsync_directory(directory)
        _fsync_directory(target); _fsync_directory(target.parent)
    except Exception:
        # Preserve a partial target as failure evidence; resume must reject it.
        raise
    shutil.rmtree(building)


def _tree_manifest(root: Path, *, exclude_complete: bool = True) -> Dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink forbidden in private output: {path}")
        if path.is_file() and not (exclude_complete and path == root / "complete.json"):
            result[path.relative_to(root).as_posix()] = sha256_file(path)
    return result


def verify_private_output(root: Path, expected_identity: Optional[Mapping[str, object]] = None) -> bool:
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or (root.parent / (root.name + ".building")).exists():
        raise ValueError("private output is missing, linked, or incomplete")
    complete_path = root / "complete.json"
    if not complete_path.is_file() or complete_path.is_symlink():
        raise ValueError("private completion marker missing")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("schema") != SCHEMA or not isinstance(complete.get("files"), dict):
        raise ValueError("private completion schema invalid")
    actual = _tree_manifest(root)
    if complete["files"] != actual:
        raise ValueError("private file set or SHA256 mismatch")
    identity = complete.get("identity", {})
    if expected_identity is not None and any(identity.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("private source identity drift")
    run_identity = json.loads((root / "run_identity.json").read_text(encoding="utf-8")) if (root / "run_identity.json").exists() else identity
    if run_identity != identity:
        raise ValueError("private run identity mismatch")
    return True


def _g3_tree_identity(root: Path) -> Mapping[str, object]:
    paths = sorted(path for path in Path(root).rglob("*") if path.is_file())
    digest = hashlib.sha256(); total_bytes = 0
    for path in paths:
        relative = path.relative_to(root).as_posix(); file_digest = sha256_file(path)
        digest.update(relative.encode("utf-8")); digest.update(file_digest.encode("ascii")); total_bytes += path.stat().st_size
    return {"file_count": len(paths), "total_bytes": total_bytes, "sha256": digest.hexdigest()}


def _verify_hashed_root(root: Path, schema: str) -> Mapping[str, object]:
    complete_path = Path(root) / "complete.json"
    if not complete_path.is_file() or complete_path.is_symlink():
        raise ValueError("completion evidence unavailable")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if complete.get("schema") != schema or not isinstance(complete.get("files"), dict):
        raise ValueError("completion evidence schema mismatch")
    actual_names = {path.name for path in Path(root).iterdir() if path.is_file()}
    if actual_names != set(complete["files"]) | {"complete.json"}:
        raise ValueError("completion evidence contains missing or extra files")
    for name, digest in complete["files"].items():
        path = Path(root) / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f"completion evidence drift: {name}")
    return complete


def preflight(config: G3BConfig) -> Mapping[str, object]:
    if sha256_file(config.manifest_path) != MANIFEST_SHA256 or sha256_file(config.spec_path) != SPEC_SHA256:
        raise ValueError("Frozen Manifest or Spec drift")
    g2_identity = json.loads((config.g2_root / "run_identity.json").read_text(encoding="utf-8"))
    if g2_identity.get("cache_schema_version") != "g2b1_full_v1" or g2_identity.get("manifest_sha256") != MANIFEST_SHA256 or g2_identity.get("spec_sha256") != SPEC_SHA256:
        raise ValueError("G2-B1 identity drift")
    metadata = load_metadata_index(config.demographics_csv, config.g2_root)
    join_audit = check_identity_join(metadata)

    g3_tree = _g3_tree_identity(config.g3_root)
    if g3_tree != {"file_count": G3_ARTIFACT_FILES, "total_bytes": G3_ARTIFACT_BYTES, "sha256": G3_ARTIFACT_FINGERPRINT}:
        raise ValueError(f"G3-A artifact tree drift: {g3_tree}")
    g3_identity = json.loads((config.g3_root / "run_identity.json").read_text(encoding="utf-8"))
    expected_g3 = {"schema": "g3a_rank256_v1", "g2b1_cache_fingerprint": G2_CACHE_FINGERPRINT,
                   "manifest_sha256": MANIFEST_SHA256, "spec_sha256": SPEC_SHA256,
                   "levels": 256, "rounding": "half_up"}
    if any(g3_identity.get(key) != value for key, value in expected_g3.items()):
        raise ValueError("G3-A run identity drift")

    _verify_hashed_root(config.g3av_root, "g3av_verification_v1")
    verification_identity = json.loads((config.g3av_root / "verification_identity.json").read_text(encoding="utf-8"))
    verification_summary = json.loads((config.g3av_root / "verification_summary.json").read_text(encoding="utf-8"))
    if verification_identity.get("status") != "PASS" or verification_summary.get("status") != "PASS" or len(verification_summary.get("units", [])) != 6:
        raise ValueError("G3-A-V PASS evidence incomplete")
    if any(unit.get("status") != "PASS" for unit in verification_summary["units"]):
        raise ValueError("G3-A-V unit not PASS")
    expected_verifier_files = {
        "verifier_sha256": config.repo / "reproduction/siren_a/pipeline/verify_g3a.py",
        "rank_verifier_sha256": config.repo / "reproduction/siren_a/pipeline/verify_rank.py",
    }
    for key, path in expected_verifier_files.items():
        if verification_identity.get(key) != sha256_file(path):
            raise ValueError(f"G3-A-V code fingerprint drift: {key}")
    expected_verification = {"source_cache_fingerprint": G2_CACHE_FINGERPRINT,
                             "manifest_sha256": MANIFEST_SHA256, "spec_sha256": SPEC_SHA256,
                             "inventory_fingerprint": g2_identity["inventory_fingerprint"]}
    if any(verification_identity.get(key) != value for key, value in expected_verification.items()):
        raise ValueError("G3-A-V source identity drift")

    patient_counts: Dict[tuple, int] = {}
    for record in metadata.records:
        if record.patient_id:
            key = (record.site, record.patient_id); patient_counts[key] = patient_counts.get(key, 0) + 1
    patient_duplicates = sum(count > 1 for count in patient_counts.values())
    overlaps = {}
    for holdout, (train_site, _, _, _) in EXPECTED_FOLDS.items():
        train = np.flatnonzero(metadata.sites == train_site).astype(np.int64)
        test = np.flatnonzero(metadata.sites == holdout).astype(np.int64)
        overlaps[holdout] = check_patient_overlap(metadata, train, test)

    code_paths = {
        "label_join_sha256": config.repo / "reproduction/siren_a/pipeline/label_join.py",
        "input_data_sha256": config.repo / "reproduction/siren_a/pipeline/input_data.py",
        "adapters_sha256": config.repo / "reproduction/siren_a/pipeline/adapters.py",
        "run_inputs_sha256": config.repo / "reproduction/siren_a/pipeline/run_inputs.py",
    }
    code_hashes = {key: sha256_file(path) for key, path in code_paths.items()}
    identity = {
        "schema": SCHEMA,
        "g2b1_cache_fingerprint": G2_CACHE_FINGERPRINT,
        "g2b1_inventory_fingerprint": g2_identity["inventory_fingerprint"],
        "g3a_artifact_fingerprint": G3_ARTIFACT_FINGERPRINT,
        "g3a_run_identity_sha256": sha256_file(config.g3_root / "run_identity.json"),
        "g3av_verification_identity_sha256": sha256_file(config.g3av_root / "verification_identity.json"),
        "manifest_sha256": MANIFEST_SHA256, "spec_sha256": SPEC_SHA256,
        "demographics_sha256": metadata.demographics_sha256,
        "metadata_join_sha256": metadata.join_fingerprint,
        "python_version": sys.version.split()[0], "numpy_version": np.__version__,
        **code_hashes,
    }
    return {"identity": identity, "metadata": metadata, "join_audit": join_audit,
            "patient_duplicate_records": patient_duplicates, "patient_overlaps": overlaps,
            "g3_tree": g3_tree, "g3av_units": len(verification_summary["units"])}


def _save_adapter(path: Path, state: AdapterState, identity_sha: str) -> None:
    atomic_npz_no_clobber(
        path,
        model_type=np.asarray(state.model_type), feature_ids=np.asarray(state.feature_ids, dtype="U"),
        feature_ids_sha256=np.asarray(state.feature_ids_sha256), fit_indices_sha256=np.asarray(state.fit_indices_sha256),
        fit_scope=np.asarray(state.fit_scope), finite_counts=state.finite_counts,
        fill_values=state.fill_values, all_missing=state.all_missing,
        run_identity_sha256=np.asarray(identity_sha),
    )


def _load_adapter(path: Path) -> AdapterState:
    with np.load(path, allow_pickle=False) as archive:
        required = {"model_type", "feature_ids", "feature_ids_sha256", "fit_indices_sha256", "fit_scope",
                    "finite_counts", "fill_values", "all_missing", "run_identity_sha256"}
        if set(archive.files) != required:
            raise ValueError("adapter state keys invalid")
        scalar = lambda key: str(np.asarray(archive[key]).item())
        return AdapterState(scalar("model_type"), tuple(np.asarray(archive["feature_ids"]).tolist()),
                            scalar("feature_ids_sha256"), scalar("fit_indices_sha256"), scalar("fit_scope"),
                            np.asarray(archive["finite_counts"]), np.asarray(archive["fill_values"]),
                            np.asarray(archive["all_missing"]))


def _summary_row(fold, model: str, state: AdapterState, train_output: np.ndarray, age_audit: Mapping[str, object]) -> Dict[str, object]:
    return {
        "holdout_site": fold.holdout_site, "train_site": fold.train_site,
        "protocol": fold.protocol, "view": fold.view, "model_type": model,
        "train_count": len(fold.X_train), "test_count": len(fold.X_test), "feature_count": len(fold.feature_ids),
        "source_missing_count": int(np.isnan(fold.X_train).sum()),
        "adapter_missing_count": int(np.isnan(train_output).sum()),
        "train_all_missing_columns": int(state.all_missing.sum()),
        "train_positive_count": int(np.sum(fold.y_train == 1)),
        "train_negative_count": int(np.sum(fold.y_train == 0)),
        "age_group_count": age_audit["group_count"],
        "groups_with_both_classes": age_audit["groups_with_both_classes"],
        "metadata_join_status": "PASS", "leakage_check_status": "PASS", "adapter_status": "PASS",
    }


def _evaluate_combinations(config: G3BConfig, pre: Mapping[str, object], holds: Sequence[str], building: Optional[Path]) -> Sequence[Mapping[str, object]]:
    metadata = pre["metadata"]; identity = pre["identity"]
    identity_sha = hashlib.sha256(canonical_json(identity)).hexdigest()
    rows = []
    for holdout in holds:
        protocol_folds = {}
        train_meta_saved = False
        for protocol in PROTOCOLS:
            protocol_folds[protocol] = {}
            for view in VIEWS:
                fold = load_fold(config.g3_root, holdout, protocol, view, metadata, config.manifest_path, identity)
                protocol_folds[protocol][view] = fold
                age_audit = audit_age_groups(fold.train_age_groups, fold.y_train)
                if building is not None and not train_meta_saved:
                    atomic_npz_no_clobber(building / "folds" / f"holdout_{holdout}" / "train_metadata.npz",
                                           train_indices=fold.train_indices, y_train=fold.y_train,
                                           age_train=fold.age_train, train_age_groups=fold.train_age_groups)
                    atomic_json_no_clobber(building / "folds" / f"holdout_{holdout}" / "age_group_audit.json", age_audit)
                    train_meta_saved = True
                for model in MODELS:
                    state = fit_adapter(fold.X_train, model, fold.feature_ids,
                                        fit_indices=fold.train_indices, fit_scope="full_train_preview")
                    transformed_train = transform_adapter(fold.X_train, state)
                    transformed_test = transform_adapter(fold.X_test, state)
                    check_adapter(fold.X_train, transformed_train, state)
                    check_adapter(fold.X_test, transformed_test, state)
                    rows.append(_summary_row(fold, model, state, transformed_train, age_audit))
                    if building is not None:
                        base = building / "folds" / f"holdout_{holdout}" / protocol / view / model
                        _save_adapter(base / "adapter_state.npz", state, identity_sha)
                        atomic_json_no_clobber(base / "checks.json", {
                            "status": "PASS", "train_shape": list(fold.X_train.shape), "test_shape": list(fold.X_test.shape),
                            "train_missing_before": int(np.isnan(fold.X_train).sum()),
                            "train_missing_after": int(np.isnan(transformed_train).sum()),
                            "test_missing_before": int(np.isnan(fold.X_test).sum()),
                            "test_missing_after": int(np.isnan(transformed_test).sum()),
                            "feature_ids_sha256": state.feature_ids_sha256, "fit_indices_sha256": state.fit_indices_sha256,
                            "train_indices_sha256": array_sha256(fold.train_indices),
                            "test_indices_sha256": array_sha256(fold.test_indices),
                            "source_fingerprints": dict(fold.source_fingerprints), "fit_scope": state.fit_scope,
                        })
        for view in VIEWS:
            check_cross_protocol_train(protocol_folds["transductive"][view], protocol_folds["inductive"][view])
    return rows


def _write_private(config: G3BConfig, pre: Mapping[str, object], holds: Sequence[str]) -> Sequence[Mapping[str, object]]:
    target = config.private_root; building = target.parent / (target.name + ".building")
    if target.exists() or target.is_symlink():
        verify_private_output(target, pre["identity"])
        return json.loads((target / "summary.json").read_text(encoding="utf-8"))["rows"]
    if building.exists() or building.is_symlink():
        raise ValueError("incomplete G3-B building directory exists; refusing overwrite")
    building.mkdir(parents=True)
    started = time.time()
    identity = dict(pre["identity"])
    identity.update({"mode": "build", "holdouts": list(holds), "expected_combinations": len(holds) * 30})
    # Completion identity is the full run identity.
    pre_for_run = dict(pre); pre_for_run["identity"] = identity
    atomic_json_no_clobber(building / "run_identity.json", identity)
    metadata = pre["metadata"]
    private_mapping = [
        {"cache_name": metadata.cache_names[i], "relative_path": metadata.relative_paths[i], "site": record.site,
         "bids_folder": record.bids_folder, "session_id": record.session_id, "csv_row_index": record.csv_row_index}
        for i, record in enumerate(metadata.records)
    ]
    atomic_json_no_clobber(building / "metadata_join" / "private_mapping.json", private_mapping)
    atomic_json_no_clobber(building / "metadata_join" / "join_identity.json", {
        "demographics_sha256": metadata.demographics_sha256, "metadata_join_sha256": metadata.join_fingerprint,
        "cache_count": len(metadata.records), "status": "PASS",
    })
    atomic_json_no_clobber(building / "metadata_join" / "join_audit.json", {
        **dict(pre["join_audit"]), "patient_duplicate_records": pre["patient_duplicate_records"],
        "patient_overlaps": pre["patient_overlaps"],
    })
    rows = _evaluate_combinations(config, pre_for_run, holds, building)
    elapsed = time.time() - started
    summary = {"status": "PASS", "combination_count": len(rows), "rows": rows,
               "duration_seconds": elapsed, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    atomic_json_no_clobber(building / "summary.json", summary)
    files = _tree_manifest(building)
    atomic_json_no_clobber(building / "complete.json", {"schema": SCHEMA, "identity": identity, "files": files})
    publish_directory_no_clobber(building, target)
    verify_private_output(target, identity)
    return rows


def verify_inputs(config: G3BConfig, pre: Optional[Mapping[str, object]] = None) -> Sequence[Mapping[str, object]]:
    pre = preflight(config) if pre is None else pre
    complete = json.loads((config.private_root / "complete.json").read_text(encoding="utf-8"))
    identity = complete.get("identity", {})
    expected_base = pre["identity"]
    if any(identity.get(key) != value for key, value in expected_base.items()):
        raise ValueError("G3-B private identity drift")
    verify_private_output(config.private_root, identity)
    holds = tuple(identity.get("holdouts", ()))
    if holds != tuple(EXPECTED_FOLDS):
        raise ValueError("G3-B full build does not contain all frozen holdouts")
    metadata = pre["metadata"]
    expected_mapping = [
        {"cache_name": metadata.cache_names[i], "relative_path": metadata.relative_paths[i], "site": record.site,
         "bids_folder": record.bids_folder, "session_id": record.session_id, "csv_row_index": record.csv_row_index}
        for i, record in enumerate(metadata.records)
    ]
    saved_mapping = json.loads((config.private_root / "metadata_join/private_mapping.json").read_text(encoding="utf-8"))
    if saved_mapping != expected_mapping:
        raise ValueError("private metadata identity mapping mismatch")
    expected_join_identity = {"demographics_sha256": metadata.demographics_sha256,
                              "metadata_join_sha256": metadata.join_fingerprint,
                              "cache_count": len(metadata.records), "status": "PASS"}
    saved_join_identity = json.loads((config.private_root / "metadata_join/join_identity.json").read_text(encoding="utf-8"))
    if saved_join_identity != expected_join_identity:
        raise ValueError("private metadata join identity mismatch")
    expected_join_audit = {**dict(pre["join_audit"]), "patient_duplicate_records": pre["patient_duplicate_records"],
                           "patient_overlaps": pre["patient_overlaps"]}
    saved_join_audit = json.loads((config.private_root / "metadata_join/join_audit.json").read_text(encoding="utf-8"))
    if saved_join_audit != expected_join_audit:
        raise ValueError("private metadata join audit mismatch")
    for holdout in holds:
        fold = load_fold(config.g3_root, holdout, "transductive", "V1", metadata,
                         config.manifest_path, identity)
        metadata_path = config.private_root / "folds" / f"holdout_{holdout}" / "train_metadata.npz"
        with np.load(metadata_path, allow_pickle=False) as archive:
            expected_arrays = {"train_indices": fold.train_indices, "y_train": fold.y_train,
                               "age_train": fold.age_train, "train_age_groups": fold.train_age_groups}
            if set(archive.files) != set(expected_arrays):
                raise ValueError("private training metadata keys mismatch")
            for key, expected in expected_arrays.items():
                if not np.array_equal(np.asarray(archive[key]), expected):
                    raise ValueError(f"private training metadata mismatch: {holdout}:{key}")
        expected_age_audit = audit_age_groups(fold.train_age_groups, fold.y_train)
        saved_age_audit = json.loads((config.private_root / "folds" / f"holdout_{holdout}" / "age_group_audit.json").read_text(encoding="utf-8"))
        if saved_age_audit != expected_age_audit:
            raise ValueError(f"private age-group audit mismatch: {holdout}")
    expected_rows = _evaluate_combinations(config, pre, holds, None)
    saved_rows = json.loads((config.private_root / "summary.json").read_text(encoding="utf-8"))["rows"]
    if expected_rows != saved_rows or len(saved_rows) != 90:
        raise ValueError("G3-B aggregate semantic verification failed")
    identity_sha = hashlib.sha256(canonical_json(identity)).hexdigest()
    for row in saved_rows:
        path = config.private_root / "folds" / f"holdout_{row['holdout_site']}" / row["protocol"] / row["view"] / row["model_type"] / "adapter_state.npz"
        state = _load_adapter(path)
        fold = load_fold(config.g3_root, row["holdout_site"], row["protocol"], row["view"],
                         pre["metadata"], config.manifest_path, identity)
        expected_state = fit_adapter(fold.X_train, row["model_type"], fold.feature_ids,
                                     fit_indices=fold.train_indices, fit_scope="full_train_preview")
        scalar_fields = ("model_type", "feature_ids", "feature_ids_sha256", "fit_indices_sha256", "fit_scope")
        if any(getattr(state, key) != getattr(expected_state, key) for key in scalar_fields):
            raise ValueError("adapter state semantic verification failed")
        for key in ("finite_counts", "fill_values", "all_missing"):
            if not np.array_equal(getattr(state, key), getattr(expected_state, key), equal_nan=True):
                raise ValueError(f"adapter state array mismatch: {key}")
        with np.load(path, allow_pickle=False) as archive:
            if str(np.asarray(archive["run_identity_sha256"]).item()) != identity_sha:
                raise ValueError("adapter state run identity mismatch")
        checks_path = path.with_name("checks.json")
        checks = json.loads(checks_path.read_text(encoding="utf-8"))
        train_output = transform_adapter(fold.X_train, state); test_output = transform_adapter(fold.X_test, state)
        check_adapter(fold.X_train, train_output, state); check_adapter(fold.X_test, test_output, state)
        expected_checks = {
            "status": "PASS", "train_shape": list(fold.X_train.shape), "test_shape": list(fold.X_test.shape),
            "train_missing_before": int(np.isnan(fold.X_train).sum()),
            "train_missing_after": int(np.isnan(train_output).sum()),
            "test_missing_before": int(np.isnan(fold.X_test).sum()),
            "test_missing_after": int(np.isnan(test_output).sum()),
            "feature_ids_sha256": state.feature_ids_sha256, "fit_indices_sha256": state.fit_indices_sha256,
            "train_indices_sha256": array_sha256(fold.train_indices),
            "test_indices_sha256": array_sha256(fold.test_indices),
            "source_fingerprints": dict(fold.source_fingerprints), "fit_scope": state.fit_scope,
        }
        if checks != expected_checks:
            raise ValueError("adapter checks semantic verification failed")
    return saved_rows


def _public_csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    fields = ("holdout_site", "train_site", "protocol", "view", "model_type", "train_count", "test_count",
              "feature_count", "source_missing_count", "adapter_missing_count", "train_all_missing_columns",
              "train_positive_count", "train_negative_count", "age_group_count", "groups_with_both_classes",
              "metadata_join_status", "leakage_check_status", "adapter_status")
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\r\n")
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue()


def ensure_public_csv(path: Path, rows: Sequence[Mapping[str, object]], *, expected_count: int = 90) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) != expected_count:
        raise ValueError(f"public summary row count must equal {expected_count}")
    expected_text = _public_csv_text(rows)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != expected_text.encode("utf-8"):
            raise ValueError("existing public summary does not match verified rows")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name); handle.write(expected_text); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()


def write_public_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Compatibility wrapper retaining strict no-clobber verified semantics."""
    ensure_public_csv(path, rows)


def validate_public_rows(rows: Sequence[Mapping[str, object]]) -> bool:
    text = json.dumps(rows, sort_keys=True).lower()
    if any(token in text for token in PRIVATE_TOKENS):
        raise ValueError("private identity leaked into public rows")
    if len(rows) != 90:
        raise ValueError("public summary must contain exactly 90 combinations")
    return True


def prepare_inputs(config: G3BConfig, mode: str) -> Mapping[str, object]:
    pre = preflight(config)
    if mode == "preflight-only":
        return {"status": "PASS", "matched": pre["join_audit"]["matched"], "combinations": 0}
    if mode == "smoke":
        rows = _evaluate_combinations(config, pre, ("I0002",), None)
        return {"status": "PASS", "matched": pre["join_audit"]["matched"], "combinations": len(rows), "rows": rows}
    if mode == "build":
        rows = _write_private(config, pre, tuple(EXPECTED_FOLDS))
        verified = verify_inputs(config, pre)
        validate_public_rows(verified)
        ensure_public_csv(config.public_csv, verified)
        return {"status": "PASS", "matched": pre["join_audit"]["matched"], "combinations": len(rows), "rows": verified}
    if mode == "verify-only":
        rows = verify_inputs(config, pre)
        validate_public_rows(rows)
        return {"status": "PASS", "matched": pre["join_audit"]["matched"], "combinations": len(rows), "rows": rows}
    raise ValueError(f"unsupported mode: {mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and audit label-safe G3-B model inputs; no model training.")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--build", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    mode = next(name for name in ("preflight_only", "smoke", "build", "verify_only") if getattr(args, name)).replace("_", "-")
    result = prepare_inputs(default_config(), mode)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
