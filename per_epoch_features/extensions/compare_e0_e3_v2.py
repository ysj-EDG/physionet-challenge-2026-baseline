"""Fixed old/v2 prediction references, rank fusion, and delayed scoring."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from reproduction.siren_a.g5.core import (
    HOLDOUTS,
    HOLDOUT_TO_TRAIN,
    MODEL_TYPES,
    PROTOCOLS,
    VIEWS,
    aggregate_protocol,
    load_eval_targets,
    mean_ranks,
    score_site,
)

EXPERIMENT_SOURCES = {
    "E0": {"V1": "old", "V2": "old", "V3": "old", "V4": "old", "V5": "old"},
    "E1": {"V1": "v2", "V2": "v2", "V3": "v2", "V4": "v2", "V5": "old"},
    "E2": {"V1": "old", "V2": "old", "V3": "old", "V4": "old", "V5": "v2"},
    "E3": {"V1": "v2", "V2": "v2", "V3": "v2", "V4": "v2", "V5": "v2"},
}
SEEDS = tuple(range(42, 57))
OLD_MANIFEST_SCHEMA = "g4b_predictions_manifest_v1"
SEED42_PREDICTION_SCHEMA = "featurelab_v2_seed42_prediction_v1"
MULTISEED_PREDICTION_SCHEMA = "featurelab_v2_multiseed_prediction_v1"
PREDICTION_FILES = {"prediction.npz", "metadata.json", "complete.json"}


def fixed_column_keys() -> tuple[tuple[int, str, str], ...]:
    return tuple((seed, view, model) for seed in SEEDS for view in VIEWS for model in MODEL_TYPES)


def _prediction_keys() -> set[tuple[str, str, str, str, int]]:
    return {
        (holdout, protocol, view, model, seed)
        for holdout in HOLDOUTS
        for protocol in PROTOCOLS
        for view in VIEWS
        for model in MODEL_TYPES
        for seed in SEEDS
    }


def normalize_score_type(value: str) -> str:
    if value == "raw_ranking_score":
        return "lambdarank_raw"
    if value in {"lambdarank_raw", "positive_probability"}:
        return value
    raise ValueError(f"unknown score type: {value}")


def _expected_score_type(model_type: str, *, old: bool) -> str:
    if model_type == "lambdarank":
        return "raw_ranking_score" if old else "lambdarank_raw"
    return "positive_probability"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _array_sha(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _v2_array_sha(value: np.ndarray) -> str:
    """Native Seed42/multiseed digest; kept distinct from the bridge digest."""
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical(list(array.shape)))
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc


def _tree_summaries(path: Path) -> dict[str, str]:
    return {
        item.name: _file_sha(item)
        for item in sorted(Path(path).iterdir())
        if item.is_file() and item.name != "complete.json"
    }


def _inside_approved(path: Path, approved_roots: Sequence[Path], *, directory: bool) -> Path:
    candidate = Path(path)
    if candidate.is_symlink() or (not candidate.is_dir() if directory else not candidate.is_file()):
        raise ValueError(f"prediction path missing, wrong type, or symlinked: {candidate}")
    resolved = candidate.resolve()
    for root_value in approved_roots:
        root_path = Path(root_value)
        if root_path.is_symlink() or not root_path.is_dir():
            continue
        root_absolute = root_path.absolute()
        candidate_absolute = candidate.absolute()
        try:
            raw_relative = candidate_absolute.relative_to(root_absolute)
        except ValueError:
            continue
        raw_cursor = root_absolute
        for part in raw_relative.parts:
            raw_cursor = raw_cursor / part
            if raw_cursor.is_symlink():
                raise ValueError(f"symlinked prediction component: {raw_cursor}")
        root = root_path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        cursor = root
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError(f"symlinked prediction component: {cursor}")
        return resolved
    raise ValueError(f"prediction path outside approved roots: {candidate}")


def _approved_root_for(path: Path, approved_roots: Sequence[Path]) -> Path:
    resolved = Path(path).resolve()
    for value in approved_roots:
        root = Path(value).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        return root
    raise ValueError(f"prediction path outside approved roots: {path}")


def _read_prediction_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = Path(path) / "prediction.npz" if Path(path).is_dir() else Path(path)
    try:
        with np.load(payload, allow_pickle=False) as archive:
            if set(archive.files) != {"scores", "test_global_indices"}:
                raise ValueError("prediction payload keys mismatch")
            scores = np.asarray(archive["scores"])
            indices = np.asarray(archive["test_global_indices"])
    except (OSError, ValueError, TypeError) as exc:
        if isinstance(exc, ValueError) and str(exc) == "prediction payload keys mismatch":
            raise
        raise ValueError(f"prediction payload unreadable: {payload}") from exc
    if scores.ndim != 1 or scores.dtype != np.float64 or not np.isfinite(scores).all():
        raise ValueError("invalid score vector")
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) != len(scores)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("invalid test indices")
    return scores, indices.astype(np.int64, copy=False)


def _validate_score_semantics(scores: np.ndarray, model_type: str, score_type: str, *, old: bool) -> None:
    expected = _expected_score_type(model_type, old=old)
    if score_type != expected:
        raise ValueError(f"prediction score_type mismatch: expected {expected}, got {score_type}")
    if model_type in {"lr", "hgb"} and (np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("probability prediction outside [0,1]")


def validate_v2_prediction_unit(
    path: Path,
    expected_identity: Mapping[str, object],
    *,
    seed42_legacy: bool,
    approved_root: Path,
) -> dict[str, object]:
    """Validate one frozen v2 prediction unit without changing its metadata."""
    resolved = _inside_approved(Path(path), (Path(approved_root),), directory=True)
    if {item.name for item in resolved.iterdir()} != PREDICTION_FILES:
        raise ValueError("v2 prediction unit file set mismatch")
    metadata_path = resolved / "metadata.json"
    complete_path = resolved / "complete.json"
    metadata = _json(metadata_path)
    complete = _json(complete_path)
    expected_schema = SEED42_PREDICTION_SCHEMA if seed42_legacy else MULTISEED_PREDICTION_SCHEMA
    if metadata.get("schema") != expected_schema:
        raise ValueError("v2 prediction schema mismatch")
    for field, expected in expected_identity.items():
        if metadata.get(field) != expected:
            raise ValueError(f"v2 prediction identity mismatch: {field}")
    source_version = metadata.get("source_version")
    if seed42_legacy:
        if source_version not in (None, "v2"):
            raise ValueError("Seed42 source_version mismatch")
    elif source_version != "v2":
        raise ValueError("v2 prediction source_version mismatch")
    if (complete.get("schema") != expected_schema or complete.get("status") != "COMPLETE"
            or complete.get("file_summaries") != _tree_summaries(resolved)):
        raise ValueError("v2 prediction completion digest mismatch")
    scores, indices = _read_prediction_arrays(resolved)
    model_type = str(expected_identity["model_type"])
    _validate_score_semantics(scores, model_type, str(metadata.get("score_type")), old=False)
    if metadata.get("finite") is not True or metadata.get("nonconstant") is not True:
        raise ValueError("v2 prediction validity metadata mismatch")
    if int(metadata.get("test_count", -1)) != len(indices):
        raise ValueError("v2 prediction test_count mismatch")
    native_prediction_sha = _v2_array_sha(scores)
    native_indices_sha = _v2_array_sha(indices)
    if metadata.get("prediction_sha256") != native_prediction_sha:
        raise ValueError("v2 prediction array SHA256 mismatch")
    if metadata.get("test_indices_sha256") != native_indices_sha:
        raise ValueError("v2 test-index array SHA256 mismatch")
    prediction_sha = _array_sha(scores)
    indices_sha = _array_sha(indices)
    for field in ("model_complete_sha256", "run_identity_sha256"):
        value = metadata.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"v2 prediction trace mismatch: {field}")
    root = _approved_root_for(resolved, (Path(approved_root),))
    run_identity_path = root / "run_identity.json"
    if (run_identity_path.is_symlink() or not run_identity_path.is_file()
            or _file_sha(run_identity_path) != metadata["run_identity_sha256"]):
        raise ValueError("v2 run identity completion mismatch")
    model_complete_path = (root / "models" / str(expected_identity["train_site"])
                           / f"seed{int(expected_identity['seed'])}" / str(expected_identity["view"])
                           / str(expected_identity["model_type"]) / "complete.json")
    if (model_complete_path.is_symlink() or not model_complete_path.is_file()
            or _file_sha(model_complete_path) != metadata["model_complete_sha256"]):
        raise ValueError("v2 model completion identity mismatch")
    source_identity = hashlib.sha256(_canonical({
        "schema": expected_schema,
        "identity": dict(expected_identity),
        "metadata_sha256": _file_sha(metadata_path),
        "complete_sha256": _file_sha(complete_path),
        "prediction_file_sha256": _file_sha(resolved / "prediction.npz"),
    })).hexdigest()
    return {
        **metadata,
        "source_version": "v2",
        "prediction_reference": str(resolved),
        "bridge_prediction_sha256": prediction_sha,
        "bridge_test_indices_sha256": indices_sha,
        "source_identity_sha256": source_identity,
        "source_metadata_sha256": _file_sha(metadata_path),
        "source_complete_sha256": _file_sha(complete_path),
        "prediction_file_sha256": _file_sha(resolved / "prediction.npz"),
    }


def validate_old_prediction_row(
    row: Mapping[str, object],
    expected_identity: Mapping[str, object],
    *,
    approved_roots: Sequence[Path],
) -> dict[str, object]:
    """Validate a G4A/G4B prediction using its native unit contract."""
    for field, expected in expected_identity.items():
        if row.get(field) != expected:
            raise ValueError(f"old prediction identity mismatch: {field}")
    seed = int(expected_identity["seed"])
    model = str(expected_identity["model_type"])
    path = _inside_approved(Path(str(row["prediction_reference"])), approved_roots, directory=False)
    root = _approved_root_for(path, approved_roots)
    unit = path.parent
    if unit.is_symlink() or {item.name for item in unit.iterdir()} != PREDICTION_FILES:
        raise ValueError("old prediction unit file set mismatch")
    metadata_path = unit / "metadata.json"
    complete_path = unit / "complete.json"
    metadata = _json(metadata_path)
    complete = _json(complete_path)
    schema = "g4a_prediction_v1" if seed == 42 else "g4b_prediction_v1"
    if metadata.get("schema") != schema or complete.get("schema") != schema:
        raise ValueError("old prediction schema mismatch")
    for field, expected in expected_identity.items():
        if metadata.get(field) != expected:
            raise ValueError(f"old prediction metadata identity mismatch: {field}")
    expected_files = {
        "metadata.json": _file_sha(metadata_path),
        "prediction.npz": _file_sha(path),
    }
    if complete.get("files") != expected_files:
        raise ValueError("old prediction completion file digest mismatch")
    identity = complete.get("identity")
    if not isinstance(identity, dict):
        raise ValueError("old prediction completion identity missing")
    for field in ("schema", "holdout_site", "train_site", "protocol", "view", "model_type", "seed",
                  "run_identity_sha256", "model_complete_sha256", "prediction_sha256", "test_indices_sha256"):
        if identity.get(field) != metadata.get(field):
            raise ValueError(f"old prediction completion identity mismatch: {field}")
    if row.get("completion_sha256") != _file_sha(complete_path):
        raise ValueError("old prediction completion SHA256 mismatch")
    trace_fields = {
        "model_sha256": "checkpoint_sha256",
        "adapter_sha256": "adapter_state_sha256",
        "feature_order_sha256": "feature_ids_sha256",
    }
    for row_field, metadata_field in trace_fields.items():
        if row.get(row_field) != metadata.get(metadata_field):
            raise ValueError(f"old prediction {row_field.split('_')[0]} trace mismatch")
    model_complete_path = (
        root / "models" / str(expected_identity["train_site"]) / "seed42"
        / str(expected_identity["view"]) / model / "complete.json"
        if seed == 42 else
        root / "seeds" / f"seed{seed}" / "models" / str(expected_identity["train_site"])
        / str(expected_identity["view"]) / model / "complete.json"
    )
    if (model_complete_path.is_symlink() or not model_complete_path.is_file()
            or _file_sha(model_complete_path) != metadata.get("model_complete_sha256")):
        raise ValueError("old model completion identity mismatch")
    scores, indices = _read_prediction_arrays(path)
    _validate_score_semantics(scores, model, str(row.get("score_type")), old=True)
    if metadata.get("score_type") != row.get("score_type"):
        raise ValueError("old prediction metadata score_type mismatch")
    prediction_sha = _array_sha(scores)
    indices_sha = _array_sha(indices)
    if row.get("prediction_sha256") != prediction_sha or metadata.get("prediction_sha256") != prediction_sha:
        raise ValueError("old prediction array SHA256 mismatch")
    if row.get("test_indices_sha256") != indices_sha or metadata.get("test_indices_sha256") != indices_sha:
        raise ValueError("old test-index array SHA256 mismatch")
    if int(row.get("prediction_length", -1)) != len(indices) or int(metadata.get("test_count", -1)) != len(indices):
        raise ValueError("old prediction length mismatch")
    if metadata.get("finite") is not True or metadata.get("nonconstant") is not True:
        raise ValueError("old prediction validity metadata mismatch")
    source_identity = hashlib.sha256(_canonical({
        "identity": dict(expected_identity),
        "metadata_sha256": _file_sha(metadata_path),
        "complete_sha256": _file_sha(complete_path),
        "prediction_file_sha256": _file_sha(path),
    })).hexdigest()
    return {
        **row,
        "source_version": "old",
        "prediction_reference": str(path),
        "bridge_prediction_sha256": prediction_sha,
        "bridge_test_indices_sha256": indices_sha,
        "source_identity_sha256": source_identity,
        "source_metadata_sha256": _file_sha(metadata_path),
        "source_complete_sha256": _file_sha(complete_path),
        "prediction_file_sha256": _file_sha(path),
    }


def collect_old_rows(
    manifest_path: Path,
    approved_roots: Sequence[Path],
) -> dict[tuple[str, str, str, str, int], Mapping[str, object]]:
    payload = _json(manifest_path)
    if payload.get("schema") != OLD_MANIFEST_SCHEMA:
        raise ValueError("old prediction manifest schema mismatch")
    rows = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("old prediction manifest entries missing")
    indexed = {}
    for source in rows:
        key = (str(source["holdout_site"]), str(source["protocol"]), str(source["view"]),
               str(source["model_type"]), int(source["seed"]))
        if key in indexed:
            raise ValueError("duplicate old prediction identity")
        indexed[key] = dict(source)
    if set(indexed) != _prediction_keys():
        raise ValueError("old prediction manifest coverage mismatch")
    result = {}
    for key, row in indexed.items():
        holdout, protocol, view, model, seed = key
        expected = {
            "seed": seed,
            "holdout_site": holdout,
            "train_site": HOLDOUT_TO_TRAIN[holdout],
            "protocol": protocol,
            "view": view,
            "model_type": model,
        }
        result[key] = validate_old_prediction_row(row, expected, approved_roots=approved_roots)
    return result


def collect_v2_rows(
    seed42_root: Path,
    new_root: Path,
) -> dict[tuple[str, str, str, str, int], Mapping[str, object]]:
    result = {}
    roots = ((42, Path(seed42_root)),) + tuple((seed, Path(new_root)) for seed in SEEDS if seed != 42)
    for seed, root in roots:
        for holdout in HOLDOUTS:
            for protocol in PROTOCOLS:
                for view in VIEWS:
                    for model in MODEL_TYPES:
                        path = (root / "predictions" / f"holdout_{holdout}" / protocol / view / model
                                if seed == 42 else root / "predictions" / f"seed{seed}" / f"holdout_{holdout}" / protocol / view / model)
                        expected = {
                            "seed": seed,
                            "holdout_site": holdout,
                            "train_site": HOLDOUT_TO_TRAIN[holdout],
                            "protocol": protocol,
                            "view": view,
                            "model_type": model,
                        }
                        row = validate_v2_prediction_unit(
                            path,
                            expected,
                            seed42_legacy=seed == 42,
                            approved_root=root,
                        )
                        key = (holdout, protocol, view, model, seed)
                        if key in result:
                            raise ValueError("duplicate v2 prediction identity")
                        result[key] = row
    if set(result) != _prediction_keys():
        raise ValueError("v2 prediction manifest coverage mismatch")
    return result


def build_reference_entries(old_rows: Mapping, v2_rows: Mapping, experiment: str) -> list[Mapping[str, object]]:
    if experiment not in EXPERIMENT_SOURCES:
        raise ValueError(experiment)
    expected = _prediction_keys()
    if set(old_rows) != expected or set(v2_rows) != expected:
        raise ValueError("prediction source coverage mismatch")
    rows = []
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            for seed in SEEDS:
                for view in VIEWS:
                    source = EXPERIMENT_SOURCES[experiment][view]
                    for model in MODEL_TYPES:
                        key = (holdout, protocol, view, model, seed)
                        row = old_rows[key] if source == "old" else v2_rows[key]
                        if row.get("source_version") != source:
                            raise ValueError("source version conflicts with experiment map")
                        rows.append({
                            "experiment": experiment,
                            "holdout_site": holdout,
                            "protocol": protocol,
                            "seed": seed,
                            "view": view,
                            "model_type": model,
                            "source_version": source,
                            "prediction_reference": str(row["prediction_reference"]),
                            "score_type": normalize_score_type(str(row["score_type"])),
                            "bridge_prediction_sha256": str(row["bridge_prediction_sha256"]),
                            "bridge_test_indices_sha256": str(row["bridge_test_indices_sha256"]),
                            "source_identity_sha256": str(row["source_identity_sha256"]),
                            "prediction_file_sha256": str(row.get("prediction_file_sha256", "")),
                            "source_metadata_sha256": str(row.get("source_metadata_sha256", "")),
                            "source_complete_sha256": str(row.get("source_complete_sha256", "")),
                        })
    if len(rows) != 1350:
        raise ValueError("reference entry count mismatch")
    return rows


def group_reference_entries(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], list[Mapping[str, object]]]:
    expected_group_keys = {
        (experiment, holdout, protocol)
        for experiment in EXPERIMENT_SOURCES
        for holdout in HOLDOUTS
        for protocol in PROTOCOLS
    }
    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["experiment"]), str(row["holdout_site"]), str(row["protocol"]))
        groups.setdefault(key, []).append(row)
    if set(groups) != expected_group_keys:
        raise ValueError("reference group coverage mismatch")
    for (experiment, _, _), group in groups.items():
        if len(group) != 225:
            raise ValueError("reference group size mismatch")
        logical = tuple((int(row["seed"]), str(row["view"]), str(row["model_type"])) for row in group)
        if len(set(logical)) != 225 or set(logical) != set(fixed_column_keys()):
            raise ValueError("reference group logical identity mismatch")
        for row in group:
            view = str(row["view"])
            if row.get("source_version") != EXPERIMENT_SOURCES[experiment][view]:
                raise ValueError("reference experiment source mismatch")
    return groups


def build_all_reference_entries(old_rows: Mapping, v2_rows: Mapping) -> list[Mapping[str, object]]:
    rows = [
        row
        for experiment in EXPERIMENT_SOURCES
        for row in build_reference_entries(old_rows, v2_rows, experiment)
    ]
    if len(rows) != 5400:
        raise ValueError("all-experiment reference count mismatch")
    group_reference_entries(rows)
    for holdout in HOLDOUTS:
        digests = {
            str(row["bridge_test_indices_sha256"])
            for row in rows
            if row["holdout_site"] == holdout
        }
        if len(digests) != 1:
            raise ValueError(f"cross-experiment test-index identity mismatch: {holdout}")
    return rows


def load_prediction_row(row: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(row["prediction_reference"]))
    payload_path = path / "prediction.npz" if path.is_dir() else path
    expected_file_sha = str(row.get("prediction_file_sha256", ""))
    if expected_file_sha and _file_sha(payload_path) != expected_file_sha:
        raise ValueError("prediction file SHA256 drift")
    if path.is_dir():
        metadata_sha = str(row.get("source_metadata_sha256", ""))
        complete_sha = str(row.get("source_complete_sha256", ""))
        if not metadata_sha or _file_sha(path / "metadata.json") != metadata_sha:
            raise ValueError("prediction metadata SHA256 drift")
        if not complete_sha or _file_sha(path / "complete.json") != complete_sha:
            raise ValueError("prediction completion SHA256 drift")
    scores, indices = _read_prediction_arrays(path)
    if _array_sha(scores) != row.get("bridge_prediction_sha256"):
        raise ValueError("bridge prediction SHA256 mismatch")
    if _array_sha(indices) != row.get("bridge_test_indices_sha256"):
        raise ValueError("bridge test-index SHA256 mismatch")
    expected = normalize_score_type(str(row["score_type"]))
    if expected == "positive_probability" and (np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError("probability range")
    return scores, indices


def fuse_prediction_columns(score_matrix: np.ndarray, test_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores = np.asarray(score_matrix)
    indices = np.asarray(test_indices)
    if scores.ndim != 2 or scores.shape[1] != 225 or scores.dtype != np.float64 or not np.isfinite(scores).all():
        raise ValueError("fusion matrix must be finite N x 225")
    if indices.ndim != 1 or len(indices) != len(scores) or len(np.unique(indices)) != len(indices):
        raise ValueError("fusion indices invalid")
    ranks, fused = mean_ranks(scores)
    return indices.astype(np.int64, copy=True), fused.astype(np.float64, copy=False), ranks


def fuse_reference_group(entries: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    if len(entries) != 225:
        raise ValueError("each group requires 225 references")
    identities = {
        (str(row["experiment"]), str(row["holdout_site"]), str(row["protocol"]))
        for row in entries
    }
    if len(identities) != 1:
        raise ValueError("mixed fusion group identity")
    experiment, _, _ = next(iter(identities))
    if experiment not in EXPERIMENT_SOURCES:
        raise ValueError("unknown experiment")
    ordered = sorted(
        entries,
        key=lambda row: (
            int(row["seed"]),
            VIEWS.index(str(row["view"])),
            MODEL_TYPES.index(str(row["model_type"])),
        ),
    )
    expected = fixed_column_keys()
    actual = tuple((int(row["seed"]), str(row["view"]), str(row["model_type"])) for row in ordered)
    if actual != expected:
        raise ValueError("fusion column order or identity mismatch")
    for row in ordered:
        if row.get("source_version") != EXPERIMENT_SOURCES[experiment][str(row["view"])]:
            raise ValueError("fusion source version mismatch")
    columns = []
    canonical = None
    for row in ordered:
        scores, indices = load_prediction_row(row)
        if canonical is None:
            canonical = indices
        elif not np.array_equal(indices, canonical):
            raise ValueError("test index order mismatch")
        columns.append(scores)
    matrix = np.column_stack(columns).astype(np.float64, copy=False)
    out_indices, fused, ranks = fuse_prediction_columns(matrix, canonical)
    model_keys = tuple(f"seed{int(row['seed'])}|{row['view']}|{row['model_type']}" for row in ordered)
    return out_indices, fused, ranks, model_keys


def _array_equal_exact(left: np.ndarray, right: np.ndarray) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def assert_e0_fusion_regression(
    e0_groups: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    old_g5_root: Path,
) -> dict[str, object]:
    expected_groups = {(holdout, protocol) for holdout in HOLDOUTS for protocol in PROTOCOLS}
    if set(e0_groups) != expected_groups:
        raise ValueError("E0 regression group coverage mismatch")
    fields = ("model_keys", "test_global_indices", "model_ranks", "ensemble_scores")
    digests = {}
    for key in sorted(expected_groups):
        holdout, protocol = key
        path = Path(old_g5_root) / "groups" / f"{holdout}_{protocol}" / "fusion.npz"
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"old G5 fusion missing or symlinked: {key}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(fields):
                raise ValueError("old G5 fusion payload keys mismatch")
            old = {field: np.asarray(archive[field]) for field in fields}
        new = e0_groups[key]
        if set(new) != set(fields):
            raise ValueError("new E0 fusion payload keys mismatch")
        for field in fields:
            if not _array_equal_exact(np.asarray(new[field]), old[field]):
                raise ValueError(f"E0 fusion regression mismatch: {holdout}/{protocol}/{field}")
        digests[f"{holdout}|{protocol}"] = {
            field: _array_sha(old[field]) for field in fields
        }
    return {"groups": 6, "status": "EXACT_MATCH", "array_sha256": digests}


def verify_official_scoring_module(path: Path, expected_sha256: str) -> Path:
    """Pin the scoring implementation to the file and digest frozen by old G5."""
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("official scoring path missing or symlinked")
    resolved = candidate.resolve()
    if _file_sha(resolved) != str(expected_sha256):
        raise ValueError("official scoring SHA256 mismatch")
    return resolved


def score_fused_groups(
    groups: Mapping[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    metadata,
    official,
) -> tuple[list[dict[str, object]], dict[str, dict[str, float]]]:
    expected = {(holdout, protocol) for holdout in HOLDOUTS for protocol in PROTOCOLS}
    if set(groups) != expected:
        raise ValueError("scoring group coverage mismatch")
    rows = []
    for holdout in HOLDOUTS:
        shared_indices = None
        for protocol in PROTOCOLS:
            indices, fused = groups[(holdout, protocol)]
            indices = np.asarray(indices)
            if shared_indices is not None and not np.array_equal(indices, shared_indices):
                raise ValueError("protocol test-index mismatch")
            shared_indices = indices.copy()
            labels, ages = load_eval_targets(metadata, indices)
            metrics = score_site(labels, fused, ages, official)
            rows.append({
                "holdout_site": holdout,
                "train_site": HOLDOUT_TO_TRAIN[holdout],
                "protocol": protocol,
                "ensemble_method": "equal_mean_of_225_cohort_average_tie_ranks",
                "n_models": 225,
                **metrics,
            })
    summaries = {
        protocol: aggregate_protocol([row for row in rows if row["protocol"] == protocol])
        for protocol in PROTOCOLS
    }
    return rows, summaries


__all__ = [
    "EXPERIMENT_SOURCES",
    "SEEDS",
    "assert_e0_fusion_regression",
    "build_all_reference_entries",
    "build_reference_entries",
    "collect_old_rows",
    "collect_v2_rows",
    "fixed_column_keys",
    "fuse_prediction_columns",
    "fuse_reference_group",
    "group_reference_entries",
    "load_prediction_row",
    "normalize_score_type",
    "score_fused_groups",
    "validate_old_prediction_row",
    "validate_v2_prediction_unit",
    "verify_official_scoring_module",
]
