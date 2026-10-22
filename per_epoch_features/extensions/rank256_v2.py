"""Rank256 v2 adapter over the immutable FeatureLab candidate pool.

The numerical contract is the frozen G3-A average-tie percentile and half-up
256-level quantization.  This module adds only 1444-column input validation,
LOSO/view mapping, and recoverable no-clobber unit publication.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .pool_v2 import _pool_digest
from reproduction.siren_a.pipeline.normalize import (
    fit_rank_reference as _fit_rank_reference,
    apply_rank_reference as _apply_rank_reference,
    quantize_rank as _quantize_rank,
)

SCHEMA = "featurelab_v2_rank256_v1"
PROTOCOLS = ("transductive", "inductive")
HOLDOUTS = ("I0002", "I0006", "S0001")
VIEW_IDS = ("V1", "V2", "V3", "V4", "V5")
VIEW_DIMS_V1 = {"V1": 113, "V2": 129, "V3": 349, "V4": 113, "V5": 123}
VIEW_DIMS_V2 = {"V1": 161, "V2": 177, "V3": 493, "V4": 257, "V5": 376}


class UnitPublishError(RuntimeError):
    """An existing Rank256 unit cannot be safely reused or replaced."""


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_array(value) -> str:
    array = np.asarray(value)
    h = hashlib.sha256()
    h.update(array.dtype.str.encode("ascii")); h.update(_canonical(list(array.shape)).encode("ascii")); h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def _validate_matrix(values, source_valid_mask, expected_width: int | None = None):
    values = np.asarray(values, dtype=np.float64); mask = np.asarray(source_valid_mask, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape:
        raise ValueError("values/source_valid_mask must be matching 2-D arrays")
    if expected_width is not None and values.shape[1] != expected_width:
        raise ValueError(f"expected {expected_width} columns")
    if np.any(mask & ~np.isfinite(values)) or np.any((~mask) & ~np.isnan(values)):
        raise ValueError("source valid mask and NaN semantics disagree")
    return values, mask


def fit_reference(values, source_valid_mask):
    """Fit the frozen G3-A sorted duplicate reference, without labels or holdout data."""
    values, mask = _validate_matrix(values, source_valid_mask)
    return _fit_rank_reference(values, mask)


def rank_against_reference(values, source_valid_mask, reference):
    """Apply average-tie percentile ranks while preserving source/rank masks."""
    values, mask = _validate_matrix(values, source_valid_mask)
    percentile, rank_valid = _apply_rank_reference(values, mask, reference)
    if np.any(rank_valid & ~np.isfinite(percentile)) or np.any((~rank_valid) & ~np.isnan(percentile)):
        raise ValueError("rank output validity mismatch")
    return percentile, rank_valid


def quantize_rank256(percentile, rank_valid_mask):
    """Apply the frozen k=floor(255*u+0.5), q=k/255 transform."""
    percentile = np.asarray(percentile, dtype=np.float64); rank_valid_mask = np.asarray(rank_valid_mask, dtype=bool)
    return _quantize_rank(percentile, rank_valid_mask)


def build_loso_indices(sites, holdout_site: str) -> dict:
    sites = np.asarray(sites).astype(str)
    if sites.ndim != 1 or holdout_site not in HOLDOUTS:
        raise ValueError("invalid site vector or holdout")
    train_site = {"I0002": "S0001", "I0006": "S0001", "S0001": "I0006"}[holdout_site]
    unused_site = ({"I0002", "I0006", "S0001"} - {holdout_site, train_site}).pop()
    train = np.flatnonzero(sites == train_site).astype(np.int64)
    test = np.flatnonzero(sites == holdout_site).astype(np.int64)
    unused = np.flatnonzero(sites == unused_site).astype(np.int64)
    if not len(train) or not len(test) or not len(unused):
        raise ValueError("LOSO site has no rows")
    return {"holdout_site": holdout_site, "train_site": train_site, "unused_site": unused_site,
            "train_indices": train, "test_indices": test, "unused_indices": unused}


def build_experiment_selection(feature_ids: Sequence[str]) -> dict:
    ids = tuple(str(x) for x in feature_ids)
    if len(ids) != 1444 or len(set(ids)) != 1444:
        raise ValueError("experiment selection requires 1444 unique feature IDs")
    selections = {
        "E0": list(range(0, 807)),
        "E1": list(range(0, 1191)),
        "E2": list(range(0, 807)) + list(range(1191, 1444)),
        "E3": list(range(0, 1444)),
    }
    return {name: {"indices": indices, "feature_ids": [ids[i] for i in indices], "count": len(indices)} for name, indices in selections.items()}


def _normalize_view_rows(view_rows):
    rows = []
    for row in view_rows:
        try:
            item = {"view_id": str(row["view_id"]), "view_position": int(row["view_position"]),
                    "feature_id": str(row["feature_id"]), "unique_column_index": int(row["unique_column_index"]),
                    "source_block": str(row.get("source_block", ""))}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid view row") from exc
        rows.append(item)
    by_view = {view: [] for view in VIEW_IDS}
    for row in rows:
        if row["view_id"] not in by_view:
            raise ValueError("unknown view")
        by_view[row["view_id"]].append(row)
    if {view: len(by_view[view]) for view in VIEW_IDS} != VIEW_DIMS_V2:
        raise ValueError("v2 view dimensions mismatch")
    ordered = []
    for view in VIEW_IDS:
        by_view[view].sort(key=lambda item: item["view_position"])
        if [row["view_position"] for row in by_view[view]] != list(range(VIEW_DIMS_V2[view])):
            raise ValueError("view positions are not continuous")
        ordered.extend(by_view[view])
    return ordered


def map_ranked_views(values, rank_valid_mask, feature_ids: Sequence[str], view_rows):
    values = np.asarray(values, dtype=np.float64); rank_valid_mask = np.asarray(rank_valid_mask, dtype=bool)
    if values.ndim != 2 or rank_valid_mask.shape != values.shape:
        raise ValueError("ranked values/mask shape mismatch")
    ids = tuple(str(x) for x in feature_ids)
    if values.shape[1] != len(ids) or len(ids) != 1444 or len(set(ids)) != len(ids):
        raise ValueError("ranked feature ID contract mismatch")
    ordered = _normalize_view_rows(view_rows); mapped = {}
    for view in VIEW_IDS:
        rows = [row for row in ordered if row["view_id"] == view]
        indices = []
        for row in rows:
            idx = row["unique_column_index"]
            if idx < 0 or idx >= len(ids) or ids[idx] != row["feature_id"]:
                raise ValueError("view feature ID/index mismatch")
            indices.append(idx)
        mapped[view] = (values[:, indices], rank_valid_mask[:, indices])
    return mapped


def _reference_payload(reference, prefix: str) -> dict[str, np.ndarray]:
    pieces = [np.asarray(item, dtype=np.float64) for item in reference.sorted_values]
    offsets = [0]
    for piece in pieces:
        offsets.append(offsets[-1] + len(piece))
    return {
        f"{prefix}_reference_values": np.concatenate(pieces) if pieces else np.empty(0, dtype=np.float64),
        f"{prefix}_reference_offsets": np.asarray(offsets, dtype=np.int64),
        f"{prefix}_reference_counts": np.asarray(reference.counts, dtype=np.int64),
        f"{prefix}_reference_fingerprint": np.asarray(reference.fingerprint),
    }


def _reference_bounds(reference):
    lo = np.full(len(reference.sorted_values), np.nan, dtype=np.float64); hi = lo.copy()
    for index, values in enumerate(reference.sorted_values):
        if len(values): lo[index] = values[0]; hi[index] = values[-1]
    return lo, hi


def build_rank_unit(values, source_valid_mask, feature_ids, sites, holdout_site: str, protocol: str,
                    support_count=None, missing_reason_codes=None, row_keys=None) -> dict:
    values, source_valid_mask = _validate_matrix(values, source_valid_mask, expected_width=1444)
    ids = tuple(str(x) for x in feature_ids); sites = np.asarray(sites).astype(str)
    if len(ids) != 1444 or len(set(ids)) != 1444 or sites.shape != (len(values),):
        raise ValueError("pool identity mismatch")
    if protocol not in PROTOCOLS:
        raise ValueError("unknown protocol")
    fold = build_loso_indices(sites, holdout_site); train_idx = fold["train_indices"]; test_idx = fold["test_indices"]
    train_values, train_source = values[train_idx], source_valid_mask[train_idx]
    test_values, test_source = values[test_idx], source_valid_mask[test_idx]
    train_reference = fit_reference(train_values, train_source)
    train_percentile, train_rank = rank_against_reference(train_values, train_source, train_reference)
    if protocol == "transductive":
        test_reference = fit_reference(test_values, test_source)
    else:
        test_reference = train_reference
    test_percentile, test_rank = rank_against_reference(test_values, test_source, test_reference)
    train_quantized, train_levels = quantize_rank256(train_percentile, train_rank)
    test_quantized, test_levels = quantize_rank256(test_percentile, test_rank)
    result = {"fold": fold, "protocol": protocol, "feature_ids": ids,
              "train_percentile": train_percentile, "test_percentile": test_percentile,
              "train_source_valid_mask": train_source, "test_source_valid_mask": test_source,
              "train_rank_valid_mask": train_rank, "test_rank_valid_mask": test_rank,
              "train_quantized": train_quantized, "test_quantized": test_quantized,
              "train_levels": train_levels, "test_levels": test_levels,
              "train_reference": train_reference, "test_reference": test_reference,
              "train_values": train_values, "test_values": test_values}
    if support_count is not None:
        support_count = np.asarray(support_count, dtype=np.int64)
        if support_count.shape != values.shape or np.any(support_count < 0): raise ValueError("support_count mismatch")
        result["train_support_count"] = support_count[train_idx]; result["test_support_count"] = support_count[test_idx]
    if missing_reason_codes is not None:
        reasons = np.asarray(missing_reason_codes, dtype=np.int64)
        if reasons.shape != values.shape: raise ValueError("missing_reason_codes mismatch")
        result["train_missing_reason_codes"] = reasons[train_idx]; result["test_missing_reason_codes"] = reasons[test_idx]
    if row_keys is not None:
        keys = np.asarray(row_keys).astype(str)
        if keys.shape != (len(values),): raise ValueError("row_keys mismatch")
        result["train_record_keys"] = keys[train_idx]; result["test_record_keys"] = keys[test_idx]
    return result


def _unit_payload(unit, view_rows):
    payload = {key: value for key, value in unit.items() if key.endswith(("percentile", "valid_mask", "quantized", "levels", "support_count", "reason_codes", "record_keys"))}
    payload.update({"train_indices": unit["fold"]["train_indices"], "test_indices": unit["fold"]["test_indices"], "unused_indices": unit["fold"]["unused_indices"]})
    payload.update(_reference_payload(unit["train_reference"], "train"))
    if unit["protocol"] == "transductive": payload.update(_reference_payload(unit["test_reference"], "test"))
    else:
        payload.update(_reference_payload(unit["train_reference"], "test"))
    train_views = map_ranked_views(unit["train_quantized"], unit["train_rank_valid_mask"], unit["feature_ids"], view_rows)
    test_views = map_ranked_views(unit["test_quantized"], unit["test_rank_valid_mask"], unit["feature_ids"], view_rows)
    for view in VIEW_IDS:
        payload[f"train_{view}_values"], payload[f"train_{view}_rank_valid_mask"] = train_views[view]
        payload[f"test_{view}_values"], payload[f"test_{view}_rank_valid_mask"] = test_views[view]
    return payload


def _write_npz(path: Path, payload: Mapping[str, np.ndarray]):
    arrays = {str(key): np.asarray(value) for key, value in payload.items()}
    if any(value.dtype == object for value in arrays.values()): raise ValueError("object arrays are forbidden")
    with path.open("wb") as handle: np.savez_compressed(handle, **arrays); handle.flush(); os.fsync(handle.fileno())


def _file_summary(path: Path): return {"sha256": _sha_file(path), "size": path.stat().st_size}


def validate_unit(path: Path, expected_metadata: Mapping) -> bool:
    path = Path(path)
    try:
        complete = json.loads((path / "complete.json").read_text(encoding="utf-8")); metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if complete.get("status") != "COMPLETE" or metadata != json.loads(_canonical(expected_metadata)): return False
        files = complete.get("file_summaries", {})
        if set(files) != {"arrays.npz", "metadata.json"}: return False
        if any(_file_summary(path / name) != summary for name, summary in files.items()): return False
        with np.load(path / "arrays.npz", allow_pickle=False) as arrays:
            if any(np.asarray(arrays[name]).dtype == object for name in arrays.files): return False
        return True
    except Exception:
        return False


def publish_unit_atomic(path: Path, payload: Mapping[str, np.ndarray], metadata: Mapping) -> dict:
    path = Path(path); expected_metadata = json.loads(_canonical(metadata))
    if path.exists():
        if validate_unit(path, expected_metadata): return {"status": "SKIP_VERIFIED", "path": path}
        raise UnitPublishError(f"existing unit is invalid or has a different identity: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{path.name}.building-", dir=str(path.parent)))
    try:
        _write_npz(building / "arrays.npz", payload)
        (building / "metadata.json").write_text(_canonical(expected_metadata), encoding="utf-8")
        # Reload before publishing; this catches dtype/object and serialization drift.
        with np.load(building / "arrays.npz", allow_pickle=False) as arrays:
            for key, value in payload.items():
                loaded = np.asarray(arrays[key]); original = np.asarray(value)
                same = np.array_equal(loaded, original, equal_nan=True) if np.issubdtype(original.dtype, np.floating) else np.array_equal(loaded, original)
                if not same: raise UnitPublishError(f"array round-trip mismatch: {key}")
        complete = {"schema": SCHEMA, "status": "COMPLETE", "file_summaries": {name: _file_summary(building / name) for name in ("arrays.npz", "metadata.json")}}
        (building / "complete.json").write_text(_canonical(complete), encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists(): raise UnitPublishError(f"target appeared during publication: {path}")
        os.rename(building, path)
        if not validate_unit(path, expected_metadata): raise UnitPublishError("published unit failed validation")
        return {"status": "BUILD_NEW", "path": path}
    except Exception:
        if building.exists():
            for child in building.iterdir(): child.unlink()
            building.rmdir()
        raise


def _load_view_rows(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as handle: return [dict(row) for row in csv.DictReader(handle)]


def load_pool_input(pool_root: Path) -> dict:
    """Load only the already-published fix1 pool; never reads family caches or EDF."""
    root = Path(pool_root); completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
    if completion.get("status") != "COMPLETE": raise ValueError("pool is not complete")
    files = completion.get("file_summaries", {})
    for name, summary in files.items():
        if not (root / name).is_file() or _file_summary(root / name) != summary: raise ValueError(f"pool file mismatch: {name}")
    values = np.load(root / "X_raw.npy", allow_pickle=False, mmap_mode="r"); source = np.load(root / "valid_mask.npy", allow_pickle=False, mmap_mode="r"); support = np.load(root / "support_count.npy", allow_pickle=False, mmap_mode="r")
    values, source = _validate_matrix(values, source, expected_width=1444)
    if values.shape[0] != 6600: raise ValueError("pool row count")
    ids = tuple(str(x) for x in np.load(root / "unique_feature_ids.npy", allow_pickle=False).tolist())
    row_keys = tuple(str(x) for x in np.load(root / "row_record_keys.npy", allow_pickle=False).tolist()); sites = np.asarray(np.load(root / "row_sites.npy", allow_pickle=False)).astype(str)
    codes = np.load(root / "missing_reason_codes.npy", allow_pickle=False, mmap_mode="r")
    reasons = json.loads((root / "missing_reason_dictionary.json").read_text(encoding="utf-8"))
    view_rows = _load_view_rows(root / "view_manifest.csv")
    expected = {"values": np.asarray(values), "valid_mask": np.asarray(source), "support_count": np.asarray(support), "missing_reason_codes": np.asarray(codes), "missing_reason_dictionary": {int(k): v for k, v in reasons.items()}, "unique_feature_ids": ids, "row_keys": row_keys, "groups": tuple(sites.tolist()), "view_rows": view_rows}
    if completion.get("matrix_digest") and _pool_digest(expected) != completion["matrix_digest"]: raise ValueError("pool matrix digest mismatch")
    if len(ids) != 1444 or len(row_keys) != 6600 or sites.shape != (6600,): raise ValueError("pool identity shape")
    return {"root": root, "values": values, "source_valid_mask": source, "support_count": support, "missing_reason_codes": codes, "missing_reason_dictionary": {int(k): v for k, v in reasons.items()}, "feature_ids": ids, "row_keys": row_keys, "sites": sites, "view_rows": view_rows, "completion": completion}


def build_units_from_pool(pool: Mapping, output_root: Path) -> list[dict]:
    values = np.asarray(pool["values"]); source = np.asarray(pool["source_valid_mask"]); sites = np.asarray(pool["sites"]).astype(str); ids = tuple(pool["feature_ids"])
    selections = build_experiment_selection(ids); pool_digest = pool["completion"].get("matrix_digest", "")
    results = []
    for holdout in HOLDOUTS:
        for protocol in PROTOCOLS:
            unit = build_rank_unit(values, source, ids, sites, holdout, protocol, pool.get("support_count"), pool.get("missing_reason_codes"), pool.get("row_keys"))
            fold = unit["fold"]; payload = _unit_payload(unit, pool["view_rows"])
            metadata = {"schema": SCHEMA, "holdout_site": holdout, "train_site": fold["train_site"], "unused_site": fold["unused_site"], "protocol": protocol, "train_count": len(fold["train_indices"]), "test_count": len(fold["test_indices"]), "unused_count": len(fold["unused_indices"]), "feature_count": 1444, "feature_id_sha256": _sha_bytes(_canonical(list(ids)).encode()), "pool_matrix_digest": pool_digest, "formula": "(L+0.5E)/n", "levels": 256, "rounding": "half_up", "numpy_version": np.__version__, "python_version": platform.python_version(), "experiment_selection": selections, "view_dimensions": VIEW_DIMS_V2, "view_manifest_sha256": _sha_bytes(_canonical(pool["view_rows"]).encode()), "source_valid_mask_field": "valid_mask", "support_field": "support_count", "missing_reason_field": "missing_reason_codes"}
            result = publish_unit_atomic(Path(output_root) / f"holdout_{holdout}" / protocol, payload, metadata)
            results.append({"holdout": holdout, "protocol": protocol, **result, "train_source_valid": int(unit["train_source_valid_mask"].sum()), "test_source_valid": int(unit["test_source_valid_mask"].sum()), "train_rank_valid": int(unit["train_rank_valid_mask"].sum()), "test_rank_valid": int(unit["test_rank_valid_mask"].sum())})
    return results


def compare_old807_unit(v2_unit_path: Path, old_unit_path: Path, old_view_sizes: Mapping[str, int] = VIEW_DIMS_V1) -> dict:
    """Compare v2's freshly computed first 807 columns with an immutable G3-A unit."""
    errors = []
    v2 = Path(v2_unit_path)
    old = Path(old_unit_path)
    with (np.load(v2 / "arrays.npz", allow_pickle=False) as nv,
          np.load(old / "indices.npz", allow_pickle=False) as oi,
          np.load(old / "percentile.npz", allow_pickle=False) as op,
          np.load(old / "quantized.npz", allow_pickle=False) as oq,
          np.load(old / "views.npz", allow_pickle=False) as ov,
          np.load(old / "reference.npz", allow_pickle=False) as oreference):
        for key in ("train_indices", "test_indices", "unused_indices"):
            if not np.array_equal(np.asarray(nv[key]), np.asarray(oi[key])):
                errors.append(f"indices:{key}")
        for role in ("train", "test"):
            for key in ("percentile", "source_valid_mask", "rank_valid_mask"):
                a = np.asarray(nv[f"{role}_{key}"])[:, :807]
                b = np.asarray(op[f"{role}_{key}"])
                if not np.array_equal(a, b, equal_nan=True):
                    errors.append(f"percentile:{role}_{key}")
            for key in ("levels", "quantized"):
                a = np.asarray(nv[f"{role}_{key}"])[:, :807]
                b = np.asarray(oq[f"{role}_{key}"])
                if not np.array_equal(a, b, equal_nan=True):
                    errors.append(f"quantized:{role}_{key}")
            reference_role = role if f"{role}_counts" in oreference.files else "train"
            a = np.asarray(nv[f"{role}_reference_counts"])[:807]
            b = np.asarray(oreference[f"{reference_role}_counts"])
            if not np.array_equal(a, b):
                errors.append(f"reference:{role}_counts")
        for view, width in old_view_sizes.items():
            for role in ("train", "test"):
                for suffix in ("values", "rank_valid_mask"):
                    a = np.asarray(nv[f"{role}_{view}_{suffix}"])[:, :width]
                    b = np.asarray(ov[f"{role}_{view}_{suffix}"])
                    if not np.array_equal(a, b, equal_nan=True):
                        errors.append(f"view:{role}_{view}_{suffix}")
    return {"passed": not errors, "errors": errors}
