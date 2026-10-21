"""Private 1444-column Batch1 candidate-pool contracts and safe publication."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .pool_sources import Alignment, read_aligned_record

FAMILY_ORDER = ("eeg_tail", "eeg_temporal", "eeg_coherence_stage", "caisr_stage", "emg_stage", "resp_stage")
BASE_VIEW_DIMS = {"V1": 113, "V2": 129, "V3": 349, "V4": 113, "V5": 123}
NEW_VIEW_COUNTS = {"V1": 48, "V2": 48, "V3": 144, "V4": 144, "V5": 253}
VIEW_IDS = ("V1", "V2", "V3", "V4", "V5")


class PoolPublishError(RuntimeError):
    """A complete or incomplete target cannot be safely reused or replaced."""


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_reason_dictionary(dictionary: Mapping) -> dict[int, str]:
    """Accept only a reversible code->reason mapping (JSON numeric keys allowed)."""
    out: dict[int, str] = {}
    for raw_code, raw_reason in dictionary.items():
        if isinstance(raw_code, bool):
            raise ValueError("reason code must be an integer")
        try:
            code = int(raw_code)
        except (TypeError, ValueError) as exc:
            raise ValueError("reason dictionary must use code->reason keys") from exc
        if str(raw_code).strip() not in {str(code), f"+{code}"} and not isinstance(raw_code, int):
            raise ValueError("reason dictionary has a non-numeric key")
        reason = str(raw_reason)
        if not reason:
            raise ValueError("reason labels cannot be empty")
        if code in out and out[code] != reason:
            raise ValueError("conflicting reason code")
        out[code] = reason
    if out.get(0) != "VALID" or out.get(1) != "OLD_SOURCE_MISSING":
        raise ValueError("reason codebook must reserve 0=VALID and 1=OLD_SOURCE_MISSING")
    if len(set(out.values())) != len(out):
        raise ValueError("reason dictionary maps multiple codes to one reason")
    return dict(sorted(out.items()))


def encode_reason_matrix(valid_mask, missing_reason, dictionary: Mapping | None = None):
    """Encode a reason matrix with deterministic, serializable code->reason semantics."""
    valid = np.asarray(valid_mask, dtype=bool)
    reasons = np.asarray(missing_reason, dtype=str)
    if valid.shape != reasons.shape or valid.ndim != 2:
        raise ValueError("valid_mask and missing_reason must be matching 2-D arrays")
    if dictionary is None:
        codebook = {0: "VALID", 1: "OLD_SOURCE_MISSING"}
    else:
        codebook = _normalize_reason_dictionary(dictionary)
    labels = set()
    for value, ok in zip(reasons.reshape(-1), valid.reshape(-1)):
        label = str(value)
        if ok:
            if label:
                raise ValueError("valid positions must have an empty missing_reason")
        else:
            if not label:
                raise ValueError("invalid positions require a non-empty missing_reason")
            labels.add(label)
    for label in sorted(labels):
        if label not in codebook.values():
            codebook[max(codebook) + 1] = label
    inverse = {label: code for code, label in codebook.items()}
    codes = np.zeros(valid.shape, dtype=np.int64)
    for index in np.ndindex(valid.shape):
        if not valid[index]:
            codes[index] = inverse[str(reasons[index])]
    return codes, dict(sorted(codebook.items()))


def decode_reason_codes(codes, valid_mask, dictionary: Mapping) -> np.ndarray:
    """Decode and validate code->reason payloads; never fall back to VALID."""
    code_arr = np.asarray(codes, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool)
    codebook = _normalize_reason_dictionary(dictionary)
    if code_arr.shape != valid.shape or code_arr.ndim != 2:
        raise ValueError("codes and valid_mask must be matching 2-D arrays")
    labels = np.empty(valid.shape, dtype="<U128")
    for index in np.ndindex(valid.shape):
        code = int(code_arr[index])
        if code not in codebook:
            raise ValueError(f"unknown reason code: {code}")
        if valid[index] and code != 0:
            raise ValueError("valid position is not encoded as VALID")
        if not valid[index] and code == 0:
            raise ValueError("missing position is encoded as VALID")
        labels[index] = codebook[code] if not valid[index] else ""
    return labels


def build_column_contract(old_feature_ids: Sequence[str], family_feature_ids: Mapping[str, Sequence[str]]):
    old = tuple(str(x) for x in old_feature_ids)
    if len(old) != 807 or len(set(old)) != 807:
        raise ValueError("old feature contract must contain 807 unique IDs")
    ordered = []
    for family in FAMILY_ORDER:
        if family not in family_feature_ids:
            raise ValueError(f"missing family: {family}")
        ids = tuple(str(x) for x in family_feature_ids[family])
        ordered.extend(ids)
    all_ids = old + tuple(ordered)
    if len(all_ids) != 1444 or len(set(all_ids)) != 1444:
        raise ValueError("v2 feature IDs must be globally unique")
    return {
        "unique_feature_ids": all_ids,
        "old_feature_ids": old,
        "family_feature_ids": {f: tuple(str(x) for x in family_feature_ids[f]) for f in FAMILY_ORDER},
        "column_index": {fid: i for i, fid in enumerate(all_ids)},
        "experiment_sets": {"E0": 807, "E1": 1191, "E2": 1060, "E3": 1444},
    }


def _old_rows_from_contract(contract, old_view_rows):
    if old_view_rows is None or not list(old_view_rows):
        raise ValueError("verified old view manifest is required")
    by_view = {view: [] for view in VIEW_IDS}
    for row in old_view_rows:
        view = str(row.get("view_id", ""))
        fid = str(row.get("unique_feature_id", row.get("feature_id", "")))
        if view not in by_view or fid not in contract["old_feature_ids"]:
            raise ValueError("invalid old view manifest row")
        by_view[view].append({
            "view_id": view,
            "view_position": int(row["view_position"]),
            "feature_id": fid,
            "unique_column_index": contract["column_index"][fid],
            "source_block": "old807",
        })
    for view in VIEW_IDS:
        rows = sorted(by_view[view], key=lambda item: item["view_position"])
        if len(rows) != BASE_VIEW_DIMS[view] or [r["view_position"] for r in rows] != list(range(len(rows))):
            raise ValueError(f"invalid old view positions for {view}")
        if any(r["unique_column_index"] >= 807 for r in rows) or len({r["feature_id"] for r in rows}) != len(rows):
            raise ValueError(f"invalid old view columns for {view}")
        by_view[view] = rows
    if len({r["feature_id"] for view in VIEW_IDS for r in by_view[view]}) != 807:
        raise ValueError("old view manifest must cover all 807 IDs exactly once")
    return by_view


def _new_rows(contract, new_manifest_rows):
    if new_manifest_rows is None or not list(new_manifest_rows):
        raise ValueError("verified new feature manifest is required")
    by_view = {view: [] for view in VIEW_IDS}
    expected_ids = {fid: family for family, ids in contract["family_feature_ids"].items() for fid in ids}
    seen = set()
    for row in new_manifest_rows:
        view = str(row.get("proposed_view", "")); fid = str(row.get("feature_id", "")); family = str(row.get("family_id", ""))
        if view not in by_view or fid not in expected_ids or expected_ids[fid] != family or fid in seen:
            raise ValueError("invalid new feature manifest row")
        seen.add(fid)
        by_view[view].append((contract["column_index"][fid], fid, family))
    if seen != set(expected_ids):
        raise ValueError("new feature manifest must cover all 637 IDs exactly once")
    return by_view


def build_view_manifest(contract, old_view_rows, new_manifest_rows=None):
    """Build a view map only from explicit verified manifests; no dimension fallback."""
    rows = _old_rows_from_contract(contract, old_view_rows)
    additions = _new_rows(contract, new_manifest_rows)
    for view in VIEW_IDS:
        start = len(rows[view])
        for i, (idx, fid, family) in enumerate(additions[view]):
            rows[view].append({"view_id": view, "view_position": start + i, "feature_id": fid,
                               "unique_column_index": idx, "source_block": family})
    expected = {v: BASE_VIEW_DIMS[v] + NEW_VIEW_COUNTS[v] for v in VIEW_IDS}
    if {v: len(rows[v]) for v in VIEW_IDS} != expected:
        raise ValueError(f"Unexpected v2 view dimensions: { {v: len(rows[v]) for v in VIEW_IDS} }")
    for view in VIEW_IDS:
        positions = [row["view_position"] for row in rows[view]]
        if positions != list(range(len(rows[view]))):
            raise ValueError(f"non-contiguous view positions: {view}")
        for row in rows[view]:
            if contract["unique_feature_ids"][row["unique_column_index"]] != row["feature_id"]:
                raise ValueError("view feature ID/index mismatch")
    return rows


def _as_matrix(array, n, f, name, dtype):
    arr = np.asarray(array, dtype=dtype)
    if arr.ndim == 1 and n == 1:
        arr = arr.reshape(1, -1)
    if arr.shape != (n, f):
        raise ValueError(f"{name} shape {arr.shape} != {(n, f)}")
    return arr


def assemble_arrays(old_values, old_valid, old_support, family_blocks, old_feature_ids, family_feature_ids):
    old_values = np.asarray(old_values, dtype=np.float64)
    old_valid = np.asarray(old_valid, dtype=bool)
    old_support = np.asarray(old_support, dtype=np.int64)
    if old_values.ndim == 1:
        old_values = old_values.reshape(1, -1); old_valid = old_valid.reshape(1, -1); old_support = old_support.reshape(1, -1)
    n = old_values.shape[0]; old_ids = tuple(old_feature_ids)
    if old_values.shape[1] != len(old_ids):
        raise ValueError("old feature length mismatch")
    blocks = family_blocks if isinstance(family_blocks, Mapping) else {f: b for f, b in zip(FAMILY_ORDER, family_blocks)}
    family_ids = family_feature_ids if isinstance(family_feature_ids, Mapping) else {f: ids for f, ids in zip(FAMILY_ORDER, family_feature_ids)}
    vals = [old_values]; masks = [old_valid]; supports = [old_support]
    reasons = [np.where(old_valid, "", "OLD_SOURCE_MISSING").astype("<U64")]; ids = [old_ids]
    for family in FAMILY_ORDER:
        if family not in blocks:
            raise ValueError(f"missing family block: {family}")
        fids = tuple(family_ids[family]); block = blocks[family]
        v = _as_matrix(block["values"], n, len(fids), "values", np.float64)
        m = _as_matrix(block["valid_mask"], n, len(fids), "valid_mask", bool)
        s = _as_matrix(block["support_count"], n, len(fids), "support_count", np.int64)
        if np.any(m & ~np.isfinite(v)) or np.any((~m) & ~np.isnan(v)):
            raise ValueError(f"{family} values/mask disagreement")
        if np.any(s < 0):
            raise ValueError(f"{family} support negative")
        if "missing_reason" not in block:
            raise ValueError(f"{family} missing_reason is required")
        reason_arr = np.asarray(block["missing_reason"], dtype=str)
        if reason_arr.ndim == 1 and n == 1:
            reason_arr = reason_arr.reshape(1, -1)
        if reason_arr.shape != (n, len(fids)):
            raise ValueError(f"{family} missing_reason shape mismatch")
        for value, ok in zip(reason_arr.reshape(-1), m.reshape(-1)):
            if ok and str(value):
                raise ValueError(f"{family} valid position has missing_reason")
            if not ok and not str(value):
                raise ValueError(f"{family} missing position has empty missing_reason")
        vals.append(v); masks.append(m); supports.append(s); reasons.append(reason_arr.astype("<U128")); ids.append(fids)
    all_values = np.concatenate(vals, axis=1); all_valid = np.concatenate(masks, axis=1)
    all_support = np.concatenate(supports, axis=1); all_reasons = np.concatenate(reasons, axis=1)
    codes, dictionary = encode_reason_matrix(all_valid, all_reasons)
    result = {"values": all_values, "valid_mask": all_valid, "support_count": all_support,
              "missing_reason_codes": codes, "missing_reason_dictionary": dictionary,
              "missing_reasons": all_reasons, "unique_feature_ids": tuple(x for group in ids for x in group)}
    if result["values"].shape[1] != 1444:
        raise ValueError("assembled pool must have 1444 columns")
    return result


def _array_digest(array) -> str:
    arr = np.asarray(array)
    h = hashlib.sha256(); h.update(str(arr.dtype).encode()); h.update(_canonical(list(arr.shape)).encode()); h.update(np.ascontiguousarray(arr).tobytes()); return h.hexdigest()


def _normalized_view_rows(rows):
    return [{"view_id": str(row["view_id"]), "view_position": int(row["view_position"]),
             "feature_id": str(row["feature_id"]), "unique_column_index": int(row["unique_column_index"]),
             "source_block": str(row["source_block"])} for row in rows]


def _pool_digest(arrays) -> str:
    h = hashlib.sha256()
    for name in ("values", "valid_mask", "support_count", "missing_reason_codes", "unique_feature_ids", "row_keys", "groups"):
        value = arrays[name]
        if name in {"unique_feature_ids", "row_keys", "groups"}:
            h.update(name.encode()); h.update(_canonical([str(x) for x in value]).encode())
        else:
            h.update(name.encode()); h.update(_array_digest(value).encode())
    h.update(_canonical({str(k): str(v) for k, v in sorted(arrays["missing_reason_dictionary"].items(), key=lambda x: int(x[0]))}).encode())
    h.update(_canonical(_normalized_view_rows(arrays.get("view_rows", []))).encode())
    return h.hexdigest()


def _validate_arrays_for_publish(arrays):
    values = np.asarray(arrays["values"], dtype=np.float64); valid = np.asarray(arrays["valid_mask"], dtype=bool)
    support = np.asarray(arrays["support_count"], dtype=np.int64); codes = np.asarray(arrays["missing_reason_codes"], dtype=np.int64)
    if values.ndim != 2 or valid.shape != values.shape or support.shape != values.shape or codes.shape != values.shape:
        raise PoolPublishError("pool arrays must have identical 2-D shapes")
    if np.any(valid & ~np.isfinite(values)) or np.any((~valid) & ~np.isnan(values)) or np.any(support < 0):
        raise PoolPublishError("pool array invariants failed")
    dictionary = _normalize_reason_dictionary(arrays["missing_reason_dictionary"])
    decode_reason_codes(codes, valid, dictionary)
    fids = tuple(str(x) for x in arrays["unique_feature_ids"])
    row_keys = tuple(str(x) for x in arrays["row_keys"]); groups = tuple(str(x) for x in arrays["groups"])
    if len(fids) != values.shape[1] or len(set(fids)) != len(fids) or len(row_keys) != values.shape[0] or len(groups) != values.shape[0]:
        raise PoolPublishError("pool identity lengths do not match arrays")
    return {"values": values, "valid_mask": valid, "support_count": support, "missing_reason_codes": codes,
            "missing_reason_dictionary": dictionary, "unique_feature_ids": fids, "row_keys": row_keys, "groups": groups,
            "view_rows": list(arrays.get("view_rows", []))}


def _file_summary(path: Path) -> dict:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest, "size": path.stat().st_size}


def _write_view_manifest(path: Path, rows):
    fields = ("view_id", "view_position", "feature_id", "unique_column_index", "source_block")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def _read_view_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _verify_published_target(target: Path, expected, identity: Mapping):
    try:
        completion = json.loads((target / "completion.json").read_text(encoding="utf-8"))
        if completion.get("status") != "COMPLETE" or completion.get("input_identity") != json.loads(_canonical(identity)):
            raise PoolPublishError("existing pool identity mismatch")
        if completion.get("matrix_digest") != _pool_digest(expected):
            raise PoolPublishError("existing pool matrix digest mismatch")
        for name, summary in completion.get("file_summaries", {}).items():
            if _file_summary(target / name) != summary:
                raise PoolPublishError(f"existing pool file digest mismatch: {name}")
        loaded = {
            "values": np.load(target / "X_raw.npy", allow_pickle=False),
            "valid_mask": np.load(target / "valid_mask.npy", allow_pickle=False),
            "support_count": np.load(target / "support_count.npy", allow_pickle=False),
            "missing_reason_codes": np.load(target / "missing_reason_codes.npy", allow_pickle=False),
            "missing_reason_dictionary": json.loads((target / "missing_reason_dictionary.json").read_text(encoding="utf-8")),
            "unique_feature_ids": tuple(np.load(target / "unique_feature_ids.npy", allow_pickle=False).tolist()),
            "row_keys": tuple(np.load(target / "row_record_keys.npy", allow_pickle=False).tolist()),
            "groups": tuple(np.load(target / "row_sites.npy", allow_pickle=False).tolist()),
            "view_rows": _read_view_manifest(target / "view_manifest.csv"),
        }
        loaded = _validate_arrays_for_publish(loaded)
        if _pool_digest(loaded) != _pool_digest(expected):
            raise PoolPublishError("existing pool reload digest mismatch")
    except PoolPublishError:
        raise
    except Exception as exc:
        raise PoolPublishError("existing pool cannot be verified") from exc


def write_candidate_pool_atomic(arrays: Mapping, output_root: Path, input_identity: Mapping) -> dict:
    """Build in a private temporary directory and publish once, never clobbering a target."""
    expected = _validate_arrays_for_publish(arrays); target = Path(output_root)
    if target.exists():
        if not target.is_dir() or not (target / "completion.json").exists():
            raise PoolPublishError("existing target is incomplete or not a pool")
        _verify_published_target(target, expected, input_identity)
        return {"status": "SKIP_VERIFIED", "output_root": target, "matrix_digest": _pool_digest(expected)}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(target.parent)))
    try:
        np.save(temp / "X_raw.npy", expected["values"]); np.save(temp / "valid_mask.npy", expected["valid_mask"])
        np.save(temp / "support_count.npy", expected["support_count"]); np.save(temp / "missing_reason_codes.npy", expected["missing_reason_codes"])
        np.save(temp / "unique_feature_ids.npy", np.asarray(expected["unique_feature_ids"], dtype=np.str_))
        np.save(temp / "row_record_keys.npy", np.asarray(expected["row_keys"], dtype=np.str_)); np.save(temp / "row_sites.npy", np.asarray(expected["groups"], dtype=np.str_))
        (temp / "missing_reason_dictionary.json").write_text(_canonical({str(k): v for k, v in expected["missing_reason_dictionary"].items()}), encoding="utf-8")
        _write_view_manifest(temp / "view_manifest.csv", expected["view_rows"])
        reloaded = {"values": np.load(temp / "X_raw.npy", allow_pickle=False), "valid_mask": np.load(temp / "valid_mask.npy", allow_pickle=False),
                    "support_count": np.load(temp / "support_count.npy", allow_pickle=False), "missing_reason_codes": np.load(temp / "missing_reason_codes.npy", allow_pickle=False),
                    "missing_reason_dictionary": json.loads((temp / "missing_reason_dictionary.json").read_text(encoding="utf-8")),
                    "unique_feature_ids": tuple(np.load(temp / "unique_feature_ids.npy", allow_pickle=False).tolist()),
                    "row_keys": tuple(np.load(temp / "row_record_keys.npy", allow_pickle=False).tolist()), "groups": tuple(np.load(temp / "row_sites.npy", allow_pickle=False).tolist()), "view_rows": expected["view_rows"]}
        reloaded = _validate_arrays_for_publish(reloaded)
        if _pool_digest(reloaded) != _pool_digest(expected):
            raise PoolPublishError("post-write pool digest mismatch")
        files = ["X_raw.npy", "valid_mask.npy", "support_count.npy", "missing_reason_codes.npy", "missing_reason_dictionary.json", "unique_feature_ids.npy", "row_record_keys.npy", "row_sites.npy", "view_manifest.csv"]
        completion = {"status": "COMPLETE", "schema": "featurelab_v2_candidate_pool_v2", "rows": int(expected["values"].shape[0]), "columns": int(expected["values"].shape[1]),
                      "feature_ids": list(expected["unique_feature_ids"]), "row_keys_digest": hashlib.sha256(_canonical(list(expected["row_keys"])).encode()).hexdigest(),
                      "source_identity": json.loads(_canonical(input_identity)), "input_identity": json.loads(_canonical(input_identity)),
                      "matrix_digest": _pool_digest(expected), "file_summaries": {name: _file_summary(temp / name) for name in files}}
        (temp / "completion.json").write_text(json.dumps(completion, sort_keys=True, indent=2), encoding="utf-8")
        if target.exists():
            raise PoolPublishError("target appeared during publication")
        os.replace(temp, target)
        return {"status": "BUILD_NEW", "output_root": target, "matrix_digest": completion["matrix_digest"]}
    except Exception:
        if temp.exists():
            for child in temp.iterdir():
                child.unlink()
            temp.rmdir()
        raise


def assemble_candidate_pool(alignment: Alignment, output_root: Path, view_rows=None, experiment_manifest=None):
    if view_rows is None or experiment_manifest is None or not experiment_manifest.get("new_manifest_rows"):
        raise ValueError("production assembly requires explicit verified old and new manifests")
    n = len(alignment.entries); contract = build_column_contract(alignment.old_feature_ids, alignment.family_feature_ids)
    values = np.empty((n, 1444), dtype=np.float64); valid = np.empty((n, 1444), dtype=bool); support = np.empty((n, 1444), dtype=np.int64); reasons = np.empty((n, 1444), dtype="<U128")
    row_keys = []; groups = []
    for i, entry in enumerate(alignment.entries):
        block = read_aligned_record(entry, alignment)
        assembled = assemble_arrays(block["old_values"], block["old_valid_mask"], block["old_support_count"], block["family_blocks"], alignment.old_feature_ids, alignment.family_feature_ids)
        if not np.array_equal(assembled["values"], np.concatenate([np.asarray(block["old_values"], dtype=np.float64).reshape(1, -1)] + [np.asarray(block["family_blocks"][f]["values"], dtype=np.float64).reshape(1, -1) for f in FAMILY_ORDER], axis=1), equal_nan=True):
            raise ValueError(f"source equality failure at row {i}")
        values[i] = assembled["values"][0]; valid[i] = assembled["valid_mask"][0]; support[i] = assembled["support_count"][0]; reasons[i] = assembled["missing_reasons"][0]
        row_keys.append(entry.record_key); groups.append(entry.site_id)
    codes, dictionary = encode_reason_matrix(valid, reasons)
    views = build_view_manifest(contract, view_rows, experiment_manifest["new_manifest_rows"])
    arrays = {"values": values, "valid_mask": valid, "support_count": support, "missing_reason_codes": codes, "missing_reason_dictionary": dictionary,
              "unique_feature_ids": contract["unique_feature_ids"], "row_keys": tuple(row_keys), "groups": tuple(groups),
              "view_rows": [row for view in VIEW_IDS for row in views[view]]}
    identity = {"source_identity_order": "old_g2b1_index", "rows": n, "recipes": dict(alignment.recipe_ids), "source_index": "verified_alignment"}
    result = write_candidate_pool_atomic(arrays, output_root, identity)
    result.update({"contract": contract, "views": views, "values": values, "valid_mask": valid, "support_count": support, "missing_reason_codes": codes, "reason_dictionary": dictionary, "row_keys": row_keys, "groups": groups})
    return result
