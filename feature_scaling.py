#!/usr/bin/env python
"""Shared, column-aware input preprocessing for training and inference.

typed_v1 fixes the destructive raw [-50, 50] clipping while preserving the
483/12/196 feature layout.  It never changes sequence lengths or ECG alignment.
legacy_clip exactly reproduces the historical nan_to_num + clip path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SEQ_DIM = 483
ECG_DIM = 12
STATIC_DIM = 196
ECG_EPOCH_OFFSET = 10
EXPECTED_DIMS = {"X_seq": SEQ_DIM, "X_ecg": ECG_DIM, "x_static": STATIC_DIM}
SCALER_VERSION = "typed_v1"
LEGACY_MODE = "legacy_clip"
DEFAULT_MASK_CONFIG = {
    "x_static_zero_indices": [],
    "x_seq_zero_ranges": [],
}
ROOT = Path(__file__).resolve().parent
DEFAULT_RULES_PATH = ROOT / "feat_input" / "feature_rules_v1.json"


class FeatureScalingError(RuntimeError):
    """Raised when preprocessing state, schema, or input values are invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_key(record: Mapping[str, Any]) -> str:
    if "record_id" in record:
        return str(record["record_id"])
    bids = record.get("BidsFolder", record.get("bids_folder"))
    session = record.get("SessionID", record.get("session_id"))
    if bids is None or session is None:
        raise FeatureScalingError("Training record lacks BidsFolder/SessionID")
    return f"{bids}_ses-{session}"


def load_feature_rules(path: str | Path = DEFAULT_RULES_PATH) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FeatureScalingError(f"Feature rule file not found: {path}")
    rules = json.loads(path.read_text(encoding="utf-8"))
    validate_feature_rules(rules)
    return rules


def validate_feature_rules(rules: Sequence[Mapping[str, Any]]) -> None:
    if len(rules) != sum(EXPECTED_DIMS.values()):
        raise FeatureScalingError(f"Expected 691 feature rules, found {len(rules)}")
    seen = set()
    by_branch = {branch: [] for branch in EXPECTED_DIMS}
    required = {
        "branch", "index", "name", "family", "formula_source", "formula_summary",
        "original_unit", "unit_scale", "nonlinear", "reference_unit",
        "scaling", "expected_range", "missing_rule", "zero_semantics",
        "audit_source", "source_uncertainty",
    }
    for rule in rules:
        missing = required.difference(rule)
        if missing:
            raise FeatureScalingError(f"Rule missing fields {sorted(missing)}: {rule}")
        branch, index = rule["branch"], int(rule["index"])
        if branch not in EXPECTED_DIMS:
            raise FeatureScalingError(f"Unknown branch {branch}")
        key = (branch, index)
        if key in seen:
            raise FeatureScalingError(f"Duplicate feature rule {key}")
        seen.add(key)
        by_branch[branch].append(index)
        if rule["nonlinear"] not in {"identity", "log1p"}:
            raise FeatureScalingError(f"Unsupported nonlinear transform in {key}")
        if rule["scaling"] not in {"identity", "robust"}:
            raise FeatureScalingError(f"Unsupported scaling strategy in {key}")
        if not np.isfinite(float(rule["unit_scale"])) or float(rule["unit_scale"]) <= 0:
            raise FeatureScalingError(f"Invalid unit_scale in {key}")
    for branch, dim in EXPECTED_DIMS.items():
        if sorted(by_branch[branch]) != list(range(dim)):
            raise FeatureScalingError(f"Rules for {branch} do not exactly cover 0:{dim}")


def feature_order_hash(rules: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash([(r["branch"], int(r["index"]), r["name"]) for r in rules])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return math.nan
    x, w = values[valid], weights[valid]
    order = np.argsort(x, kind="mergesort")
    x, w = x[order], w[order]
    positions = (np.cumsum(w) - 0.5 * w) / np.sum(w)
    return float(np.interp(float(q), positions, x))


def _stable_indices(record_id: str, branch: str, length: int, cap: int, seed: int) -> np.ndarray:
    if length <= cap:
        return np.arange(length, dtype=np.int64)
    payload = f"{seed}:{record_id}:{branch}".encode()
    derived = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    rng = np.random.default_rng(derived)
    return np.sort(rng.choice(length, size=cap, replace=False))


def _rule_range(rule: Mapping[str, Any]) -> tuple[float | None, float | None]:
    expected = rule.get("expected_range")
    if not isinstance(expected, Mapping) or not expected.get("hard", False):
        return None, None
    return expected.get("min"), expected.get("max")


def _prepare_values(values: np.ndarray, rule: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict]:
    x = np.asarray(values, dtype=np.float64).copy()
    valid = np.isfinite(x)
    diagnostics = {
        "nonfinite": int((~valid).sum()),
        "invalid_missing_rule": 0,
        "range_invalid": 0,
        "roundoff_corrected": 0,
        "negative_for_log": 0,
    }
    if rule["missing_rule"] == "nonfinite_or_nonpositive":
        bad = valid & (x <= 0)
        diagnostics["invalid_missing_rule"] = int(bad.sum())
        valid[bad] = False

    lower, upper = _rule_range(rule)
    tolerance = float(rule.get("range_tolerance", 1e-6))
    if lower is not None:
        near = valid & (x < lower) & (x >= lower - tolerance)
        bad = valid & (x < lower - tolerance)
        x[near] = lower
        valid[bad] = False
        diagnostics["roundoff_corrected"] += int(near.sum())
        diagnostics["range_invalid"] += int(bad.sum())
    if upper is not None:
        near = valid & (x > upper) & (x <= upper + tolerance)
        bad = valid & (x > upper + tolerance)
        x[near] = upper
        valid[bad] = False
        diagnostics["roundoff_corrected"] += int(near.sum())
        diagnostics["range_invalid"] += int(bad.sum())

    transformed = np.full(x.shape, np.nan, dtype=np.float64)
    u = x[valid] / float(rule["unit_scale"])
    if rule["nonlinear"] == "log1p":
        negative = u < 0
        diagnostics["negative_for_log"] = int(negative.sum())
        if np.any(negative):
            valid_indices = np.flatnonzero(valid)
            valid[valid_indices[negative]] = False
            u = u[~negative]
        transformed[valid] = np.log1p(u)
    else:
        transformed[valid] = u
    valid &= np.isfinite(transformed)
    return transformed, valid, diagnostics


def normalize_mask_config(config: Mapping[str, Any] | None = None) -> dict:
    """Validate and canonicalize post-transform feature masking."""
    config = dict(config or {})
    unknown = set(config).difference(DEFAULT_MASK_CONFIG)
    if unknown:
        raise FeatureScalingError(f"Unsupported input mask fields: {sorted(unknown)}")

    static_indices = sorted({int(index) for index in config.get("x_static_zero_indices", [])})
    for index in static_indices:
        if index < 0 or index >= STATIC_DIM:
            raise FeatureScalingError(f"x_static mask index out of range: {index}")

    seq_ranges = []
    for item in config.get("x_seq_zero_ranges", []):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
            raise FeatureScalingError(f"Invalid X_seq mask range: {item!r}")
        start, stop = int(item[0]), int(item[1])
        if start < 0 or stop > SEQ_DIM or start >= stop:
            raise FeatureScalingError(f"X_seq mask range out of bounds: [{start}, {stop})")
        seq_ranges.append([start, stop])
    seq_ranges.sort()
    for previous, current in zip(seq_ranges, seq_ranges[1:]):
        if current[0] < previous[1]:
            raise FeatureScalingError(f"Overlapping X_seq mask ranges: {previous}, {current}")

    return {
        "x_static_zero_indices": static_indices,
        "x_seq_zero_ranges": seq_ranges,
    }


class FeatureScaler:
    """Frozen shared transformer for the three model input branches."""

    def __init__(
        self,
        mode: str,
        rules: Sequence[Mapping[str, Any]] | None = None,
        parameters: Sequence[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
        clip_z: float | None = None,
        mask_config: Mapping[str, Any] | None = None,
    ):
        if mode not in {SCALER_VERSION, LEGACY_MODE}:
            raise FeatureScalingError(f"Unsupported input preprocessing mode: {mode}")
        self.mode = mode
        self.rules = [dict(r) for r in (rules or [])]
        self.parameters = [dict(p) for p in (parameters or [])]
        self.metadata = dict(metadata or {})
        self.clip_z = None if clip_z is None else float(clip_z)
        if self.clip_z is not None and (not np.isfinite(self.clip_z) or self.clip_z <= 0):
            raise FeatureScalingError(f"clip_z must be finite and positive, got {clip_z!r}")
        self.mask_config = normalize_mask_config(mask_config)
        if mode == LEGACY_MODE and any(self.mask_config.values()):
            raise FeatureScalingError("Feature masking is supported only for typed_v1")
        if mode == SCALER_VERSION:
            validate_feature_rules(self.rules)
            if len(self.parameters) != len(self.rules):
                raise FeatureScalingError("typed_v1 requires one fitted parameter entry per rule")
            for rule, parameter in zip(self.rules, self.parameters):
                if (parameter["branch"], int(parameter["index"]), parameter["name"]) != (
                    rule["branch"], int(rule["index"]), rule["name"]
                ):
                    raise FeatureScalingError(f"Parameter/rule mismatch at {rule['branch']}[{rule['index']}]")
                if not np.isfinite(float(parameter["center"])) or not np.isfinite(float(parameter["scale"])):
                    raise FeatureScalingError(f"Non-finite fitted state at {rule['branch']}[{rule['index']}]")
                if float(parameter["scale"]) <= 0:
                    raise FeatureScalingError(f"Non-positive fitted scale at {rule['branch']}[{rule['index']}]")
            self._rule_map = {(r["branch"], int(r["index"])): r for r in self.rules}
            self._parameter_map = {
                (p["branch"], int(p["index"])): p for p in self.parameters
            }
        else:
            self._rule_map = {}
            self._parameter_map = {}

    def with_runtime_config(
        self,
        *,
        clip_z: float | None,
        mask_config: Mapping[str, Any] | None = None,
    ) -> "FeatureScaler":
        """Clone frozen fitted state while changing only post-fit runtime options."""
        return FeatureScaler(
            self.mode,
            self.rules,
            self.parameters,
            self.metadata,
            clip_z=clip_z,
            mask_config=mask_config,
        )

    @classmethod
    def legacy_clip(cls) -> "FeatureScaler":
        return cls(LEGACY_MODE, metadata={"compatibility": "historical nan_to_num then clip[-50,50]"})

    @classmethod
    def fit_from_cache(
        cls,
        train_records: Sequence[Mapping[str, Any]],
        cache_dir: str | Path,
        rules_path: str | Path = DEFAULT_RULES_PATH,
        sample_cap: int = 128,
        seed: int = 20260907,
        manifest_path: str | Path | None = None,
        code_head: str | None = None,
    ) -> "FeatureScaler":
        rules = load_feature_rules(rules_path)
        cache_dir = Path(cache_dir)
        ids = [record_key(record) for record in train_records]
        if len(ids) != len(set(ids)):
            raise FeatureScalingError("FeatureScaler.fit requires unique training records")
        rule_map = {(r["branch"], int(r["index"])): r for r in rules}
        values: dict[tuple[str, int], list[np.ndarray]] = {
            key: [] for key in rule_map
        }
        weights: dict[tuple[str, int], list[np.ndarray]] = {
            key: [] for key in rule_map
        }
        record_counts = {key: 0 for key in rule_map}
        missing_cache = []
        legacy_hrv_sentinel_windows = 0
        legacy_caisr_sentinel_records = 0
        raw_ecg_windows = 0
        consumed_ecg_windows = 0

        for rid in ids:
            path = cache_dir / f"{rid}.npz"
            if not path.is_file():
                missing_cache.append(rid)
                continue
            with np.load(path, allow_pickle=False) as data:
                X_seq = np.asarray(data["X_seq"])
                X_ecg = np.asarray(data["X_ecg"])
                x_static = np.asarray(data["x_static"])
            cls._validate_shapes(X_seq, X_ecg, x_static, rid)
            raw_ecg_windows += len(X_ecg)
            consumed = max(0, min(len(X_ecg), len(X_seq) - ECG_EPOCH_OFFSET))
            consumed_ecg_windows += consumed
            branch_values = {
                "X_seq": X_seq,
                "X_ecg": X_ecg[:consumed],
                "x_static": x_static[None, :],
            }
            hrv_sentinel = (
                np.all(np.isfinite(branch_values["X_ecg"][:, :11]), axis=1)
                & np.all(branch_values["X_ecg"][:, :11] == 0, axis=1)
            ) if consumed else np.zeros(0, dtype=bool)
            legacy_hrv_sentinel_windows += int(hrv_sentinel.sum())
            caisr_sentinel = bool(
                np.isfinite(x_static[10:]).all()
                and x_static[10] == 0
                and np.all(x_static[10:] == 0)
            )
            legacy_caisr_sentinel_records += int(caisr_sentinel)

            for branch, matrix in branch_values.items():
                cap = 1 if branch == "x_static" else int(sample_cap)
                indices = _stable_indices(rid, branch, len(matrix), cap, seed)
                sampled = matrix[indices]
                sampled_hrv = hrv_sentinel[indices] if branch == "X_ecg" else None
                for index in range(EXPECTED_DIMS[branch]):
                    rule = rule_map[(branch, index)]
                    column = sampled[:, index]
                    prepared, valid, _ = _prepare_values(column, rule)
                    if branch == "X_ecg" and index < 11:
                        valid &= ~sampled_hrv
                    if branch == "x_static" and index >= 10 and caisr_sentinel:
                        valid[:] = False
                    if np.any(valid):
                        usable = prepared[valid]
                        values[(branch, index)].append(usable)
                        weights[(branch, index)].append(np.full(len(usable), 1.0 / len(usable)))
                        record_counts[(branch, index)] += 1

        if missing_cache:
            raise FeatureScalingError(
                f"Missing {len(missing_cache)} training caches; first={missing_cache[0]}"
            )

        parameters = []
        for rule in rules:
            key = (rule["branch"], int(rule["index"]))
            parts = values[key]
            if parts:
                x = np.concatenate(parts)
                w = np.concatenate(weights[key])
            else:
                x = np.zeros(0, dtype=np.float64)
                w = np.zeros(0, dtype=np.float64)
            if rule["scaling"] == "identity":
                center, scale, method = 0.0, 1.0, "identity"
                p05 = p25 = median = p75 = p95 = math.nan
                is_constant = bool(len(x) and np.min(x) == np.max(x))
            elif len(x) == 0:
                center, scale, method = 0.0, 1.0, "all_missing_fallback"
                p05 = p25 = median = p75 = p95 = math.nan
                is_constant = False
            else:
                p05 = _weighted_quantile(x, w, 0.05)
                p25 = _weighted_quantile(x, w, 0.25)
                median = _weighted_quantile(x, w, 0.50)
                p75 = _weighted_quantile(x, w, 0.75)
                p95 = _weighted_quantile(x, w, 0.95)
                center = median
                tolerance = 1e-6 * max(1.0, abs(center))
                candidates = [
                    ("iqr", p75 - p25),
                    ("p95_p05", p95 - p05),
                    ("std", float(np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w)))),
                ]
                method, scale = "unit_fallback", 1.0
                for candidate_method, candidate_scale in candidates:
                    if np.isfinite(candidate_scale) and candidate_scale > tolerance:
                        method, scale = candidate_method, float(candidate_scale)
                        break
                is_constant = bool(np.min(x) == np.max(x))
            parameters.append({
                "branch": rule["branch"],
                "index": int(rule["index"]),
                "name": rule["name"],
                "center": float(center),
                "scale": float(scale),
                "scale_method": method,
                "fit_valid_observations": int(len(x)),
                "fit_contributing_records": int(record_counts[key]),
                "is_constant": bool(is_constant),
                "p05": float(p05) if np.isfinite(p05) else None,
                "p25": float(p25) if np.isfinite(p25) else None,
                "p50": float(median) if np.isfinite(median) else None,
                "p75": float(p75) if np.isfinite(p75) else None,
                "p95": float(p95) if np.isfinite(p95) else None,
            })

        metadata = {
            "mode": SCALER_VERSION,
            "version": SCALER_VERSION,
            "sample_cap_per_record_branch": int(sample_cap),
            "sampling_seed": int(seed),
            "sampling": "stable_sha256_seeded_without_replacement_subject_equal_per_column",
            "fit_record_count": len(ids),
            "train_unique_id_hash": stable_hash(sorted(ids)),
            "manifest_sha256": file_sha256(manifest_path) if manifest_path else None,
            "rules_sha256": stable_hash(rules),
            "feature_order_sha256": feature_order_hash(rules),
            "source": "Current extractor order applied to legacy NPZ under explicit unversioned-cache assumption",
            "source_uncertainty": "Historical NPZ archives contain no feature names/extractor version; dimensional agreement is not provenance proof.",
            "legacy_zero_hrv_sentinel_windows_fit": legacy_hrv_sentinel_windows,
            "legacy_zero_caisr_sentinel_records_fit": legacy_caisr_sentinel_records,
            "raw_ecg_windows_fit": raw_ecg_windows,
            "consumed_ecg_windows_fit": consumed_ecg_windows,
            "code_head": code_head,
            "clip_z": None,
        }
        return cls(SCALER_VERSION, rules, parameters, metadata, clip_z=None)

    @staticmethod
    def _validate_shapes(X_seq, X_ecg, x_static, record_id: str = "unknown") -> None:
        if np.asarray(X_seq).ndim != 2 or np.asarray(X_seq).shape[1:] != (SEQ_DIM,) or len(X_seq) == 0:
            raise FeatureScalingError(f"{record_id}: invalid X_seq shape {np.asarray(X_seq).shape}")
        if np.asarray(X_ecg).ndim != 2 or np.asarray(X_ecg).shape[1:] != (ECG_DIM,):
            raise FeatureScalingError(f"{record_id}: invalid X_ecg shape {np.asarray(X_ecg).shape}")
        if np.asarray(x_static).shape != (STATIC_DIM,):
            raise FeatureScalingError(f"{record_id}: invalid x_static shape {np.asarray(x_static).shape}")

    def transform_arrays(
        self,
        X_seq: np.ndarray,
        X_ecg: np.ndarray | None,
        x_static: np.ndarray,
        *,
        record_id: str = "unknown",
        fallback_sequence: bool = False,
        return_diagnostics: bool = False,
    ):
        X_seq = np.asarray(X_seq)
        X_ecg = np.zeros((0, ECG_DIM), dtype=np.float32) if X_ecg is None else np.asarray(X_ecg)
        x_static = np.asarray(x_static)
        self._validate_shapes(X_seq, X_ecg, x_static, record_id)
        if self.mode == LEGACY_MODE:
            result = tuple(
                np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0).astype(np.float32)
                for x in (X_seq, X_ecg, x_static)
            )
            diagnostics = {"mode": LEGACY_MODE}
            return (*result, diagnostics) if return_diagnostics else result

        outputs = {
            "X_seq": np.zeros(X_seq.shape, dtype=np.float64),
            "X_ecg": np.zeros(X_ecg.shape, dtype=np.float64),
            "x_static": np.zeros(x_static.shape, dtype=np.float64),
        }
        inputs = {"X_seq": X_seq, "X_ecg": X_ecg, "x_static": x_static}
        diagnostics = {
            "mode": SCALER_VERSION,
            "legacy_zero_hrv_sentinel_windows": 0,
            "legacy_zero_caisr_sentinel_records": 0,
            "fallback_sequence": bool(fallback_sequence),
            "per_branch": {},
        }
        hrv_sentinel = (
            np.all(np.isfinite(X_ecg[:, :11]), axis=1)
            & np.all(X_ecg[:, :11] == 0, axis=1)
        ) if len(X_ecg) else np.zeros(0, dtype=bool)
        diagnostics["legacy_zero_hrv_sentinel_windows"] = int(hrv_sentinel.sum())
        caisr_sentinel = bool(
            np.isfinite(x_static[10:]).all()
            and x_static[10] == 0
            and np.all(x_static[10:] == 0)
        )
        diagnostics["legacy_zero_caisr_sentinel_records"] = int(caisr_sentinel)

        for branch, array in inputs.items():
            matrix = array if array.ndim == 2 else array[None, :]
            target = outputs[branch] if outputs[branch].ndim == 2 else outputs[branch][None, :]
            branch_diag = {"nonfinite": 0, "invalid": 0, "filled": 0, "roundoff_corrected": 0}
            if branch == "X_seq" and fallback_sequence:
                diagnostics["per_branch"][branch] = branch_diag
                continue
            for index in range(EXPECTED_DIMS[branch]):
                rule = self._rule_map[(branch, index)]
                parameter = self._parameter_map[(branch, index)]
                prepared, valid, item_diag = _prepare_values(matrix[:, index], rule)
                if branch == "X_ecg" and index < 11:
                    valid &= ~hrv_sentinel
                if branch == "x_static" and index >= 10 and caisr_sentinel:
                    valid[:] = False
                if rule["scaling"] == "robust":
                    filled = np.where(valid, prepared, float(parameter["center"]))
                    transformed = (filled - float(parameter["center"])) / float(parameter["scale"])
                else:
                    transformed = np.where(valid, prepared, float(rule.get("identity_fill", 0.0)))
                if self.clip_z is not None and rule["scaling"] == "robust":
                    transformed = np.clip(transformed, -self.clip_z, self.clip_z)
                if branch == "X_ecg" and index < 11:
                    transformed[hrv_sentinel] = 0.0
                if branch == "x_static" and index >= 10 and caisr_sentinel:
                    transformed[:] = 0.0
                if not np.all(np.isfinite(transformed)):
                    raise FeatureScalingError(
                        f"{record_id}: non-finite transformed values in {branch}[{index}] {rule['name']}"
                    )
                target[:, index] = transformed
                branch_diag["nonfinite"] += item_diag["nonfinite"]
                branch_diag["invalid"] += (
                    item_diag["invalid_missing_rule"] + item_diag["range_invalid"]
                    + item_diag["negative_for_log"]
                )
                branch_diag["roundoff_corrected"] += item_diag["roundoff_corrected"]
                branch_diag["filled"] += int((~valid).sum())
            diagnostics["per_branch"][branch] = branch_diag

        for index in self.mask_config["x_static_zero_indices"]:
            outputs["x_static"][index] = 0.0
        for start, stop in self.mask_config["x_seq_zero_ranges"]:
            outputs["X_seq"][:, start:stop] = 0.0
        diagnostics["mask_config"] = self.mask_config

        result = tuple(outputs[name].astype(np.float32, copy=False) for name in ("X_seq", "X_ecg", "x_static"))
        return (*result, diagnostics) if return_diagnostics else result

    def state_dict(self) -> dict:
        return {
            "mode": self.mode,
            "version": SCALER_VERSION if self.mode == SCALER_VERSION else LEGACY_MODE,
            "rules": self.rules,
            "parameters": self.parameters,
            "metadata": self.metadata,
            "clip_z": self.clip_z,
            "mask_config": self.mask_config,
            "rules_sha256": stable_hash(self.rules) if self.rules else None,
            "feature_order_sha256": feature_order_hash(self.rules) if self.rules else None,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FeatureScaler":
        if not isinstance(state, Mapping) or "mode" not in state:
            raise FeatureScalingError("Checkpoint is missing a valid input_preprocessing state")
        mode = state["mode"]
        if mode == LEGACY_MODE:
            return cls.legacy_clip()
        if mode != SCALER_VERSION or state.get("version") != SCALER_VERSION:
            raise FeatureScalingError(f"Unsupported typed preprocessing state: {mode}/{state.get('version')}")
        rules = state.get("rules")
        parameters = state.get("parameters")
        if stable_hash(rules) != state.get("rules_sha256"):
            raise FeatureScalingError("Feature rule checksum mismatch in checkpoint")
        if feature_order_hash(rules) != state.get("feature_order_sha256"):
            raise FeatureScalingError("Feature order checksum mismatch in checkpoint")
        metadata = state.get("metadata") or {}
        if metadata.get("rules_sha256") not in {None, stable_hash(rules)}:
            raise FeatureScalingError("Fitted metadata rule checksum mismatch")
        return cls(
            mode, rules, parameters, metadata,
            clip_z=state.get("clip_z"),
            mask_config=state.get("mask_config"),
        )


def detect_fallback_sequence(X_seq: np.ndarray, X_ecg: np.ndarray, mask: np.ndarray | None) -> bool:
    mask_empty = mask is None or not np.any(np.asarray(mask, dtype=bool))
    return bool(
        np.asarray(X_seq).shape == (1, SEQ_DIM)
        and np.all(np.asarray(X_seq) == 0)
        and np.asarray(X_ecg).shape == (0, ECG_DIM)
        and mask_empty
    )
