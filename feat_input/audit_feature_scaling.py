#!/usr/bin/env python
"""Read-only audit of cached classifier inputs before the legacy [-50, 50] clip.

All outputs are aggregate and are written beside this script.  The script never
modifies NPZ files, production code, model weights, or split manifests.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
EXPECTED_DIMS = {"X_seq": 483, "X_ecg": 12, "x_static": 196}
QUANTILES = np.asarray([0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999])
Q_NAMES = ["p0_1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99_9"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_expr(expr: str) -> str:
    if expr.startswith("float(") and expr.endswith(")"):
        return expr[6:-1]
    return expr.replace(" ", "_").replace("/", "_per_")


def final_feature_list(function, expected: int) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.List):
            continue
        if any(isinstance(target, ast.Name) and target.id == "features" for target in node.targets):
            candidates.append([clean_expr(ast.unparse(item)) for item in node.value.elts])
    matches = [items for items in candidates if len(items) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"Could not identify {expected}-column feature list in {function.__name__}")
    return matches[0]


def algorithmic_names() -> list[tuple[str, str]]:
    from per_epoch_features.feature_extractor_algorithmic import AlgorithmicMixin

    sleep_head = [
        "trt_sec", "tst_sec", "se", "sol_sec", "rem_latency_sec", "wake_time_sec", "waso_sec",
        "n1_pct", "n2_pct", "n3_pct", "rem_pct", "wake_pct",
        "dur_w_sec", "dur_n1_sec", "dur_n2_sec", "dur_n3_sec", "dur_rem_sec",
        "nrem_pct", "n3_n1_ratio", "n3_n1n2_ratio", "rem_nrem_ratio",
        "stage_transition_count", "transition_rate", "wake_intrusions", "short_bout_ratio",
        "n1_bout_count", "n2_bout_count", "n3_bout_count", "rem_bout_count",
    ]
    stage_order = ["wake", "n1", "n2", "n3", "rem"]
    sleep_bouts = [f"{stat}_bout_{stage}_sec" for stat in
                   ["mean", "median", "max", "p25", "p75", "p90", "p95"] for stage in stage_order]
    sleep_tail = [
        "early_n3_pct", "late_rem_pct", "delta_n3", "delta_rem", "delta_w",
        "sleep_cycle_count", "mean_cycle_duration_sec", "first_cycle_nrem_duration_sec",
        "longest_cont_sleep_bout_sec", "longest_n3_bout_sec", "longest_rem_bout_sec",
        "mean_max_prob", "std_max_prob", "mean_stage_entropy", "high_entropy_ratio",
        "low_conf_ratio", "posterior_volatility", "sleep_fragmentation_index",
        "deep_sleep_preservation_index", "rem_integrity_index",
    ]
    sleep = sleep_head + sleep_bouts + sleep_tail
    if len(sleep) != 84:
        raise AssertionError(len(sleep))
    groups = [
        ("caisr_sleep", sleep),
        ("caisr_arousal", final_feature_list(AlgorithmicMixin.extract_algorithmic_arousal_event_features, 30)),
        ("caisr_respiratory", final_feature_list(AlgorithmicMixin.extract_algorithmic_respiratory_event_features, 42)),
        ("caisr_limb", final_feature_list(AlgorithmicMixin.extract_algorithmic_limb_event_features, 30)),
    ]
    return [(family, name) for family, names in groups for name in names]


def source_location(obj) -> tuple[str, int, str]:
    path = Path(inspect.getsourcefile(obj) or "unknown")
    try:
        path = path.resolve().relative_to(ROOT)
    except (ValueError, OSError):
        pass
    try:
        line = inspect.getsourcelines(obj)[1]
    except (OSError, TypeError):
        line = -1
    return str(path), line, getattr(obj, "__name__", type(obj).__name__)


def semantic_metadata(branch: str, index: int, name: str, family: str) -> dict:
    lower = name.lower()
    kind = "continuous"
    unit = "unknown"
    domain = "unknown"
    transform = "none confirmed"
    zero = "may be physiological zero or extraction/fallback zero"
    confidence = "medium"
    missing = "NPZ contains post-extraction zeros; original missingness is generally not recoverable"

    if family in {"stage_event", "demographic_categorical"}:
        kind, unit, domain = "binary_or_fraction", "unitless", "[0,1]"
        zero = "valid absence/category-off; may also be whole-block fallback"
        confidence = "high"
    elif family == "eeg_bsr":
        unit, domain = "percent", "[0,100]"
        zero = "valid no suppression or unavailable-channel fallback"
        confidence = "high"
    elif family == "eeg_spectral":
        if "sef50" in lower:
            unit, domain = "Hz", "nonnegative"
        else:
            unit, domain, transform = "log10 power or power ratio", "signed", "log10 in extractor"
        confidence = "high"
    elif family == "eeg_coherence":
        unit = "mixed coherence-derived units"
        if any(token in lower for token in ["entropy", "lowsig", "highsig"]):
            domain = "typically [0,1]"
        elif any(token in lower for token in ["mdelta", "mtheta", "malpha", "msigma", "mbeta"]):
            domain = "typically [0,1]"
        elif "centroid" in lower or "bandwidth" in lower:
            unit, domain = "Hz", "nonnegative"
        else:
            domain = "nonnegative or ratio; definition-specific"
        confidence = "medium"
    elif family == "emg":
        unit, domain = "feature-specific", "signed" if "log_" in lower else "nonnegative"
        if "log_" in lower:
            transform = "natural log in extractor"
        zero = "valid no activity for some columns or missing-channel fallback"
    elif family == "respiratory_epoch":
        unit = "feature-specific"
        if "fraction" in lower:
            domain = "[0,1]"
        elif "corr" in lower:
            domain = "[-1,1]"
        elif "lag_sec" in lower:
            unit, domain = "seconds", "signed"
        elif "rate_bpm" in lower:
            unit, domain = "breaths/min", "nonnegative"
        elif "sec" in lower:
            unit, domain = "seconds", "nonnegative"
        else:
            domain = "nonnegative"
        zero = "valid absence/zero or missing-channel fallback"
    elif family == "ecg_hrv":
        unit, domain = "feature-specific", "nonnegative"
        if "mediannn" in lower:
            unit = "ms"
        elif "pnn20" in lower:
            unit = "percent (NeuroKit output; verify cached version)"
        elif "model_log_" in lower:
            unit, domain, transform = "natural log power", "signed", "natural log in extractor"
        zero = "may be valid boundary but all first 11 zeros is an HRV-failure heuristic"
    elif family == "circadian":
        unit, domain = "unitless", "[-1,1]"
        zero = "valid phase value or unavailable start-time fallback"
    elif family == "demographic_continuous":
        unit = "years" if name == "age" else "kg/m^2"
        domain = "nonnegative"
        zero = "BMI zero may encode unavailable; age zero would be implausible"
        confidence = "high"
    elif family.startswith("caisr_"):
        unit = "feature-specific"
        if any(token in lower for token in ["delta_", "integrity", "preservation"]):
            domain = "signed"
        elif any(token in lower for token in ["_ratio", "_pct", "_prob", "entropy", "volatility", "burden"]):
            domain = "nonnegative; some ratios/indices may exceed 1"
        else:
            domain = "nonnegative"
        if "_sec" in lower or "duration" in lower or "latency" in lower or "interval" in lower:
            unit = "seconds"
        elif "count" in lower or lower.endswith("intrusions"):
            unit, kind = "count", "count"
        elif "index" in lower or lower.endswith("ari") or lower.endswith("ai"):
            unit = "events/hour or composite index"
        zero = "valid absence possible; all-186 zero also indicates missing CAISR block"

    log_candidate = (
        kind == "continuous" and domain == "nonnegative" and "log" not in lower
        and family not in {"stage_event", "demographic_categorical", "eeg_coherence"}
    )
    return {
        "kind": kind, "unit": unit, "expected_domain": domain,
        "existing_transform": transform, "zero_semantics": zero,
        "missing_policy": missing, "confidence": confidence,
        "log1p_preview_eligible": bool(log_candidate),
        "formula_summary": name,
    }


def build_schema() -> list[dict]:
    from per_epoch_features.per_epoch_extractor import PerEpochExtractor, per_epoch_feature_names
    from per_epoch_features.feature_extractor_ecg_neurokit import MODEL_COLUMNS, extract_5min_hrv
    from per_epoch_features.feature_extractor_hrv_circadian_cos import HRV_CIRCADIAN_FEATURE_NAMES
    from per_epoch_features.feature_extractor_demographic import DemographicMixin
    from per_epoch_features.feature_extractor_algorithmic import AlgorithmicMixin

    seq_names = per_epoch_feature_names()
    ecg_names = list(MODEL_COLUMNS) + list(HRV_CIRCADIAN_FEATURE_NAMES)
    demo = ["age", "sex_Female", "sex_Male", "sex_OtherUnknown",
            "race_Asian", "race_Black", "race_Others", "race_Unavailable", "race_White", "BMI"]
    algo = algorithmic_names()
    static_names = demo + [f"{family}_{name}" for family, name in algo]
    if [len(seq_names), len(ecg_names), len(static_names)] != [483, 12, 196]:
        raise RuntimeError("Schema dimension mismatch")

    seq_source = source_location(PerEpochExtractor.extract_all)
    ecg_source = source_location(extract_5min_hrv)
    demo_source = source_location(DemographicMixin.extract_demographic_features)
    algo_sources = {
        "caisr_sleep": source_location(AlgorithmicMixin.extract_algorithmic_annotations_features),
        "caisr_arousal": source_location(AlgorithmicMixin.extract_algorithmic_arousal_event_features),
        "caisr_respiratory": source_location(AlgorithmicMixin.extract_algorithmic_respiratory_event_features),
        "caisr_limb": source_location(AlgorithmicMixin.extract_algorithmic_limb_event_features),
    }
    schema = []
    for index, name in enumerate(seq_names):
        if index < 54: family = "eeg_spectral"
        elif index < 414: family = "eeg_coherence"
        elif index < 432: family = "eeg_bsr"
        elif index < 456: family = "emg"
        elif index < 470: family = "respiratory_epoch"
        else: family = "stage_event"
        src = seq_source
        item = {"branch": "X_seq", "index": index, "name": name, "family": family,
                "source_file": src[0], "source_line": src[1], "source_function": src[2]}
        item.update(semantic_metadata("X_seq", index, name, family)); schema.append(item)
    for index, name in enumerate(ecg_names):
        family = "ecg_hrv" if index < 11 else "circadian"
        item = {"branch": "X_ecg", "index": index, "name": name, "family": family,
                "source_file": ecg_source[0], "source_line": ecg_source[1],
                "source_function": ecg_source[2] if index < 11 else "extract_hrv_window_circadian_cos"}
        item.update(semantic_metadata("X_ecg", index, name, family)); schema.append(item)
    for index, name in enumerate(static_names):
        if index in {0, 9}: family, src = "demographic_continuous", demo_source
        elif index < 10: family, src = "demographic_categorical", demo_source
        else:
            family = name.split("_", 2)[0] + "_" + name.split("_", 2)[1]
            src = algo_sources[family]
        item = {"branch": "x_static", "index": index, "name": name, "family": family,
                "source_file": src[0], "source_line": src[1], "source_function": src[2]}
        item.update(semantic_metadata("x_static", index, name, family)); schema.append(item)
    return schema


@dataclass
class Aggregate:
    dim: int

    def __post_init__(self):
        self.total = np.zeros(self.dim, np.int64); self.finite = np.zeros(self.dim, np.int64)
        self.nan = np.zeros(self.dim, np.int64); self.posinf = np.zeros(self.dim, np.int64)
        self.neginf = np.zeros(self.dim, np.int64); self.zero = np.zeros(self.dim, np.int64)
        self.negative = np.zeros(self.dim, np.int64); self.eq_neg50 = np.zeros(self.dim, np.int64)
        self.eq_pos50 = np.zeros(self.dim, np.int64); self.lt_neg50 = np.zeros(self.dim, np.int64)
        self.gt_pos50 = np.zeros(self.dim, np.int64); self.sum = np.zeros(self.dim, np.float64)
        self.sumsq = np.zeros(self.dim, np.float64); self.minimum = np.full(self.dim, np.inf)
        self.maximum = np.full(self.dim, -np.inf); self.records = 0

    def update(self, values: np.ndarray):
        values = np.asarray(values)
        if values.ndim == 1: values = values[None, :]
        self.records += 1
        if values.shape[0] == 0:
            return
        finite = np.isfinite(values); safe = np.where(finite, values, 0.0).astype(np.float64)
        self.total += values.shape[0]; self.finite += finite.sum(0)
        self.nan += np.isnan(values).sum(0); self.posinf += np.isposinf(values).sum(0)
        self.neginf += np.isneginf(values).sum(0); self.zero += ((values == 0) & finite).sum(0)
        self.negative += ((values < 0) & finite).sum(0); self.eq_neg50 += ((values == -50) & finite).sum(0)
        self.eq_pos50 += ((values == 50) & finite).sum(0); self.lt_neg50 += ((values < -50) & finite).sum(0)
        self.gt_pos50 += ((values > 50) & finite).sum(0); self.sum += safe.sum(0)
        self.sumsq += np.square(safe).sum(0)
        lo = np.min(np.where(finite, values, np.inf), axis=0); hi = np.max(np.where(finite, values, -np.inf), axis=0)
        self.minimum = np.minimum(self.minimum, lo); self.maximum = np.maximum(self.maximum, hi)


def weighted_quantile(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any(): return np.full(len(QUANTILES), np.nan)
    values, weights = values[finite], weights[finite]
    order = np.argsort(values); values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights; cumulative /= weights.sum()
    return np.interp(QUANTILES, cumulative, values)


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any(): return math.nan, math.nan
    x, w = values[finite], weights[finite]; mean = float(np.average(x, weights=w))
    return mean, float(np.sqrt(np.average((x - mean) ** 2, weights=w)))


def select_rows(values: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(values); k = min(n, cap)
    if k == 0: return values[:0], np.zeros(0), np.zeros(0)
    indices = np.unique(np.linspace(0, n - 1, k, dtype=np.int64)); sampled = values[indices]
    return sampled, np.full(len(indices), n / len(indices)), np.full(len(indices), 1.0 / len(indices))


def scope_keys(split: str, site: str) -> list[str]:
    return ["global:all", f"role:{split}", f"site:{site}"]


def audit(cache_root: Path, split_root: Path, sample_cap: int, schema: list[dict]):
    manifests, identity = {}, {}
    for split in ["train", "val", "test", "external"]:
        records = json.loads((split_root / f"{split}_records.json").read_text())
        manifests[split] = records
        for record in records:
            key = f"{record['BidsFolder']}_ses-{record['SessionID']}"
            if key in identity: raise RuntimeError(f"Duplicate manifest record: {key}")
            identity[key] = (split, str(record["SiteID"]))

    dims = EXPECTED_DIMS
    aggregates: dict[tuple[str, str], Aggregate] = {}
    samples = {branch: [] for branch in dims}; sample_meta = {branch: [] for branch in dims}
    record_clip = {branch: [] for branch in dims}; record_meta = {branch: [] for branch in dims}
    inventory = {
        "cache_root": str(cache_root), "split_root": str(split_root), "manifest_sha256": {},
        "expected_dimensions": dims, "records_by_split": {}, "records_by_site": dict(Counter(site for _, site in identity.values())),
        "key_sets": Counter(), "dtype_sets": defaultdict(Counter), "shape_ranges": {}, "errors": [],
        "duplicate_record_keys": [], "missing_cache": [], "unexpected_cache": [],
        "label_counts_by_split": {}, "mask_semantics": "True marks ECG alignment only; it is not EEG quality or HRV-success",
        "heuristics": {}, "cache_version_assessment": "No extractor/schema version field was found; directory timestamps cannot prove a uniform version.",
    }
    for split in manifests:
        inventory["manifest_sha256"][split] = sha256(split_root / f"{split}_records.json")
        inventory["records_by_split"][split] = len(manifests[split])

    shape_values = defaultdict(list); labels = defaultdict(Counter); mask_true = []; hrv_zero_windows = 0; ecg_windows = 0; empty_ecg_records = 0
    block_zero_records = Counter(); seen = set()
    for number, (key, (split, site)) in enumerate(sorted(identity.items()), 1):
        path = cache_root / split / f"{key}.npz"
        if not path.is_file(): inventory["missing_cache"].append({"split": split, "site": site}); continue
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = tuple(sorted(data.files)); inventory["key_sets"]["|".join(keys)] += 1
                arrays = {name: np.asarray(data[name]) for name in dims}; mask = np.asarray(data["mask"]); y = int(np.asarray(data["y"]).reshape(-1)[0])
            if arrays["X_seq"].ndim != 2 or arrays["X_seq"].shape[1] != 483: raise ValueError(f"X_seq shape {arrays['X_seq'].shape}")
            if arrays["X_ecg"].ndim != 2 or arrays["X_ecg"].shape[1] != 12: raise ValueError(f"X_ecg shape {arrays['X_ecg'].shape}")
            if arrays["x_static"].shape != (196,): raise ValueError(f"x_static shape {arrays['x_static'].shape}")
            if mask.shape != (len(arrays["X_seq"]),): raise ValueError(f"mask shape {mask.shape}")
            labels[split][str(y)] += 1; mask_true.append((len(mask), int(mask.astype(bool).sum())))
            empty_ecg_records += int(len(arrays["X_ecg"]) == 0)
            ecg_windows += len(arrays["X_ecg"]); hrv_zero_windows += int(np.all(arrays["X_ecg"][:, :11] == 0, axis=1).sum())
            for branch, values in arrays.items():
                if values.ndim == 1: check_values = values[None, :]
                else: check_values = values
                shape_values[branch].append(list(values.shape)); inventory["dtype_sets"][branch][str(values.dtype)] += 1
                for group in scope_keys(split, site):
                    aggregates.setdefault((group, branch), Aggregate(dims[branch])).update(check_values)
                sampled, obs_w, subj_w = select_rows(check_values, sample_cap if branch != "x_static" else 1)
                samples[branch].append(sampled); sample_meta[branch].extend([(split, site, ow, sw) for ow, sw in zip(obs_w, subj_w)])
                finite = np.isfinite(check_values); denom = finite.sum(0); affected = (((check_values < -50) | (check_values > 50)) & finite).sum(0)
                record_clip[branch].append(np.divide(affected, denom, out=np.full(dims[branch], np.nan), where=denom > 0))
                record_meta[branch].append((split, site))
            if np.all(arrays["X_seq"][:, 414:432] == 0): block_zero_records["BSR_all_zero"] += 1
            if np.all(arrays["X_seq"][:, 432:456] == 0): block_zero_records["EMG_all_zero"] += 1
            if np.all(arrays["X_seq"][:, 456:470] == 0): block_zero_records["Resp_all_zero"] += 1
            if np.all(arrays["X_seq"][:, 470:483] == 0): block_zero_records["StageEvent_all_zero"] += 1
            if np.all(arrays["x_static"][10:] == 0): block_zero_records["CAISR_static_all_zero"] += 1
            seen.add(key)
        except Exception as exc:
            inventory["errors"].append({"split": split, "site": site, "error": f"{type(exc).__name__}: {exc}"})

    for path in cache_root.glob("*/*.npz"):
        if path.stem not in identity: inventory["unexpected_cache"].append(str(path.relative_to(cache_root)))
    for branch, shapes in shape_values.items():
        inventory["shape_ranges"][branch] = {"examples": shapes[:3], "first_axis_min": min(s[0] for s in shapes), "first_axis_max": max(s[0] for s in shapes)}
    inventory["key_sets"] = dict(inventory["key_sets"]); inventory["dtype_sets"] = {k: dict(v) for k, v in inventory["dtype_sets"].items()}
    inventory["label_counts_by_split"] = {k: dict(v) for k, v in labels.items()}
    inventory["heuristics"] = {"all_zero_block_record_counts": dict(block_zero_records),
                               "empty_ecg_records": empty_ecg_records, "hrv_first11_all_zero_windows": hrv_zero_windows, "ecg_windows": ecg_windows,
                               "hrv_zero_window_fraction": hrv_zero_windows / ecg_windows if ecg_windows else None,
                               "mask_total_epochs": sum(x for x, _ in mask_true), "mask_true_epochs": sum(y for _, y in mask_true)}
    inventory["records_read"] = len(seen); inventory["manifest_records"] = len(identity)

    schema_lookup = {(item["branch"], item["index"]): item for item in schema}
    stats_rows = []
    for branch, dim in dims.items():
        matrix = np.concatenate(samples[branch], axis=0); meta = sample_meta[branch]
        splits = np.asarray([x[0] for x in meta]); sites = np.asarray([x[1] for x in meta])
        obs_weights = np.asarray([x[2] for x in meta]); subj_weights = np.asarray([x[3] for x in meta])
        clip_matrix = np.asarray(record_clip[branch]); clip_meta = record_meta[branch]
        clip_splits = np.asarray([x[0] for x in clip_meta]); clip_sites = np.asarray([x[1] for x in clip_meta])
        groups = sorted(group for group, b in aggregates if b == branch)
        for group in groups:
            prefix, value = group.split(":", 1)
            mask = np.ones(len(matrix), bool) if prefix == "global" else (splits == value if prefix == "role" else sites == value)
            rmask = np.ones(len(clip_matrix), bool) if prefix == "global" else (clip_splits == value if prefix == "role" else clip_sites == value)
            agg = aggregates[(group, branch)]
            mean = np.divide(agg.sum, agg.finite, out=np.full(dim, np.nan), where=agg.finite > 0)
            variance = np.divide(agg.sumsq, agg.finite, out=np.full(dim, np.nan), where=agg.finite > 0) - mean ** 2
            std = np.sqrt(np.maximum(variance, 0))
            for index in range(dim):
                q_obs = weighted_quantile(matrix[mask, index], obs_weights[mask]); q_sub = weighted_quantile(matrix[mask, index], subj_weights[mask])
                clipped = np.clip(matrix[mask, index], -50, 50); _, clip_std = weighted_mean_std(clipped, obs_weights[mask])
                clip_q = weighted_quantile(clipped, obs_weights[mask]); record_values = clip_matrix[rmask, index]
                schema_item = schema_lookup[(branch, index)]; finite_n = int(agg.finite[index]); sample_n = int(np.isfinite(matrix[mask, index]).sum())
                row = {"scope": prefix, "site": value if prefix == "site" else "", "fold": "small_I0002",
                       "role": value if prefix == "role" else "all", "branch": branch, "index": index,
                       "name": schema_item["name"], "family": schema_item["family"], "records": agg.records,
                       "total_observations": int(agg.total[index]), "finite_observations": finite_n,
                       "nan_count": int(agg.nan[index]), "posinf_count": int(agg.posinf[index]), "neginf_count": int(agg.neginf[index]),
                       "recognizable_missing_count": "", "unknown_missing_marker": "zeros_ambiguous",
                       "zero_fraction_finite": agg.zero[index] / finite_n if finite_n else np.nan,
                       "negative_fraction_finite": agg.negative[index] / finite_n if finite_n else np.nan,
                       "equal_neg50_fraction_finite": agg.eq_neg50[index] / finite_n if finite_n else np.nan,
                       "equal_pos50_fraction_finite": agg.eq_pos50[index] / finite_n if finite_n else np.nan,
                       "below_neg50_fraction_finite": agg.lt_neg50[index] / finite_n if finite_n else np.nan,
                       "above_pos50_fraction_finite": agg.gt_pos50[index] / finite_n if finite_n else np.nan,
                       "flattened_by_old_clip_fraction_finite": (agg.lt_neg50[index] + agg.gt_pos50[index]) / finite_n if finite_n else np.nan,
                       "min": agg.minimum[index] if finite_n else np.nan, "max": agg.maximum[index] if finite_n else np.nan,
                       "mean": mean[index], "std": std[index], "iqr_approx": q_obs[5] - q_obs[3],
                       "old_clip_std_approx": clip_std, "old_clip_iqr_approx": clip_q[5] - clip_q[3],
                       "sample_n": sample_n, "quantile_method": f"deterministic_uniform_per_record_cap_{sample_cap}_weighted_by_record_length"}
                row.update(dict(zip(Q_NAMES, q_obs))); row.update({f"subject_equal_{n}": v for n, v in zip(Q_NAMES, q_sub)})
                row["record_clip_fraction_p50"] = float(np.nanquantile(record_values, 0.5)) if np.isfinite(record_values).any() else np.nan
                row["record_clip_fraction_p95"] = float(np.nanquantile(record_values, 0.95)) if np.isfinite(record_values).any() else np.nan
                row["record_clip_fraction_max"] = float(np.nanmax(record_values)) if np.isfinite(record_values).any() else np.nan
                if schema_item["kind"] == "binary_or_fraction":
                    vals = matrix[mask, index]; vals = vals[np.isfinite(vals)]; counts = Counter(vals.tolist()).most_common(8)
                    row["top_values_sample"] = json.dumps(counts)
                else: row["top_values_sample"] = ""
                stats_rows.append(row)
    return inventory, pd.DataFrame(stats_rows), samples, sample_meta


def candidate_preview(stats: pd.DataFrame, samples: dict, sample_meta: dict, schema: list[dict]) -> pd.DataFrame:
    schema_lookup = {(x["branch"], x["index"]): x for x in schema}; rows = []
    train_stats = stats[(stats.scope == "role") & (stats.role == "train")].set_index(["branch", "index"])
    for branch, blocks in samples.items():
        matrix = np.concatenate(blocks, axis=0); meta = sample_meta[branch]
        keep = np.asarray([x[0] == "train" for x in meta]); weights = np.asarray([x[2] for x in meta])[keep]; matrix = matrix[keep]
        for index in range(matrix.shape[1]):
            item = schema_lookup[(branch, index)]; raw = matrix[:, index].astype(float); finite = np.isfinite(raw)
            x, w = raw[finite], weights[finite]
            stat = train_stats.loc[(branch, index)]; median = float(stat.p50); iqr = float(stat.p75 - stat.p25)
            mean, sd = float(stat["mean"]), float(stat["std"])
            identity = item["kind"] == "binary_or_fraction" or any(token in item["name"].lower() for token in ("_pct", "_ratio", "fraction", "prob"))
            for candidate in ["A_old_clip", "B_mean_std", "C_median_iqr", "D_selective_log1p_robust"]:
                transform = candidate; center = 0.0; scale = 1.0; status = "previewed"
                if candidate == "A_old_clip": z = np.clip(x, -50, 50); transform = "clip[-50,50]"
                elif identity:
                    z = x; transform = "identity_for_binary_or_fraction"
                elif candidate == "B_mean_std":
                    center, scale = mean, sd
                    if not np.isfinite(scale) or scale <= 1e-6: z = x - center; scale = 1.0; status = "constant_or_tiny_scale"
                    else: z = (x - center) / scale
                elif candidate == "C_median_iqr" or not item["log1p_preview_eligible"]:
                    center, scale = median, iqr
                    if not np.isfinite(scale) or scale <= 1e-6: z = x - center; scale = 1.0; status = "constant_or_tiny_scale"
                    else: z = (x - center) / scale
                    if candidate.startswith("D_"): transform = "same_as_C_not_log_eligible"
                else:
                    center, scale = median, iqr
                    status = "D_log_not_run_missing_approved_physical_unit_fallback_to_C"
                    transform = "median_iqr_fallback; log1p pending physical unit approval"
                    if not np.isfinite(scale) or scale <= 1e-6: z = x - center; scale = 1.0; status += ";constant_or_tiny_scale"
                    else: z = (x - center) / scale
                denom = w.sum() if len(w) else 0
                frac = lambda threshold: float(w[np.abs(z) > threshold].sum() / denom) if denom else np.nan
                rows.append({"fold": "small_I0002", "fit_role": "train_only", "branch": branch, "index": index,
                             "name": item["name"], "family": item["family"], "candidate": candidate,
                             "transform": transform, "status": status, "center": center, "scale": scale, "unit_conversion": "none_unapproved",
                             "sample_n": len(x), "finite": bool(np.isfinite(z).all()),
                             "abs_gt_5_fraction": frac(5), "abs_gt_10_fraction": frac(10),
                             "abs_gt_20_fraction": frac(20), "abs_gt_50_fraction": frac(50)})
    return pd.DataFrame(rows)


def environment_snapshot(cache_root: Path, split_root: Path, args) -> dict:
    import scipy, sklearn, torch
    try:
        import neurokit2
        nk = neurokit2.__version__
    except Exception as exc:
        nk = f"unavailable: {exc}"
    return {
        "cwd": str(Path.cwd()), "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status_before": subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True).splitlines(),
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__, "neurokit2": nk,
        "relevant_environment": {k: v for k, v in os.environ.items() if k.startswith(("LSTM", "CUDA", "CUBLAS", "PYTHONHASH"))},
        "cache_root": str(cache_root), "split_root": str(split_root), "sample_cap_per_record": args.sample_cap,
    }


def write_report(inventory: dict, stats: pd.DataFrame, candidates: pd.DataFrame, elapsed: float, env: dict):
    global_stats = stats[(stats.scope == "global")].copy()
    damaged = global_stats.sort_values(["flattened_by_old_clip_fraction_finite", "record_clip_fraction_p95"], ascending=False)
    damaged.to_csv(OUT / "clipping_damage_top.csv", index=False)
    top = damaged.head(15)[["branch", "index", "name", "family", "flattened_by_old_clip_fraction_finite", "record_clip_fraction_p95", "min", "max"]]
    lines = [
        "# P0 feature clipping and scaling audit", "",
        "## Scope and result", "",
        f"This read-only audit scanned {inventory['records_read']}/{inventory['manifest_records']} cached records under `{inventory['cache_root']}`.",
        "It did not modify extractors, training/inference code, NPZ files, splits, or model weights, and it did not train a model.",
        "Full finite/zero/clip counts and extrema are exact. Quantiles are reproducible approximations from uniformly spaced rows per record; observation-weighted and subject-equal summaries are both retained.", "",
        "## Verified data layout", "",
        f"- Split counts: `{inventory['records_by_split']}`; site counts: `{inventory['records_by_site']}`.",
        f"- Required array dimensions were verified as `{inventory['expected_dimensions']}`.",
        f"- Cache shape audit found {inventory['heuristics']['empty_ecg_records']} records with `X_ecg.shape[0] == 0`; these valid empty arrays are retained as a modality-availability finding, not a read failure.",
        f"- Read errors: {len(inventory['errors'])}; missing caches: {len(inventory['missing_cache'])}; unexpected caches: {len(inventory['unexpected_cache'])}.",
        "- `mask` marks ECG alignment only. It must not filter EEG epochs and does not prove HRV computation success.",
        "- No cache schema/extractor version field exists, so uniform extraction version cannot be proven from NPZ contents or timestamps.", "",
        "## Production path and confirmed destructive preprocessing", "",
        "- Extraction concatenates the 483-dimensional sequence and replaces NaN/Inf with zero in `per_epoch_features/per_epoch_extractor.py:278-284`.",
        "- Training preserves raw age for the metric, then applies `nan_to_num` plus a uniform `[-50,50]` clip to all three branches in `train_lstm.py:109-125`.",
        "- Inference validates and zero-fills nonfinite values in `team_code.py:175-198`, then applies the same uniform clip in `team_code.py:645-670`.",
        "- Sigmoid clipping, probability clipping, source-level probability bounds, waveform QC, and gradient clipping are separate safeguards and were not treated as defects.", "",
        "## Largest observed clipping damage", "",
        "```text", top.to_string(index=False, float_format=lambda value: f"{value:.6g}"), "```", "",
        "`clipping_damage_top.csv` contains all 691 columns in transparent descending order, not only this excerpt.", "",
        "## Missingness interpretation", "",
        f"- Cached NaN/Inf counts describe the post-extraction cache, not raw-source availability. All-zero block heuristics were: `{inventory['heuristics']['all_zero_block_record_counts']}`.",
        f"- HRV first-11 all-zero windows: {inventory['heuristics']['hrv_first11_all_zero_windows']}/{inventory['heuristics']['ecg_windows']}. This is only a failure heuristic; circadian cosine does not validate HRV.",
        "- A zero may be a valid no-event value, one-hot off state, true BSR zero, or an extraction/fallback placeholder. This audit does not convert zeros back to missing values.", "",
        "## Candidate transforms (offline numeric preview only)", "",
        "Candidates A-D were fitted only on the confirmed 733-record train manifest. Test/external values were not used to fit centers, scales, units, thresholds, or transform choices.",
        "Categorical, fraction, probability, percentage and named ratio columns were left on their physical scale. Candidate D log1p was not run because feature-specific physical reference units have not been approved; eligible D rows fall back to C and are explicitly marked pending.",
        "These previews do not establish a best transform and make no performance claim. Physical unit normalization (notably percent versus fraction) still requires manual schema approval before implementation.", "",
        "## Formal repair design (not implemented)", "",
        "1. Fit one shared column-aware transformer on unique inner-training records before balanced sampling.",
        "2. Exclude padding and ECG non-alignment from fit; transform real values first, then create model padding zeros.",
        "3. Preserve a separate raw-age copy for Age-conditioned AUROC; never transform it in place.",
        "4. Save schema checksum, feature order, unit conversions, per-column transform, center/scale, missing policy, constant rules, and training-manifest checksum in the checkpoint.",
        "5. Make cached and raw-extraction inference paths call the same transform exactly once; reject schema/checksum mismatches and never attach a new scaler to old weights.",
        "6. Missing indicators or architecture changes belong to a later ablation and are not part of this P0 repair.", "",
        "## Limitations and unresolved items", "",
        "- Historical NPZ extraction-version uniformity is not provable because the archives have no version metadata.",
        "- Source-level missingness and quality cannot generally be recovered after zero filling; no EDF re-extraction was performed.",
        "- Several CAISR composite indices and coherence-derived quantities have definition-specific domains; schema confidence is explicitly marked and requires human review.",
        "- Candidate D log1p was not run: physical reference units must be approved before formal code changes.", "",
        "## Reproduction", "",
        "```bash", "conda run -n sleepfm_env python feat_input/test_audit_feature_scaling.py",
        "conda run -n sleepfm_env python feat_input/audit_feature_scaling.py --cache-root npz_new --split-root split --sample-cap 64", "```", "",
        f"Elapsed audit time: {elapsed:.1f} seconds.",
    ]
    (OUT / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=ROOT / "npz_new")
    parser.add_argument("--split-root", type=Path, default=ROOT / "split")
    parser.add_argument("--sample-cap", type=int, default=64)
    args = parser.parse_args(); start = time.time()
    cache_root, split_root = args.cache_root.resolve(), args.split_root.resolve()
    schema = build_schema(); (OUT / "feature_schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")
    env = environment_snapshot(cache_root, split_root, args)
    inventory, stats, samples, sample_meta = audit(cache_root, split_root, args.sample_cap, schema)
    stats.to_csv(OUT / "feature_stats.csv", index=False)
    candidates = candidate_preview(stats, samples, sample_meta, schema); candidates.to_csv(OUT / "candidate_transform_summary.csv", index=False)
    (OUT / "cache_inventory.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
    elapsed = time.time() - start; write_report(inventory, stats, candidates, elapsed, env)
    env["elapsed_seconds"] = elapsed; env["outputs"] = sorted(p.name for p in OUT.iterdir() if p.is_file())
    env["git_status_after"] = subprocess.check_output(["git", "status", "--short", "--branch"], cwd=ROOT, text=True).splitlines()
    (OUT / "run_log.txt").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"records": inventory["records_read"], "stats_rows": len(stats), "candidate_rows": len(candidates), "elapsed_seconds": elapsed}, indent=2))


if __name__ == "__main__":
    main()
