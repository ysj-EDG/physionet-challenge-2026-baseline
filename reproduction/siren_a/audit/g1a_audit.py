#!/usr/bin/env python3
"""G1-A evidence audit and draft manifest builder.

This module performs only streaming metadata/validity counting and builds a
reviewable draft manifest.  It intentionally has no feature-generation,
training, cache-writing, or label-dependent code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
NPZ_ROOT = Path(
    "/database/home/gaohaojie/workspace/python-example-2026/"
    "validity_v0_2_full_6600/workspace/result_bundle/npz"
)
DEMOGRAPHICS_PATH = Path("/database2/physionet2026_kaggle/data2/demographics.csv")
EXTRACTION_VERSION = "timegrid_v2.1.0"
VALIDITY_SCHEMA_VERSION = "validity_v0.2"
EXPECTED_SHAPES = {"X_seq": (483,), "X_ecg": (12,), "x_static": (196,)}
THRESHOLDS = (1, 2, 3, 5, 10, 20, 30, 60)
EEG_CHANNELS = ("F3", "F4", "C3", "C4", "O1", "O2")
EEG_PAIRS = (("F3", "F4"), ("C3", "C4"), ("O1", "O2"))
STAGE_CODES = {
    "Wake": (5,),
    "REM": (4,),
    "N1": (3,),
    "N2": (2,),
    "N3": (1,),
    "NREM": (1, 2, 3),
    "All-stage": (1, 2, 3, 4, 5),
}
NREM_STAGE_CODES = (1, 2, 3)
ALL_STAGE_CODES = (1, 2, 3, 4, 5)
STAGE_ORDER = tuple(STAGE_CODES)
CHANNEL_INDEX = {name: i for i, name in enumerate(EEG_CHANNELS)}


def _scalar(value: np.ndarray | Any) -> Any:
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value


def _json_value(value: Any) -> Any:
    value = _scalar(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def compute_stage_masks(stage_code_aligned: np.ndarray, stage_valid: np.ndarray) -> dict[str, np.ndarray]:
    """Return valid epoch masks for the fixed G1-A stage definitions."""
    codes = np.asarray(stage_code_aligned, dtype=float).reshape(-1)
    valid = np.asarray(stage_valid, dtype=bool).reshape(-1)
    if len(codes) != len(valid):
        raise ValueError("stage_code_aligned and stage_valid lengths differ")
    finite = np.isfinite(codes)
    return {
        name: valid & finite & np.isin(codes, allowed)
        for name, allowed in STAGE_CODES.items()
    }


def eeg_validity_mask(channel_available: np.ndarray, clean_subsegment_count: np.ndarray) -> np.ndarray:
    """Base EEG validity: available lead and clean_subsegment_count > 0."""
    available = np.asarray(channel_available, dtype=bool).reshape(-1)
    clean = np.asarray(clean_subsegment_count)
    if clean.ndim != 2 or clean.shape[1] != len(EEG_CHANNELS):
        raise ValueError("expected (epoch, 6) clean-subsegment array")
    if available.shape != (len(EEG_CHANNELS),):
        raise ValueError("expected six channel-availability flags")
    return (clean > 0) & available[None, :]


def intersected_count(validity: np.ndarray, columns: Iterable[int]) -> int:
    """Count epochs valid on every requested lead using one intersection mask."""
    columns = list(columns)
    if not columns:
        return 0
    return int(np.count_nonzero(np.all(np.asarray(validity, dtype=bool)[:, columns], axis=1)))


def _rule_map() -> dict[tuple[str, int], dict[str, Any]]:
    with (ROOT / "feat_input/feature_rules_v1.json").open(encoding="utf-8") as handle:
        rules = json.load(handle)
    return {(row["branch"], int(row["index"])): row for row in rules}


def _feature_rule(branch: str, index: int) -> dict[str, Any]:
    return _rule_map_cached()[(branch, index)]


_RULES_CACHE: dict[tuple[str, int], dict[str, Any]] | None = None


def _rule_map_cached() -> dict[tuple[str, int], dict[str, Any]]:
    global _RULES_CACHE
    if _RULES_CACHE is None:
        _RULES_CACHE = _rule_map()
    return _RULES_CACHE


def _caisr_manifest_rows() -> list[dict[str, str]]:
    path = ROOT / "data2_caisr20_results/feature_manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _base_row(**kwargs: Any) -> dict[str, Any]:
    fields = {
        "view_id": "",
        "view_position": "",
        "unique_feature_id": "",
        "feature_name": "",
        "family": "",
        "source_npz_field": "",
        "source_index": "",
        "source_indices": "",
        "channel": "",
        "stage_group": "",
        "aggregation": "",
        "formula": "",
        "unit": "",
        "validity_source": "",
        "validity_expression": "",
        "missing_semantics": "",
        "specification_status": "",
        "open_decision_id": "",
        "source_code_reference": "",
    }
    fields.update({key: str(value) if value is not None else "" for key, value in kwargs.items()})
    return fields


def _eeg_rows(stage: str, stat: str) -> list[dict[str, str]]:
    rows = []
    for index in range(54):
        rule = _feature_rule("X_seq", index)
        channel = EEG_CHANNELS[index // 9]
        name = f"eeg_{stage}_{rule['name']}_{stat.lower()}"
        rows.append(_base_row(
            unique_feature_id=name,
            feature_name=name,
            family="EEG54_stage_spectral",
            source_npz_field="X_seq",
            source_index=index,
            channel=channel,
            stage_group=stage,
            aggregation=stat,
            formula=f"Mean/IQR of X_seq[:, {index}] on stage_valid & stage_code_aligned in {stage}",
            unit=rule.get("original_unit", "source-defined"),
            validity_source="stage_valid; eeg_channel_available; eeg_clean_subsegment_count",
            validity_expression="stage_valid & stage_code_aligned in stage set & eeg_channel_available[channel] & eeg_clean_subsegment_count[:,channel] > 0",
            missing_semantics="No final minimum valid-epoch threshold; IQR small-n rule is open.",
            specification_status="PENDING_G1B_REVIEW",
            open_decision_id="DEC-EEG-MIN-EPOCHS; DEC-IQR-SMALL-N",
            source_code_reference=f"feat_input/feature_rules_v1.json[{index}]; per_epoch_features/per_epoch_extractor.py:_extract_per_epoch_eeg; per_epoch_features/eeg_sleep_features.py:FEATURE_NAMES",
        ))
    return rows


def _band_index(channel: str, band: str) -> int:
    lead = {"F3": 0, "F4": 9, "C3": 18, "C4": 27, "O1": 36, "O2": 45}[channel]
    offset = {"sigma": 1, "alpha": 2, "theta": 3, "delta": 4}[band.lower()]
    return lead + offset


def _beta_index(channel: str) -> int:
    return {"F3": 7, "F4": 16, "C3": 25, "C4": 34, "O1": 43, "O2": 52}[channel]


def _p0_index(channel: str) -> int:
    return {"F3": 6, "F4": 15, "C3": 24, "C4": 33, "O1": 42, "O2": 51}[channel]


def spatial_candidate_rows() -> list[dict[str, str]]:
    """Build exactly the 24 asymmetry + 12 AP-gradient candidate rows."""
    rows: list[dict[str, str]] = []
    bands = ("Delta", "Theta", "Alpha", "Sigma")
    for stage in ("N2", "REM"):
        for left, right in EEG_PAIRS:
            for band in bands:
                indices = [_band_index(left, band), _band_index(right, band), _beta_index(left), _beta_index(right), _p0_index(left), _p0_index(right)]
                name = f"spatial_asym_{stage}_{left}-{right}_{band.lower()}_mean"
                rows.append(_base_row(
                    unique_feature_id=name, feature_name=name,
                    family="spatial_asymmetry_candidate", source_npz_field="X_seq",
                    source_indices=";".join(map(str, indices)), channel=f"{left}-{right}",
                    stage_group=stage, aggregation="Mean",
                    formula="(logPbandPbeta + logPbeta - logP0)[left] - (logPbandPbeta + logPbeta - logP0)[right]",
                    unit="log10 relative power difference",
                    validity_source="EEG validity metadata",
                    validity_expression=f"same epoch: {left} and {right} available and clean_subsegment_count > 0",
                    missing_semantics="Candidate only; no final minimum valid-epoch threshold.",
                    specification_status="PENDING_G1B_REVIEW", open_decision_id="DEC-SPATIAL-36",
                    source_code_reference="G1-A candidate formula; not an existing extractor output; X_seq EEG54 indices from feature_rules_v1.json",
                ))
    for stage in ("N2", "N3", "REM"):
        for band in bands:
            indices = [_band_index(ch, band) for ch in EEG_CHANNELS]
            indices += [_beta_index(ch) for ch in EEG_CHANNELS]
            indices += [_p0_index(ch) for ch in EEG_CHANNELS]
            name = f"spatial_ap_gradient_{stage}_{band.lower()}_mean"
            rows.append(_base_row(
                unique_feature_id=name, feature_name=name,
                family="spatial_ap_gradient_candidate", source_npz_field="X_seq",
                source_indices=";".join(map(str, indices)), channel="F3,F4,O1,O2",
                stage_group=stage, aggregation="Mean",
                formula="mean(F3,F4 logPbandPbeta + logPbeta - logP0) - mean(O1,O2 logPbandPbeta + logPbeta - logP0)",
                unit="log10 relative power difference",
                validity_source="EEG validity metadata",
                validity_expression="same epoch: F3,F4,O1,O2 available and clean_subsegment_count > 0",
                missing_semantics="Candidate only; no final minimum valid-epoch threshold.",
                specification_status="PENDING_G1B_REVIEW", open_decision_id="DEC-SPATIAL-36",
                source_code_reference="G1-A candidate formula; not an existing extractor output; X_seq EEG54 indices from feature_rules_v1.json",
            ))
    return rows


def _demo_rows() -> list[dict[str, str]]:
    rows = []
    for slot in range(1, 6):
        name = f"demo5_slot_{slot}_pending"
        rows.append(_base_row(
            unique_feature_id=name, feature_name=name, family="demographic_candidate",
            source_npz_field="x_static and demographics.csv", source_index="pending",
            stage_group="All-night", aggregation="pending", formula="PENDING_G1B_REVIEW",
            unit="pending", validity_source="demographics.csv and x_static[0:10]",
            validity_expression="PENDING_G1B_REVIEW", missing_semantics="PENDING_G1B_REVIEW",
            specification_status="PENDING_G1B_REVIEW", open_decision_id="DEC-DEMO5",
            source_code_reference="per_epoch_features/helper_code.py; per_epoch_features/feature_extractor_demographic.py",
        ))
    return rows


def build_draft_manifest() -> list[dict[str, str]]:
    """Return the 827-row five-view draft with shared Demo5 IDs."""
    demo = _demo_rows()
    v1 = _eeg_rows("Wake", "Mean") + _eeg_rows("Wake", "IQR") + demo
    v2 = _eeg_rows("REM", "Mean") + _eeg_rows("REM", "IQR")
    v2 += [r for r in spatial_candidate_rows() if r["stage_group"] == "REM"] + demo
    v3 = []
    for stage in ("N2", "N3", "NREM"):
        v3 += _eeg_rows(stage, "Mean") + _eeg_rows(stage, "IQR")
    v3 += [r for r in spatial_candidate_rows() if r["stage_group"] in ("N2", "N3")] + demo
    v4 = _eeg_rows("All-stage", "Mean") + _eeg_rows("All-stage", "IQR") + demo

    non_spectral: list[dict[str, str]] = []
    for source_row in _caisr_manifest_rows():
        idx = int(source_row["x_static_index"])
        non_spectral.append(_base_row(
            unique_feature_id=source_row["feature_name"], feature_name=source_row["feature_name"],
            family="CAISR20", source_npz_field="x_static", source_index=idx,
            stage_group="All-night", aggregation="identity", formula=f"x_static[{idx}]",
            unit="source-defined", validity_source="CAISR20 manifest and x_static",
            validity_expression="existing CAISR20 missing semantics", missing_semantics="existing CAISR20 semantics",
            specification_status="PENDING_G1B_REVIEW", open_decision_id="DEC-NONSPECTRAL-VALIDITY",
            source_code_reference="data2_caisr20_results/feature_manifest.csv",
        ))
    for index in range(11):
        rule = _feature_rule("X_ecg", index)
        for stat in ("Mean", "IQR"):
            name = f"hrv_{rule['name']}_{stat.lower()}"
            non_spectral.append(_base_row(
                unique_feature_id=name, feature_name=name, family="HRV11",
                source_npz_field="X_ecg", source_index=index, stage_group="All-night",
                aggregation=stat, formula=f"{stat}(X_ecg[:, {index}] over hrv_success & hrv_feature_valid)",
                unit=rule.get("original_unit", "source-defined"), validity_source="hrv_feature_valid; hrv_success; ECG window alignment",
                validity_expression=f"hrv_success[j] & hrv_feature_valid[j,{index}] with ECG window j aligned to epoch grid",
                missing_semantics="No use of X_ecg[11] circadian cosine; zero semantics require later review.",
                specification_status="PENDING_G1B_REVIEW", open_decision_id="DEC-NONSPECTRAL-VALIDITY",
                source_code_reference=f"feat_input/feature_rules_v1.json[X_ecg,{index}]; per_epoch_features/feature_extractor_ecg_neurokit.py:BASE_MODEL_COLUMNS",
            ))
    for index in range(432, 456):
        rule = _feature_rule("X_seq", index)
        channel = re.search(r"emg_(chin|lleg|rleg)_", rule["name"]).group(1)
        for stat in ("Mean", "IQR"):
            name = f"{rule['name']}_{stat.lower()}"
            non_spectral.append(_base_row(
                unique_feature_id=name, feature_name=name, family="EMG24",
                source_npz_field="X_seq", source_index=index, channel=channel, stage_group="All-night",
                aggregation=stat, formula=f"{stat}(X_seq[:, {index}] over emg_epoch_success[{channel}])",
                unit=rule.get("original_unit", "source-defined"), validity_source="emg_epoch_success; emg_channel_available",
                validity_expression=f"emg_channel_available[{channel}] & emg_epoch_success[:,{['chin','lleg','rleg'].index(channel)}]",
                missing_semantics="Zeros can be valid or an unavailable-channel fill; do not infer validity from values.",
                specification_status="PENDING_G1B_REVIEW", open_decision_id="DEC-NONSPECTRAL-VALIDITY",
                source_code_reference=f"feat_input/feature_rules_v1.json[{index}]; per_epoch_features/per_epoch_extractor.py:_extract_per_epoch_emg; per_epoch_features/feature_extractor_emg.py",
            ))
    for index in range(456, 470):
        rule = _feature_rule("X_seq", index)
        for stat in ("Mean", "IQR"):
            name = f"{rule['name']}_{stat.lower()}"
            non_spectral.append(_base_row(
                unique_feature_id=name, feature_name=name, family="Resp14",
                source_npz_field="X_seq", source_index=index, stage_group="All-night",
                aggregation=stat, formula=f"{stat}(X_seq[:, {index}] over resp_feature_valid[:,{index - 456}])",
                unit=rule.get("original_unit", "source-defined"), validity_source="resp_feature_valid",
                validity_expression=f"resp_feature_valid[:,{index - 456}]",
                missing_semantics="Zeros can be valid or an unavailable-channel fill; do not infer validity from values.",
                specification_status="PENDING_G1B_REVIEW", open_decision_id="DEC-NONSPECTRAL-VALIDITY",
                source_code_reference=f"feat_input/feature_rules_v1.json[{index}]; per_epoch_features/per_epoch_extractor.py:_extract_per_epoch_resp; per_epoch_features/feature_extractor_resp.py:MODEL_COLS",
            ))
    v5 = non_spectral + demo

    views = [("V1", v1), ("V2", v2), ("V3", v3), ("V4", v4), ("V5", v5)]
    output: list[dict[str, str]] = []
    for view_id, view_rows in views:
        for position, row in enumerate(view_rows):
            copy = dict(row)
            copy["view_id"] = view_id
            copy["view_position"] = str(position)
            output.append(copy)
    return output


def _finite_float(value: str | None) -> float | None:
    if value is None or value == "" or value.strip().lower() in {"nan", "none", "na", "n/a", "null"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _missing_text(value: str | None) -> bool:
    return value is None or value.strip().lower() in {"", "nan", "none", "na", "n/a", "null", "unknown", "unavailable", "not reported", "declined", "unreported"}


def _helper_sex(value: str | None) -> str:
    text = str(value if value is not None else "").lower()
    if text.startswith("f"):
        return "Female"
    if text.startswith("m"):
        return "Male"
    return "Unknown"


def _helper_race(value: str | None) -> str:
    text = str(value if value is not None else "").lower()
    if any(word in text for word in ("white", "caucasian")):
        return "White"
    if any(word in text for word in ("black", "african american")):
        return "Black"
    if "asian" in text:
        return "Asian"
    unavailable = ("unknown", "unavailable", "declined", "unreported", "nan", "none", "not specified", "prefer not to say")
    if not text.strip() or any(word == text or word in text for word in unavailable):
        return "Unavailable"
    return "Others"


def _demo_vector(sex: str | None, race: str | None, bmi: str | None) -> np.ndarray:
    vector = np.zeros(10, dtype=float)
    sex_code = _helper_sex(sex)
    vector[{"Female": 1, "Male": 2, "Unknown": 3}[sex_code]] = 1.0
    race_code = _helper_race(race)
    vector[{"Asian": 4, "Black": 5, "Others": 6, "Unavailable": 7, "White": 8}[race_code]] = 1.0
    vector[9] = _finite_float(bmi) or 0.0
    return vector


def _record_paths(npz_root: Path) -> list[Path]:
    return sorted(npz_root.rglob("*.npz"))


def _record_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        required = {"X_seq", "X_ecg", "x_static", "stage_code_aligned", "stage_valid", "eeg_channel_available", "eeg_clean_subsegment_count", "eeg_total_subsegment_count"}
        missing = sorted(required - keys)
        if missing:
            raise KeyError(f"{path}: missing required fields {missing}")
        x_seq_shape = tuple(z["X_seq"].shape)
        x_ecg_shape = tuple(z["X_ecg"].shape)
        x_static_shape = tuple(z["x_static"].shape)
        if x_seq_shape[1:] != EXPECTED_SHAPES["X_seq"] or x_ecg_shape[1:] != EXPECTED_SHAPES["X_ecg"] or x_static_shape != EXPECTED_SHAPES["x_static"]:
            raise ValueError(f"shape mismatch in {path}: {x_seq_shape}, {x_ecg_shape}, {x_static_shape}")
        codes = np.asarray(z["stage_code_aligned"], dtype=float).reshape(-1)
        stage_valid = np.asarray(z["stage_valid"], dtype=bool).reshape(-1)
        channel_available = np.asarray(z["eeg_channel_available"], dtype=bool).reshape(-1)
        clean = np.asarray(z["eeg_clean_subsegment_count"])
        total = np.asarray(z["eeg_total_subsegment_count"])
        if len(codes) != x_seq_shape[0] or len(stage_valid) != len(codes) or clean.shape != (len(codes), 6) or total.shape != (len(codes), 6):
            raise ValueError(f"epoch metadata shape mismatch in {path}")
        masks = compute_stage_masks(codes, stage_valid)
        eeg_valid = eeg_validity_mask(channel_available, clean)
        stage_counts = {}
        for stage, stage_mask in masks.items():
            stage_counts[stage] = {
                "raw": int(np.count_nonzero(np.isfinite(codes) & np.isin(codes, STAGE_CODES[stage]))),
                "effective": int(np.count_nonzero(stage_mask)),
                "lead": [int(np.count_nonzero(stage_mask & eeg_valid[:, i])) for i in range(6)],
                "pair": {f"{a}-{b}": intersected_count(eeg_valid & stage_mask[:, None], [CHANNEL_INDEX[a], CHANNEL_INDEX[b]]) for a, b in EEG_PAIRS},
                "four_lead": intersected_count(eeg_valid & stage_mask[:, None], [0, 1, 4, 5]),
                "six_lead": intersected_count(eeg_valid & stage_mask[:, None], list(range(6))),
            }
        return {
            "site_id": str(_scalar(z["site_id"])),
            "bids_folder": str(_scalar(z["bids_folder"])),
            "session_id": str(_scalar(z["session_id"])),
            "record_id": str(_scalar(z["record_id"])),
            "stage_counts": stage_counts,
            "stage_valid_true_invalid_code": int(np.count_nonzero(stage_valid & ~(np.isfinite(codes) & np.isin(codes, ALL_STAGE_CODES)))),
            "stage_valid_false_valid_code": int(np.count_nonzero(~stage_valid & np.isfinite(codes) & np.isin(codes, ALL_STAGE_CODES))),
            "extraction_version": str(_scalar(z["extraction_version"])),
            "validity_schema_version": str(_scalar(z["validity_schema_version"])),
            "x_static": np.asarray(z["x_static"], dtype=float).copy(),
        }


def _quantiles(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {key: 0 for key in ("min", "p05", "p25", "median", "p75", "p95", "max")}
    q = np.percentile(np.asarray(values, dtype=float), [5, 25, 50, 75, 95])
    return {"min": int(min(values)), "p05": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4]), "max": int(max(values))}


def _add_stage_rows(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    sites = sorted({s["site_id"] for s in summaries})
    for site in ["ALL"] + sites:
        subset = [s for s in summaries if site == "ALL" or s["site_id"] == site]
        total = len(subset)
        for stage in STAGE_ORDER:
            values = [s["stage_counts"][stage]["effective"] for s in subset]
            raw_values = [s["stage_counts"][stage]["raw"] for s in subset]
            for metric, value in _quantiles(raw_values).items():
                rows.append({"section": "stage_raw_distribution", "site_id": site, "stage_group": stage, "threshold": "", "metric": metric, "count": value, "total": total, "proportion": "", "notes": "raw finite stage-code count; stage_valid not applied"})
            for metric, value in _quantiles(values).items():
                rows.append({"section": "stage_distribution", "site_id": site, "stage_group": stage, "threshold": "", "metric": metric, "count": value, "total": total, "proportion": "", "notes": "effective = stage_valid and valid stage code"})
            for k in (0, 1, 2):
                count = sum(v == k for v in values)
                rows.append({"section": "stage_distribution", "site_id": site, "stage_group": stage, "threshold": "", "metric": f"exactly_{k}", "count": count, "total": total, "proportion": count / total if total else 0, "notes": "descriptive; no patient exclusion"})
            for threshold in THRESHOLDS:
                count = sum(v >= threshold for v in values)
                rows.append({"section": "stage_threshold", "site_id": site, "stage_group": stage, "threshold": threshold, "metric": "effective_epoch_ge_threshold", "count": count, "total": total, "proportion": count / total if total else 0, "notes": "candidate comparison only"})
        for threshold in THRESHOLDS:
            for metric, func, note in [
                ("EEG54_all_six_leads", lambda s: all(s["stage_counts"][g]["six_lead"] >= threshold for g in ("Wake", "REM", "N2", "N3", "NREM", "All-stage")), "all six stage groups have six-lead support"),
                ("spatial_asymmetry_candidate", lambda s: all(s["stage_counts"][g]["pair"][f"{a}-{b}"] >= threshold for g in ("N2", "REM") for a, b in EEG_PAIRS), "all N2/REM lead pairs support candidate"),
                ("spatial_ap_gradient_candidate", lambda s: all(s["stage_counts"][g]["four_lead"] >= threshold for g in ("N2", "N3", "REM")), "all N2/N3/REM four-lead intersections support candidate"),
            ]:
                count = sum(bool(func(s)) for s in subset)
                rows.append({"section": "candidate_feature_support", "site_id": site, "stage_group": "multiple", "threshold": threshold, "metric": metric, "count": count, "total": total, "proportion": count / total if total else 0, "notes": note})
            insufficient = 0
            for s in subset:
                required = [s["stage_counts"][g]["six_lead"] >= threshold for g in ("Wake", "REM", "N2", "N3", "NREM", "All-stage")]
                required += [s["stage_counts"][g]["pair"][f"{a}-{b}"] >= threshold for g in ("N2", "REM") for a, b in EEG_PAIRS]
                required += [s["stage_counts"][g]["four_lead"] >= threshold for g in ("N2", "N3", "REM")]
                insufficient += not all(required)
            rows.append({"section": "candidate_feature_support", "site_id": site, "stage_group": "multiple", "threshold": threshold, "metric": "at_least_one_required_feature_insufficient", "count": insufficient, "total": total, "proportion": insufficient / total if total else 0, "notes": "required set is documented above; no threshold selected"})
    for s in summaries:
        for stage in STAGE_ORDER:
            data = s["stage_counts"][stage]
            for i, channel in enumerate(EEG_CHANNELS):
                rows.append({"section": "lead_support", "site_id": s["site_id"], "stage_group": stage, "threshold": "", "metric": f"{channel}_effective_epochs", "count": data["lead"][i], "total": 1, "proportion": "", "notes": "record-level audit; same-epoch mask"})
            for pair, count in data["pair"].items():
                rows.append({"section": "intersection_support", "site_id": s["site_id"], "stage_group": stage, "threshold": "", "metric": f"pair_{pair}", "count": count, "total": 1, "proportion": "", "notes": "same-epoch intersection"})
            for label, count in (("F3_F4_O1_O2", data["four_lead"]), ("all_six", data["six_lead"])):
                rows.append({"section": "intersection_support", "site_id": s["site_id"], "stage_group": stage, "threshold": "", "metric": label, "count": count, "total": 1, "proportion": "", "notes": "same-epoch intersection"})


def _read_demographics(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    required = ["SiteID", "BidsFolder", "SessionID", "Sex", "Race", "BMI"]
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(field not in reader.fieldnames for field in required):
            raise ValueError(f"demographics.csv missing required columns; found {reader.fieldnames}")
        return [{field: row.get(field, "") for field in required} for row in reader], required


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def run_audit(npz_root: Path = NPZ_ROOT, demographics_path: Path = DEMOGRAPHICS_PATH, output_root: Path | None = None) -> dict[str, Any]:
    """Run the bounded G1-A audit and write only the approved outputs."""
    output_root = output_root or ROOT / "reproduction/siren_a"
    audit_dir = output_root / "audit"
    docs_dir = output_root / "docs"
    paths = _record_paths(npz_root)
    if len(paths) != 6600:
        raise RuntimeError(f"expected 6600 NPZ records, found {len(paths)}")
    summaries = [_record_summary(path) for path in paths]
    versions = Counter((s["extraction_version"], s["validity_schema_version"]) for s in summaries)
    site_counts = Counter(s["site_id"] for s in summaries)
    consistency = Counter()
    for s in summaries:
        consistency[(s["site_id"], "stage_valid_true_invalid_code")] += s["stage_valid_true_invalid_code"]
        consistency[(s["site_id"], "stage_valid_false_valid_code")] += s["stage_valid_false_valid_code"]

    stage_rows: list[dict[str, Any]] = []
    _add_stage_rows(stage_rows, summaries)
    _write_csv(audit_dir / "g1a_stage_support.csv", stage_rows, ["section", "site_id", "stage_group", "threshold", "metric", "count", "total", "proportion", "notes"])

    demo_rows, _ = _read_demographics(demographics_path)
    csv_keys = [(row["BidsFolder"], row["SessionID"]) for row in demo_rows]
    csv_key_counts = Counter(csv_keys)
    npz_key_counts = Counter((s["bids_folder"], s["session_id"]) for s in summaries)
    matched = set(csv_key_counts) & set(npz_key_counts)
    join_rows: list[dict[str, Any]] = []
    for site in sorted(set(site_counts) | {row["SiteID"] for row in demo_rows}):
        npz_site = [s for s in summaries if s["site_id"] == site]
        csv_site = [row for row in demo_rows if row["SiteID"] == site]
        csv_site_keys = {(row["BidsFolder"], row["SessionID"]) for row in csv_site}
        npz_site_keys = {(s["bids_folder"], s["session_id"]) for s in npz_site}
        metrics = {
            "npz_records": len(npz_site), "csv_rows": len(csv_site),
            "csv_unique_keys": len(csv_site_keys), "npz_unique_keys": len(npz_site_keys),
            "matched_unique_keys": len(csv_site_keys & npz_site_keys),
            "csv_duplicate_rows": sum(v - 1 for k, v in csv_key_counts.items() if v > 1 and k in csv_site_keys),
            "csv_without_npz": len(csv_site_keys - npz_site_keys),
            "npz_without_csv": len(npz_site_keys - csv_site_keys),
        }
        for metric, count in metrics.items():
            join_rows.append({"site_id": site, "metric": metric, "count": count, "denominator": max(len(csv_site), len(npz_site)), "proportion": count / max(len(csv_site), len(npz_site)) if max(len(csv_site), len(npz_site)) else 0, "notes": "aggregate only; no patient-level identifiers written"})

    demo_cat_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    npz_by_key = {(s["bids_folder"], s["session_id"]): s for s in summaries}
    for site in sorted({row["SiteID"] for row in demo_rows}):
        site_rows = [row for row in demo_rows if row["SiteID"] == site]
        for field in ("Sex", "Race"):
            counts = Counter("<MISSING>" if _missing_text(row[field]) else row[field] for row in site_rows)
            for category, count in sorted(counts.items()):
                demo_cat_rows.append({"site_id": site, "variable": field, "category": category, "metric": "raw_category_count", "count": count, "denominator": len(site_rows), "proportion": count / len(site_rows) if site_rows else 0, "minimum": "", "p05": "", "median": "", "p95": "", "maximum": "", "notes": "aggregate category only"})
        bmi_values = [_finite_float(row["BMI"]) for row in site_rows]
        raw_bmi = [row["BMI"] for row in site_rows]
        bmi_metrics = {
            "missing_or_nonfinite": sum(v is None for v in bmi_values),
            "zero": sum(v == 0 for v in bmi_values if v is not None),
            "negative": sum(v < 0 for v in bmi_values if v is not None),
            "suspicious_lt_10": sum(0 < v < 10 for v in bmi_values if v is not None),
            "suspicious_gt_80": sum(v > 80 for v in bmi_values if v is not None),
        }
        finite_bmi = [v for v in bmi_values if v is not None]
        q = _quantiles([int(round(v * 100)) for v in finite_bmi]) if finite_bmi else {}
        for metric, count in bmi_metrics.items():
            demo_cat_rows.append({"site_id": site, "variable": "BMI", "category": "", "metric": metric, "count": count, "denominator": len(site_rows), "proportion": count / len(site_rows) if site_rows else 0, "minimum": min(finite_bmi) if finite_bmi else "", "p05": (q.get("p05", "") / 100 if q else ""), "median": (q.get("median", "") / 100 if q else ""), "p95": (q.get("p95", "") / 100 if q else ""), "maximum": max(finite_bmi) if finite_bmi else "", "notes": "descriptive flags only; no BMI correction rule"})
        matches = [row for row in site_rows if (row["BidsFolder"], row["SessionID"]) in npz_by_key and npz_by_key[(row["BidsFolder"], row["SessionID"])] ["site_id"] == site]
        for variable in ("Sex", "Race", "BMI"):
            if variable == "Sex":
                equal = sum(np.array_equal(_demo_vector(row["Sex"], row["Race"], row["BMI"])[1:4], npz_by_key[(row["BidsFolder"], row["SessionID"])] ["x_static"][1:4]) for row in matches)
            elif variable == "Race":
                equal = sum(np.array_equal(_demo_vector(row["Sex"], row["Race"], row["BMI"])[4:9], npz_by_key[(row["BidsFolder"], row["SessionID"])] ["x_static"][4:9]) for row in matches)
            else:
                equal = sum(np.isclose(_demo_vector(row["Sex"], row["Race"], row["BMI"])[9], npz_by_key[(row["BidsFolder"], row["SessionID"])] ["x_static"][9], equal_nan=False) for row in matches)
            compare_rows.append({"site_id": site, "metric": f"{variable}_encoding_consistent", "count": equal, "denominator": len(matches), "proportion": equal / len(matches) if matches else 0, "notes": "compared against x_static; aggregate only"})
        compare_rows.extend([
            {"site_id": site, "metric": "sex_unknown_or_other_raw", "count": sum(_helper_sex(row["Sex"]) == "Unknown" for row in matches), "denominator": len(matches), "proportion": sum(_helper_sex(row["Sex"]) == "Unknown" for row in matches) / len(matches) if matches else 0, "notes": "Other and missing are merged by existing helper"},
            {"site_id": site, "metric": "race_unavailable_after_helper", "count": sum(_helper_race(row["Race"]) == "Unavailable" for row in matches), "denominator": len(matches), "proportion": sum(_helper_race(row["Race"]) == "Unavailable" for row in matches) / len(matches) if matches else 0, "notes": "existing helper mapping"},
            {"site_id": site, "metric": "bmi_missing_to_zero_expected", "count": sum(_finite_float(row["BMI"]) is None for row in matches), "denominator": len(matches), "proportion": sum(_finite_float(row["BMI"]) is None for row in matches) / len(matches) if matches else 0, "notes": "existing helper load_bmi behavior"},
        ])
    _write_csv(audit_dir / "g1a_join_consistency.csv", join_rows + compare_rows, ["site_id", "metric", "count", "denominator", "proportion", "notes"])
    _write_csv(audit_dir / "g1a_demographics_categories.csv", demo_cat_rows, ["site_id", "variable", "category", "metric", "count", "denominator", "proportion", "minimum", "p05", "median", "p95", "maximum", "notes"])

    manifest = build_draft_manifest()
    manifest_fields = list(_base_row().keys())
    _write_csv(docs_dir / "G1A_FEATURE_MANIFEST_DRAFT.csv", manifest, manifest_fields)

    view_counts = Counter(row["view_id"] for row in manifest)
    unique_ids = {row["unique_feature_id"] for row in manifest}
    report = _build_report(summaries, versions, site_counts, demo_rows, matched, csv_key_counts, npz_key_counts, view_counts, len(unique_ids), demographics_path)
    (docs_dir / "G1A_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    (docs_dir / "G1A_DECISION_REQUESTS.md").write_text(_build_decisions(summaries, demo_rows, compare_rows), encoding="utf-8")
    (docs_dir / "G1A_FEATURE_SPEC_DRAFT.md").write_text(_build_spec(view_counts), encoding="utf-8")
    metadata = {
        "audit_name": "SIREN Branch A G1-A",
        "audit_version": "g1a-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_run": _git_head(),
        "npz_root": str(npz_root),
        "demographics_path": str(demographics_path),
        "npz_record_count": len(summaries),
        "site_record_counts": dict(sorted(site_counts.items())),
        "extraction_validity_versions": {f"{a}|{b}": n for (a, b), n in versions.items()},
        "demographics_row_count": len(demo_rows),
        "demographics_unique_primary_keys": len(set(csv_keys)),
        "npz_unique_primary_keys": len(set(npz_key_counts)),
        "matched_unique_primary_keys": len(matched),
        "manifest_rows": len(manifest),
        "manifest_unique_feature_ids": len(unique_ids),
        "min_valid_epochs": "NOT FROZEN",
        "demo5_encoding": "NOT FROZEN",
        "night_level_cache": "NOT GENERATED",
        "model_training": "NOT STARTED",
        "privacy": "aggregate outputs only; no raw demographics or patient-level identifiers written",
    }
    (audit_dir / "g1a_run_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def _build_spec(view_counts: Counter) -> str:
    return f"""# G1-A Feature Specification Draft\n\nThis is a draft for review, not a frozen implementation specification. It is derived from the committed extractor and `feature_rules_v1.json`.\n\n## View sizes\n\n| View | Rows |\n|---|---:|\n| V1 Wake | {view_counts['V1']} |\n| V2 REM | {view_counts['V2']} |\n| V3 N2/N3/NREM | {view_counts['V3']} |\n| V4 All-stage | {view_counts['V4']} |\n| V5 Non-spectral | {view_counts['V5']} |\n| Total | {sum(view_counts.values())} |\n\nThe draft has 807 unique feature IDs and repeats the same five pending Demo5 IDs in each view.\n\n## Fixed source boundaries\n\n- EEG54 uses the six committed stage groups, six leads, nine source features, and Mean/IQR.\n- Spatial rows are the 36 explicitly listed SIREN-inspired candidates: 24 interhemispheric asymmetries and 12 anterior–posterior gradients. They are not claimed to reproduce an original-paper formula.\n- Non-spectral rows are CAISR20, HRV11 Mean/IQR, EMG24 Mean/IQR, and Resp14 Mean/IQR. X_ecg index 11 (`hrv_circadian_cos`) is excluded.\n- Demo5 has five repeated pending positions but no frozen encoding. Age is not a model input.\n\n## Open status\n\n`MIN_VALID_EPOCHS`, small-n IQR behavior, spatial support threshold, Demo5 coding, and any unresolved validity semantics remain `PENDING_G1B_REVIEW`.\n"""


def _build_decisions(summaries: list[dict[str, Any]], demo_rows: list[dict[str, str]], compare_rows: list[dict[str, Any]]) -> str:
    stage_values = {stage: [s["stage_counts"][stage]["effective"] for s in summaries] for stage in STAGE_ORDER}
    unknown = sum(_helper_sex(row["Sex"]) == "Unknown" for row in demo_rows)
    race_unavailable = sum(_helper_race(row["Race"]) == "Unavailable" for row in demo_rows)
    bmi_missing = sum(_finite_float(row["BMI"]) is None for row in demo_rows)
    lines = ["# G1-A Decision Requests", "", "The following are evidence-backed requests for GPT and user review. No choice is made here.", ""]
    lines.append("## 1. Minimum valid EEG epochs")
    lines.append("The audit counts are available in `audit/g1a_stage_support.csv`; descriptive distributions are:")
    for stage in STAGE_ORDER:
        q = _quantiles(stage_values[stage])
        lines.append(f"- {stage}: min={q['min']}, P05={q['p05']:.2f}, median={q['median']:.2f}, P95={q['p95']:.2f}, max={q['max']}.")
    lines += ["Candidates to review: 1, 2, 3, 5, 10, 20, 30, or 60 effective epochs. No threshold was selected and no patient was excluded.", ""]
    lines.append("## 2. IQR with very few valid epochs")
    lines.append("Choose whether one/two observations use an explicitly defined deterministic IQR convention, are represented as missing, or are withheld until a larger support threshold. This audit does not choose among them.")
    lines.append("## 3. Spatial support threshold")
    lines.append("The 36 candidates use same-epoch intersections: three left/right pairs for N2 and REM, and four-lead F3/F4/O1/O2 intersections for N2/N3/REM. Review whether their support threshold must equal EEG54's threshold.")
    lines.append("## 4. Demo5 sex encoding")
    lines.append(f"Existing helper mapping merges Other and missing into Other/Unknown; {unknown} of {len(demo_rows)} demographic rows map to Unknown. Candidates: scalar Female/Male with an explicit Unknown code; scalar Female/Male with Unknown missing; or a pre-approved one-hot allocation that still preserves five columns.")
    lines.append("## 5. Demo5 race three columns")
    lines.append(f"Existing helper categories are Asian/Black/Others/Unavailable/White; {race_unavailable} of {len(demo_rows)} rows map to Unavailable. Candidates: [White, Black, Asian] with Others/Unavailable all-zero; [White, Black, Other-or-Unavailable]; or [Asian, Black, White] with an explicit missing policy. The three columns and order are not frozen.")
    lines.append("## 6. Demo5 unknown and BMI representation")
    lines.append(f"Existing BMI helper maps invalid/nonfinite values to zero; {bmi_missing} rows are nonfinite/missing under the audit parser. Candidates include preserving zero as the existing sentinel, using a fold-fitted imputation plus a missing indicator only if the dimensionality is changed by approval, or retaining a deterministic missing state outside the five columns. No choice is made.")
    lines.append("## 7. Other unresolved semantics")
    lines.append("Review CAISR20 missing semantics, lawful zero versus unavailable fills in EMG/Resp, HRV window-level validity and alignment, and any source unit that remains `source-defined`. These are marked `PENDING_G1B_REVIEW` in the CSV manifest.")
    lines += ["", "Labels, model performance, and LOSO metrics were not used."]
    return "\n".join(lines) + "\n"


def _build_report(summaries: list[dict[str, Any]], versions: Counter, site_counts: Counter, demo_rows: list[dict[str, str]], matched: set[tuple[str, str]], csv_key_counts: Counter, npz_key_counts: Counter, view_counts: Counter, unique_manifest_count: int, demographics_path: Path) -> str:
    invalid_true = sum(s["stage_valid_true_invalid_code"] for s in summaries)
    false_valid = sum(s["stage_valid_false_valid_code"] for s in summaries)
    coverage_lines = ["", "### Candidate threshold record coverage", "", "| threshold | Wake | REM | N1 | N2 | N3 | NREM | All-stage |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for threshold in THRESHOLDS:
        cells = []
        for stage in STAGE_ORDER:
            count = sum(s["stage_counts"][stage]["effective"] >= threshold for s in summaries)
            cells.append(f"{count}/{len(summaries)} ({count / len(summaries):.3f})")
        coverage_lines.append("| " + str(threshold) + " | " + " | ".join(cells) + " |")
    coverage_lines += ["", "The table reports count/total and proportion for every candidate threshold; it is descriptive and does not select a threshold.", "", "### Site-stage median effective epochs", "", "| site | Wake | REM | N1 | N2 | N3 | NREM | All-stage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for site in sorted(site_counts):
        subset = [s for s in summaries if s["site_id"] == site]
        cells = []
        for stage in STAGE_ORDER:
            median = statistics.median([s["stage_counts"][stage]["effective"] for s in subset])
            cells.append(f"{median:.1f}")
        coverage_lines.append("| " + site + " | " + " | ".join(cells) + " |")
    coverage_lines += ["", "### Demographic aggregate distributions", ""]
    for site in sorted(site_counts):
        site_rows = [row for row in demo_rows if row["SiteID"] == site]
        sex_counts = Counter("<MISSING>" if _missing_text(row["Sex"]) else row["Sex"] for row in site_rows)
        race_counts = Counter("<MISSING>" if _missing_text(row["Race"]) else row["Race"] for row in site_rows)
        bmi_values = [_finite_float(row["BMI"]) for row in site_rows]
        finite_bmi = [value for value in bmi_values if value is not None]
        bmi_missing = sum(value is None for value in bmi_values)
        bmi_zero = sum(value == 0 for value in finite_bmi)
        bmi_suspicious = sum(value < 0 or value > 80 or 0 < value < 10 for value in finite_bmi)
        coverage_lines.append(f"- {site} (n={len(site_rows)}): Sex {dict(sorted(sex_counts.items()))}; Race {dict(sorted(race_counts.items()))}; BMI missing/nonfinite={bmi_missing}, zero={bmi_zero}, descriptive suspicious-range flags={bmi_suspicious}.")
    lines = [
        "# G1-A Audit Report",
        "",
        f"- Git/input scope: v2.1 NPZ only; repository HEAD at audit run: `{_git_head()}`.",
        f"- NPZ root: `{NPZ_ROOT}`; records scanned: **{len(summaries)}**.",
        f"- Versions: `{dict(versions)}`; expected extraction `{EXTRACTION_VERSION}`, validity `{VALIDITY_SCHEMA_VERSION}`.",
        f"- Site counts: {dict(sorted(site_counts.items()))}.",
        f"- Stage consistency observations: stage_valid=true with invalid code = {invalid_true}; stage_valid=false with valid code = {false_valid}.",
        "",
        "## Stage and EEG support",
        "",
        "`stage_code_aligned` and `stage_valid` were checked together. Invalid codes 0, 9, and nonfinite values were excluded from all stage groups. NREM is codes 1/2/3; All-stage is codes 1/2/3/4/5. EEG validity is `eeg_channel_available & eeg_clean_subsegment_count > 0` per lead and epoch. `eeg_total_subsegment_count` was retained for source verification but no clean-ratio threshold was added.",
        "",
        "Per-record distributions, site summaries, threshold comparisons, six-lead counts, pair intersections, and F3/F4/O1/O2 intersections are in `audit/g1a_stage_support.csv`. All candidate thresholds 1, 2, 3, 5, 10, 20, 30, and 60 are descriptive only. No threshold was selected and no patient was excluded.",
        "Raw stage-code counts are reported separately from effective stage counts; invalid codes remain excluded and stage_valid is not applied to the raw count.",
        *coverage_lines,
        "",
        "The audit uses exact same-epoch intersections for all spatial support counts; it does not substitute the minimum of marginal lead counts. The NPZ `mask` field was not used for EEG, EMG, Resp, or stage filtering; source code identifies it as ECG alignment only.",
        "",
        "## Demographics audit",
        "",
        f"- Candidate source: `{demographics_path}`; only SiteID, BidsFolder, SessionID, Sex, Race, BMI were read.",
        f"- CSV rows: {len(demo_rows)}; unique primary keys: {len(csv_key_counts)}; duplicate-row excess: {sum(max(0, n - 1) for n in csv_key_counts.values())}.",
        f"- NPZ unique keys: {len(npz_key_counts)}; exact key intersection: {len(matched)}; unmatched and encoding summaries are aggregate-only in `audit/g1a_join_consistency.csv`.",
        "- Existing helper semantics were applied exactly for sex, race, and BMI comparison to x_static[1:4], x_static[4:9], and x_static[9]. Age was not included in Demo5.",
        "- Raw category and BMI missing/zero/nonfinite/suspicious-range summaries are in `audit/g1a_demographics_categories.csv`; no patient-level identifiers are written.",
        "",
        "## Draft Manifest",
        "",
        f"View row counts: {dict(view_counts)}; total rows: {sum(view_counts.values())}; unique feature IDs: {unique_manifest_count}.",
        "The manifest has 827 rows, contiguous zero-based positions within each view, and the same five Demo5 unique IDs repeated across views. EEG648, candidate spatial36, and non-spectral118 are source-defined drafts. Demo5 and unresolved validity/small-n rules are `PENDING_G1B_REVIEW`.",
        "",
        "## Findings and limits",
        "",
        "- 1103 NPZ is unavailable and was not searched for or substituted.",
        "- This run did not calculate patient-level features, create a Night-level Feature Cache, fit transforms, use labels, train models, or rerun Validity v0.2 QC.",
        "- Any anomaly is preserved as evidence for review; this Gate does not repair source data or mappings.",
        "",
        "See `docs/G1A_DECISION_REQUESTS.md` for the decisions intentionally left open.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-root", type=Path, default=NPZ_ROOT)
    parser.add_argument("--demographics", type=Path, default=DEMOGRAPHICS_PATH)
    parser.add_argument("--output-root", type=Path, default=ROOT / "reproduction/siren_a")
    args = parser.parse_args()
    print(json.dumps(run_audit(args.npz_root, args.demographics, args.output_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
