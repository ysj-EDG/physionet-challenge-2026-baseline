#!/usr/bin/env python3
"""D0-D5 low-cost data2 baselines using frozen P3/P5/P6 implementations."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import warnings

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import feature_scaling as fs
import pooled_logistic_v2 as pooled
from feat_input.feat_mody import run_loso_1103 as p5
from feat_input.feat_mody import run_p3_v2 as p3
from feat_input.feat_mody import run_p6_site_audit as p6


DATASET = "NEW1103 timegrid_v2"
CACHE_ROOT = ROOT / "npz_1103_timegrid_v2"
SOURCE_MANIFEST = CACHE_ROOT / "manifest.jsonl"
SOURCE_SUMMARY = CACHE_ROOT / "manifest_summary.json"
RULES_PATH = ROOT / "feat_input/feature_rules_v1.json"
OUT = ROOT / "output/bridge_1103_timegrid_v2"
QC_OUT = OUT / "qc"
SITE_OUT = OUT / "site_audit"
LOSO_OUT = OUT / "loso_lr"
SUMMARY_CACHE = OUT / "cache/bridge_patient_summaries.npz"
REGISTRY_OUT = ROOT / "feat_input2/output/baselines"
SITES = ("I0002", "I0006", "S0001")
EXPECTED_COUNTS = {
    "I0002": (319, 52, 267),
    "I0006": (1142, 112, 1030),
    "S0001": (5139, 334, 4805),
}
ARMS = {
    "demo10": "P3_demo10",
    "compact30": "P3_compact30",
    "global59": "P3_global59",
}
SELECTED_STATIC = tuple(dict.fromkeys([*range(10), *pooled.COMPACT_INDICES]))
SELECTED_SEQ = tuple(sorted({channel * 9 + offset for channel in range(6)
                            for _, offset in pooled.METRICS}))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame, decimals: int = 4) -> str:
    def render(value):
        if pd.isna(value):
            return "N/A"
        if isinstance(value, (float, np.floating)):
            return format(float(value), f".{decimals}f")
        return str(value)
    headers = [str(column) for column in frame.columns]
    rows = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    rows.extend("| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(rows)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def json_rows() -> tuple[list[dict], list[dict]]:
    records, events = [], []
    with SOURCE_MANIFEST.open() as handle:
        for line in handle:
            item = json.loads(line)
            (records if "record_id" in item else events).append(item)
    return records, events


def validate_source_manifest() -> tuple[list[dict], list[dict]]:
    if (OUT / "run_metadata.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed output: {OUT}")
    for path in (SOURCE_MANIFEST, SOURCE_SUMMARY, RULES_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(SOURCE_SUMMARY.read_text())
    if summary.get("success") != 6600 or summary.get("failed") != 0:
        raise RuntimeError(f"Unexpected extraction summary: {summary}")
    records, events = json_rows()
    if len(records) != 6600 or len({row["record_id"] for row in records}) != 6600:
        raise RuntimeError("Manifest does not uniquely contain 6600 records")
    if set(row.get("status") for row in records) != {"success"}:
        raise RuntimeError("Manifest contains a non-success record")
    expected_paths = {
        CACHE_ROOT / row["site_id"] / f"{row['record_id']}.npz" for row in records
    }
    actual_paths = set(CACHE_ROOT.glob("*/*.npz"))
    if expected_paths != actual_paths:
        raise RuntimeError(
            f"Manifest/NPZ mismatch: missing={len(expected_paths-actual_paths)} "
            f"extra={len(actual_paths-expected_paths)}"
        )
    counts = collections.Counter((row["site_id"], int(row["label"])) for row in records)
    for site, (n, positive, negative) in EXPECTED_COUNTS.items():
        if counts[(site, 1)] != positive or counts[(site, 0)] != negative:
            raise RuntimeError(f"Site/label mismatch for {site}: {counts}")
        if counts[(site, 1)] + counts[(site, 0)] != n:
            raise RuntimeError(f"Site count mismatch for {site}")
    return sorted(records, key=lambda row: (row["site_id"], row["record_id"])), events


def empty_contributions(rules: list[dict]):
    keys = [("x_static", index) for index in SELECTED_STATIC]
    keys += [("X_seq", index) for index in SELECTED_SEQ]
    return {
        site: {key: {"values": [], "weights": [], "records": 0} for key in keys}
        for site in SITES
    }


def add_fit_contributions(accumulator, site, record_id, seq, ecg, static, mask, rules_by_key):
    fallback = (
        seq.shape == (1, 483) and np.all(seq == 0) and len(ecg) == 0 and not np.any(mask)
    )
    caisr_sentinel = bool(
        np.isfinite(static[10:]).all() and static[10] == 0 and np.all(static[10:] == 0)
    )
    static_matrix = static[None, :]
    for index in SELECTED_STATIC:
        prepared, valid, _ = fs._prepare_values(
            static_matrix[:, index], rules_by_key[("x_static", index)]
        )
        if index >= 10 and caisr_sentinel:
            valid[:] = False
        if np.any(valid):
            usable = prepared[valid]
            item = accumulator[site][("x_static", index)]
            item["values"].append(usable)
            item["weights"].append(np.full(len(usable), 1.0 / len(usable)))
            item["records"] += 1
    if fallback:
        return
    indices = fs._stable_indices(record_id, "X_seq", len(seq), 128, 20260907)
    sampled = seq[indices]
    for index in SELECTED_SEQ:
        prepared, valid, _ = fs._prepare_values(
            sampled[:, index], rules_by_key[("X_seq", index)]
        )
        if np.any(valid):
            usable = prepared[valid]
            item = accumulator[site][("X_seq", index)]
            item["values"].append(usable)
            item["weights"].append(np.full(len(usable), 1.0 / len(usable)))
            item["records"] += 1


def fitted_parameter(rule: dict, parts: list[dict]) -> dict:
    value_parts, weight_parts = [], []
    contributing = 0
    for part in parts:
        value_parts.extend(part["values"])
        weight_parts.extend(part["weights"])
        contributing += part["records"]
    values = np.concatenate(value_parts) if value_parts else np.zeros(0, dtype=np.float64)
    weights = np.concatenate(weight_parts) if weight_parts else np.zeros(0, dtype=np.float64)
    if rule["scaling"] == "identity":
        center, scale, method = 0.0, 1.0, "identity"
        quantiles = [None] * 5
    elif len(values) == 0:
        center, scale, method = 0.0, 1.0, "all_missing_fallback"
        quantiles = [None] * 5
    else:
        q05, q25, q50, q75, q95 = [fs._weighted_quantile(values, weights, q) for q in (.05, .25, .5, .75, .95)]
        center = q50
        tolerance = 1e-6 * max(1.0, abs(center))
        mean = np.average(values, weights=weights)
        candidates = [
            ("iqr", q75 - q25), ("p95_p05", q95 - q05),
            ("std", float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))),
        ]
        method, scale = "unit_fallback", 1.0
        for candidate_method, candidate_scale in candidates:
            if np.isfinite(candidate_scale) and candidate_scale > tolerance:
                method, scale = candidate_method, float(candidate_scale)
                break
        quantiles = [float(value) for value in (q05, q25, q50, q75, q95)]
    return {
        "branch": rule["branch"], "index": int(rule["index"]), "name": rule["name"],
        "center": float(center), "scale": float(scale), "scale_method": method,
        "fit_valid_observations": int(len(values)),
        "fit_contributing_records": int(contributing),
        "p05": quantiles[0], "p25": quantiles[1], "p50": quantiles[2],
        "p75": quantiles[3], "p95": quantiles[4],
    }


def build_fold_parameters(accumulator, rules_by_key) -> dict[str, dict]:
    output = {}
    for holdout in SITES:
        training_sites = [site for site in SITES if site != holdout]
        parameters = {}
        for key in [("x_static", index) for index in SELECTED_STATIC] + [
            ("X_seq", index) for index in SELECTED_SEQ
        ]:
            parameters[key] = fitted_parameter(
                rules_by_key[key], [accumulator[site][key] for site in training_sites]
            )
        output[holdout] = parameters
    return output


def transform_column(values, rule, parameter, force_missing=False):
    prepared, valid, _ = fs._prepare_values(values, rule)
    if force_missing:
        valid[:] = False
    if rule["scaling"] == "robust":
        filled = np.where(valid, prepared, parameter["center"])
        return (filled - parameter["center"]) / parameter["scale"]
    return np.where(valid, prepared, float(rule.get("identity_fill", 0.0)))


def selective_p3_features(seq, ecg, static, mask, stage, valid, channels, parameters, rules_by_key):
    static_out = np.asarray(static, dtype=np.float64).copy()
    seq_out = np.asarray(seq, dtype=np.float64).copy()
    caisr_sentinel = bool(
        np.isfinite(static[10:]).all() and static[10] == 0 and np.all(static[10:] == 0)
    )
    fallback = (
        seq.shape == (1, 483) and np.all(seq == 0) and len(ecg) == 0 and not np.any(mask)
    )
    for index in SELECTED_STATIC:
        static_out[index] = transform_column(
            np.asarray([static[index]]), rules_by_key[("x_static", index)],
            parameters[("x_static", index)], force_missing=index >= 10 and caisr_sentinel,
        )[0]
    if fallback:
        seq_out[:, SELECTED_SEQ] = 0.0
    else:
        for index in SELECTED_SEQ:
            seq_out[:, index] = transform_column(
                seq[:, index], rules_by_key[("X_seq", index)], parameters[("X_seq", index)]
            )
    return {
        arm: pooled.assemble_features(static_out, stage, valid, channels, seq_out, p3_arm)
        for arm, p3_arm in ARMS.items()
    }


def site_family_features(seq, ecg, static, stage, valid, channels):
    return {
        "demo10": static[:10],
        "compact30": np.r_[static[:10], static[pooled.COMPACT_INDICES]],
        "static196": static,
        "eeg_spectral54": p6.summarize_continuous(seq[:, 0:54]),
        "coherence360": p6.summarize_continuous(seq[:, 54:414]),
        "bsr18": p6.summarize_continuous(seq[:, 414:432]),
        "emg24": p6.summarize_continuous(seq[:, 432:456]),
        "respiration14": p6.summarize_continuous(seq[:, 456:470]),
        "stage_event13": np.nanmean(seq[:, 470:483], axis=0),
        "ecg12": p6.summarize_continuous(ecg),
        "global59": pooled.assemble_features(static, stage, valid, channels, seq, "P3_global59"),
    }


def smoke_test(records, rules, rules_by_key):
    selected = []
    for site in SITES:
        for label in (0, 1):
            selected.extend([row for row in records if row["site_id"] == site and int(row["label"]) == label][:2])
    acc = empty_contributions(rules)
    with tempfile.TemporaryDirectory(prefix="data2_lowcost_smoke_") as tmp:
        tmp_path = Path(tmp)
        for row in selected:
            path = CACHE_ROOT / row["site_id"] / f"{row['record_id']}.npz"
            (tmp_path / f"{row['record_id']}.npz").symlink_to(path)
            with np.load(path, allow_pickle=False) as z:
                add_fit_contributions(
                    acc, row["site_id"], row["record_id"], np.asarray(z["X_seq"]),
                    np.asarray(z["X_ecg"]), np.asarray(z["x_static"]), np.asarray(z["mask"]),
                    rules_by_key,
                )
        params = {}
        for key in [("x_static", index) for index in SELECTED_STATIC] + [("X_seq", index) for index in SELECTED_SEQ]:
            params[key] = fitted_parameter(rules_by_key[key], [acc[site][key] for site in SITES])
        shared = fs.FeatureScaler.fit_from_cache(
            [{"record_id": row["record_id"]} for row in selected], tmp_path, RULES_PATH,
            sample_cap=128, seed=20260907,
        )
        shared_params = {(item["branch"], int(item["index"])): item for item in shared.parameters}
        for key, item in params.items():
            reference = shared_params[key]
            if not np.isclose(item["center"], reference["center"], rtol=0, atol=1e-12):
                raise RuntimeError(f"Selective scaler smoke center mismatch: {key}")
            if not np.isclose(item["scale"], reference["scale"], rtol=0, atol=1e-12):
                raise RuntimeError(f"Selective scaler smoke scale mismatch: {key}")
        row = selected[0]
        path = CACHE_ROOT / row["site_id"] / f"{row['record_id']}.npz"
        with np.load(path, allow_pickle=False) as z:
            seq, ecg, static, mask = [np.asarray(z[key]) for key in ("X_seq", "X_ecg", "x_static", "mask")]
            stage, valid, channels = [np.asarray(z[key]) for key in ("stage_code_aligned", "stage_valid", "eeg_channel_available")]
        full_seq, _, full_static = shared.transform_arrays(seq, ecg, static, record_id=row["record_id"])
        exact = {arm: pooled.assemble_features(full_static, stage, valid, channels, full_seq, p3_arm)
                 for arm, p3_arm in ARMS.items()}
        adapted = selective_p3_features(seq, ecg, static, mask, stage, valid, channels, params, rules_by_key)
        for arm in ARMS:
            if not np.allclose(exact[arm], adapted[arm], rtol=0, atol=1e-6, equal_nan=True):
                raise RuntimeError(f"Selective P3 adapter smoke mismatch: {arm}")
    print("SMOKE_PASS selective scaling and P3 inputs match shared FeatureScaler", flush=True)


def full_scan(records, rules, rules_by_key):
    contributions = empty_contributions(rules)
    families = {name: [] for name in p6.FAMILIES}
    manifest_rows, availability = [], []
    code_shas, versions, source_hashes = set(), set(), set()
    for ordinal, row in enumerate(records, 1):
        path = CACHE_ROOT / row["site_id"] / f"{row['record_id']}.npz"
        with np.load(path, allow_pickle=False) as z:
            required = {
                "X_seq", "X_ecg", "x_static", "mask", "y", "record_id", "site_id",
                "bdsp_patient_id", "extraction_version", "code_git_sha", "source_sha256_json",
                "stage_code_aligned", "stage_valid", "eeg_channel_available",
                "eeg_clean_subsegment_count", "eeg_total_subsegment_count", "hrv_success",
            }
            if not required.issubset(z.files):
                raise RuntimeError(f"Missing NPZ metadata {sorted(required-set(z.files))}: {path}")
            seq, ecg, static, mask = [np.asarray(z[key]) for key in ("X_seq", "X_ecg", "x_static", "mask")]
            stage, stage_valid, channels = [np.asarray(z[key]) for key in (
                "stage_code_aligned", "stage_valid", "eeg_channel_available")]
            clean, total = [np.asarray(z[key]) for key in (
                "eeg_clean_subsegment_count", "eeg_total_subsegment_count")]
            hrv = np.asarray(z["hrv_success"])
            label = int(np.asarray(z["y"]).item())
            record_id, site = str(z["record_id"].item()), str(z["site_id"].item())
            patient_id = str(z["bdsp_patient_id"].item())
            version, code_sha = str(z["extraction_version"].item()), str(z["code_git_sha"].item())
            source_hash = str(z["source_sha256_json"].item())
        if record_id != row["record_id"] or site != row["site_id"] or int(row["label"]) != label:
            raise RuntimeError(f"Manifest/NPZ identity or label mismatch: {path}")
        if seq.ndim != 2 or seq.shape[1] != 483 or len(seq) == 0:
            raise RuntimeError(f"X_seq shape mismatch: {path} {seq.shape}")
        if ecg.ndim != 2 or ecg.shape[1] != 12 or static.shape != (196,) or mask.shape != (len(seq),):
            raise RuntimeError(f"ECG/static/mask shape mismatch: {path}")
        if stage.shape != (len(seq),) or stage_valid.shape != (len(seq),) or channels.shape != (6,):
            raise RuntimeError(f"timegrid metadata shape mismatch: {path}")
        if not all(np.isfinite(array).all() for array in (seq, ecg, static)):
            raise RuntimeError(f"Non-finite core data: {path}")
        if not np.isfinite(static[0]):
            raise RuntimeError(f"Missing age: {path}")
        versions.add(version); code_shas.add(code_sha); source_hashes.add(source_hash)
        add_fit_contributions(contributions, site, record_id, seq, ecg, static, mask, rules_by_key)
        feature_values = site_family_features(seq, ecg, static, stage, stage_valid, channels)
        for name, value in feature_values.items():
            families[name].append(np.asarray(value, dtype=np.float64))
        clean_total = int(total.sum())
        availability.append({
            "site": site, "record_id": record_id,
            "stage_valid_ratio": float(stage_valid.mean()),
            "unavailable_stage_ratio": float(1.0 - stage_valid.mean()),
            "eeg_channels_available": int(channels.sum()),
            "eeg_any_available": int(channels.any()), "eeg_all6_available": int(channels.all()),
            "eeg_clean_count": int(clean.sum()), "eeg_total_count": clean_total,
            "eeg_clean_ratio": float(clean.sum() / clean_total) if clean_total else math.nan,
            "hrv_success_count": int(hrv.sum()), "hrv_total_count": int(len(hrv)),
            "hrv_success_ratio": float(hrv.mean()) if len(hrv) else math.nan,
            "hrv_any_success": int(hrv.any()), "ecg_available": int(len(ecg) > 0),
            "female": float(static[1]), "male": float(static[2]), "sex_other_unknown": float(static[3]),
            "race_asian": float(static[4]), "race_black": float(static[5]),
            "race_others": float(static[6]), "race_unavailable": float(static[7]),
            "race_white": float(static[8]), "bmi": float(static[9]),
            "bmi_missing": int(not np.isfinite(static[9]) or static[9] <= 0),
        })
        manifest_rows.append({
            "record_id": record_id, "patient_id": patient_id, "site": site,
            "label": int(row["label"]), "age": float(static[0]),
            "npz_path": str(path.relative_to(ROOT)), "extraction_version": version,
            "extraction_code_sha": code_sha,
        })
        if ordinal % 250 == 0:
            print(f"SCAN1 {ordinal}/1103", flush=True)
    if versions != {"timegrid_v2.0.0"}:
        raise RuntimeError(f"Mixed extraction versions: {versions}")
    return (pd.DataFrame(manifest_rows), pd.DataFrame(availability),
            {name: np.asarray(value, dtype=np.float64) for name, value in families.items()},
            contributions, code_shas, source_hashes)


def validate_patient_isolation(manifest: pd.DataFrame):
    cross_site = manifest.groupby("patient_id").site.nunique()
    if (cross_site > 1).any():
        raise RuntimeError(f"Patients occur across sites: {cross_site[cross_site > 1].index[:10].tolist()}")
    for holdout in SITES:
        train = manifest.loc[manifest.site != holdout]
        test = manifest.loc[manifest.site == holdout]
        if set(train.patient_id) & set(test.patient_id):
            raise RuntimeError(f"Patient leakage for holdout {holdout}")


def build_p3_matrices(records, fold_parameters, rules_by_key):
    matrices = {f"p3_{holdout}_{arm}": [] for holdout in SITES for arm in ARMS}
    for ordinal, row in enumerate(records, 1):
        path = CACHE_ROOT / row["site_id"] / f"{row['record_id']}.npz"
        with np.load(path, allow_pickle=False) as z:
            seq, ecg, static, mask = [np.asarray(z[key]) for key in ("X_seq", "X_ecg", "x_static", "mask")]
            stage, valid, channels = [np.asarray(z[key]) for key in (
                "stage_code_aligned", "stage_valid", "eeg_channel_available")]
        for holdout in SITES:
            features = selective_p3_features(
                seq, ecg, static, mask, stage, valid, channels,
                fold_parameters[holdout], rules_by_key,
            )
            for arm, values in features.items():
                matrices[f"p3_{holdout}_{arm}"].append(values)
        if ordinal % 250 == 0:
            print(f"SCAN2 {ordinal}/1103", flush=True)
    return {key: np.asarray(value, dtype=np.float64) for key, value in matrices.items()}


def write_qc(manifest, availability, code_shas, source_hashes, events):
    QC_OUT.mkdir(parents=True, exist_ok=True)
    counts = []
    for site in [*SITES, "Total"]:
        frame = manifest if site == "Total" else manifest.loc[manifest.site == site]
        counts.append({
            "site": site, "n": len(frame), "patients": frame.patient_id.nunique(),
            "positives": int(frame.label.sum()), "negatives": int((frame.label == 0).sum()),
            "positive_prevalence": float(frame.label.mean()),
            "multi_session_patients": int((frame.groupby("patient_id").size() > 1).sum()),
        })
    counts_frame = pd.DataFrame(counts)
    counts_frame.to_csv(QC_OUT / "site_counts.csv", index=False)
    age_rows = []
    for site in [*SITES, "Total"]:
        site_frame = manifest if site == "Total" else manifest.loc[manifest.site == site]
        for label in [0, 1, "all"]:
            frame = site_frame if label == "all" else site_frame.loc[site_frame.label == label]
            age_rows.append({
                "site": site, "label": label, "n": len(frame),
                "age_median": frame.age.median(), "age_q25": frame.age.quantile(.25),
                "age_q75": frame.age.quantile(.75), "age_iqr": frame.age.quantile(.75)-frame.age.quantile(.25),
                "age_min": frame.age.min(), "age_max": frame.age.max(),
            })
    pd.DataFrame(age_rows).to_csv(QC_OUT / "age_summary.csv", index=False)
    pairs = []
    for site in SITES:
        frame = manifest.loc[manifest.site == site]
        pairs.append({
            "site": site, "n": len(frame), "positives": int(frame.label.sum()),
            "negatives": int((frame.label == 0).sum()),
            "eligible_ac_pairs_gap2": p5.eligible_pairs(frame.label, frame.age),
        })
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(QC_OUT / "ac_pair_counts.csv", index=False)
    availability_rows = []
    for site in [*SITES, "Total"]:
        frame = availability if site == "Total" else availability.loc[availability.site == site]
        availability_rows.append({
            "site": site, "n": len(frame),
            "stage_valid_ratio_mean": frame.stage_valid_ratio.mean(),
            "stage_valid_ratio_median": frame.stage_valid_ratio.median(),
            "unavailable_stage_ratio_mean": frame.unavailable_stage_ratio.mean(),
            "eeg_channels_available_mean": frame.eeg_channels_available.mean(),
            "eeg_any_available_rate": frame.eeg_any_available.mean(),
            "eeg_all6_available_rate": frame.eeg_all6_available.mean(),
            "eeg_clean_ratio_aggregate": frame.eeg_clean_count.sum()/frame.eeg_total_count.sum(),
            "hrv_success_window_rate": frame.hrv_success_count.sum()/frame.hrv_total_count.sum(),
            "hrv_any_success_record_rate": frame.hrv_any_success.mean(),
            "ecg_available_record_rate": frame.ecg_available.mean(),
            "female_rate": frame.female.mean(), "male_rate": frame.male.mean(),
            "sex_other_unknown_rate": frame.sex_other_unknown.mean(),
            "race_asian_rate": frame.race_asian.mean(), "race_black_rate": frame.race_black.mean(),
            "race_others_rate": frame.race_others.mean(),
            "race_unavailable_rate": frame.race_unavailable.mean(), "race_white_rate": frame.race_white.mean(),
            "bmi_missing_rate": frame.bmi_missing.mean(),
            "bmi_median_available": frame.loc[frame.bmi_missing == 0, "bmi"].median(),
        })
    availability_frame = pd.DataFrame(availability_rows)
    availability_frame.to_csv(QC_OUT / "availability_summary.csv", index=False)
    lines = [
        "# Data2 QC report", "",
        f"- Source: `{CACHE_ROOT}`.",
        "- Exact coverage: 6600 unique manifest records and 6600 one-to-one local NPZ files; 6600 success and 0 failure.",
        "- Full streaming validation passed for record/site/label, age, version, 483/12/196 shapes, metadata alignment, and finite core arrays.",
        f"- Extraction version: `timegrid_v2.0.0`; embedded extraction code SHA value(s): `{', '.join(sorted(code_shas))}`.",
        "- The remote extraction checkout did not contain Git metadata, so `code_git_sha=unknown`; exact extraction-source SHA256 values are retained in run metadata and each NPZ.",
        f"- Unique patients: {manifest.patient_id.nunique()}; multi-session patients: {(manifest.groupby('patient_id').size()>1).sum()}; maximum sessions per patient: {manifest.groupby('patient_id').size().max()}.",
        "", "## Site composition", "",
        markdown_table(counts_frame),
        "", "## Age-conditioned pair counts", "",
        markdown_table(pair_frame, decimals=0),
        "", "## Availability summary", "",
        markdown_table(availability_frame),
        "", "Age distributions split by site and label are in `age_summary.csv`. All availability statistics use metadata already stored in the NPZ; no raw EDF was read.",
    ]
    (QC_OUT / "DATA2_QC_REPORT.md").write_text("\n".join(lines) + "\n")


def run_site_audit(manifest, families):
    SITE_OUT.mkdir(parents=True, exist_ok=True)
    confusion_dir = SITE_OUT / "confusion_matrices"
    confusion_dir.mkdir(exist_ok=True)
    labels = manifest.site.map({site: index for index, site in enumerate(SITES)}).to_numpy(int)
    groups = manifest.patient_id.to_numpy(str)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=7)
    fold_rows, summaries = [], []
    for family in p6.FAMILIES:
        matrix = families[family]
        aggregate_cm = np.zeros((3, 3), dtype=int)
        messages_all = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(matrix, labels, groups)):
            train, test, _ = p6.fit_impute_scale(matrix[train_idx], matrix[test_idx])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = p6.LogisticRegression(**p6.LR_CONFIG).fit(train, labels[train_idx])
            messages = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
            messages_all.extend(messages)
            prediction, probability = model.predict(test), model.predict_proba(test)
            aggregate_cm += confusion_matrix(labels[test_idx], prediction, labels=np.arange(3))
            fold_rows.append({
                "feature_family": family, "dimension": matrix.shape[1], "fold": fold,
                "site_balanced_accuracy": balanced_accuracy_score(labels[test_idx], prediction),
                "macro_f1": f1_score(labels[test_idx], prediction, average="macro", zero_division=0),
                "macro_ovr_auroc": roc_auc_score(labels[test_idx], probability, average="macro", multi_class="ovr"),
                "train_n": len(train_idx), "holdout_n": len(test_idx),
                "train_patients": len(set(groups[train_idx])), "holdout_patients": len(set(groups[test_idx])),
                "converged": not messages, "n_iter": int(model.n_iter_.max()),
            })
        family_frame = pd.DataFrame([row for row in fold_rows if row["feature_family"] == family])
        summaries.append({
            "feature_family": family, "dimension": matrix.shape[1],
            "site_balanced_accuracy_mean": family_frame.site_balanced_accuracy.mean(),
            "std": family_frame.site_balanced_accuracy.std(ddof=1),
            "min": family_frame.site_balanced_accuracy.min(),
            "max": family_frame.site_balanced_accuracy.max(),
            "macro_f1": family_frame.macro_f1.mean(),
            "macro_ovr_auroc": family_frame.macro_ovr_auroc.mean(),
            "all_folds_converged": not messages_all,
            "convergence_warnings": " | ".join(sorted(set(messages_all))),
        })
        pd.DataFrame(aggregate_cm, index=SITES, columns=SITES).rename_axis("true_site").to_csv(
            confusion_dir / f"{family}.csv"
        )
        print("D1", json.dumps(summaries[-1]), flush=True)
    folds = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summaries).sort_values("site_balanced_accuracy_mean", ascending=False)
    folds.to_csv(SITE_OUT / "site_feature_cv_metrics.csv", index=False)
    summary.to_csv(SITE_OUT / "site_feature_summary.csv", index=False)
    old = pd.read_csv(ROOT / "output/p6_site_domain/site_audit/site_feature_summary.csv").set_index("feature_family")
    lines = [
        "# Site decodability on data2", "",
        "The classifier protocol and patient-level summaries directly reuse P6-A. All 6600 records have unique patient IDs; five-fold splitting nevertheless remains patient-group-aware while stratifying site.", "",
        "| Feature family | Dim | Data2 balanced accuracy | SD | Min-max | Macro F1 | Old1103 P6-A | Descriptive delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        old_value = float(old.loc[row.feature_family, "mean_balanced_accuracy"])
        lines.append(
            f"| {row.feature_family} | {row.dimension} | {row.site_balanced_accuracy_mean:.3f} | "
            f"{row.std:.3f} | {row.min:.3f}-{row.max:.3f} | {row.macro_f1:.3f} | "
            f"{old_value:.3f} | {row.site_balanced_accuracy_mean-old_value:+.3f} |"
        )
    lines += [
        "", "The old1103 comparison is descriptive only: extractor version, sample size, and site composition all changed. High decodability is diagnostic and does not imply automatic feature removal.",
    ]
    (SITE_OUT / "SITE_DECODABILITY_DATA2.md").write_text("\n".join(lines) + "\n")
    return summary


def write_logits(path, frame, scores):
    pd.DataFrame({
        "record_id": frame.record_id, "patient_id": frame.patient_id,
        "site": frame.site, "label": frame.label, "raw_age": frame.age,
        "decision_score": scores,
    }).to_csv(path, index=False)


def run_loso(manifest, matrices, fold_parameters, rules):
    LOSO_OUT.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(LOSO_OUT / "data2_loso_manifest.csv", index=False)
    rule_holder = type("RuleHolder", (), {"rules": rules})()
    results = []
    for arm, p3_arm in ARMS.items():
        arm_dir = LOSO_OUT / arm
        arm_dir.mkdir(exist_ok=True)
        names = p3.feature_names(rule_holder, p3_arm)
        for holdout in SITES:
            fold_dir = arm_dir / f"holdout_{holdout}"
            fold_dir.mkdir(exist_ok=True)
            matrix = matrices[f"p3_{holdout}_{arm}"]
            train_mask = manifest.site.to_numpy() != holdout
            hold_mask = ~train_mask
            train_frame, hold_frame = manifest.loc[train_mask].reset_index(drop=True), manifest.loc[hold_mask].reset_index(drop=True)
            x_train, x_hold = matrix[train_mask], matrix[hold_mask]
            y_train, y_hold = manifest.label.to_numpy(int)[train_mask], manifest.label.to_numpy(int)[hold_mask]
            age_train, age_hold = manifest.age.to_numpy(float)[train_mask], manifest.age.to_numpy(float)[hold_mask]
            med, center, scale, missing = p3.fit_imputer_scaler(x_train)
            train_scaled = p3.apply_imputer_scaler(x_train, med, center, scale)
            hold_scaled = p3.apply_imputer_scaler(x_hold, med, center, scale)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = p3.LogisticRegression(**p3.LR).fit(train_scaled, y_train)
            convergence = [str(item.message) for item in caught if issubclass(item.category, ConvergenceWarning)]
            train_score, hold_score = model.decision_function(train_scaled), model.decision_function(hold_scaled)
            train_metrics = p5.metric_bundle(y_train, train_score, age_train)
            hold_metrics = p5.metric_bundle(y_hold, hold_score, age_hold)
            write_logits(fold_dir / "train_logits.csv", train_frame, train_score)
            write_logits(fold_dir / "holdout_logits.csv", hold_frame, hold_score)
            config = {
                "dataset": DATASET, "arm": arm, "p3_definition": p3_arm,
                "feature_names": names, "feature_dimension": len(names),
                "linear_model": p3.LR,
                "input_preprocessing": "typed_v1 selected-column adapter fit on outer training sites only",
                "outer_holdout_used_for_fit_or_selection": False,
                "holdout_site": holdout,
            }
            (fold_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
            joblib.dump({
                "model": model, "imputer_median": med, "linear_mean": center,
                "linear_scale": scale, "all_missing_columns": missing,
                "selected_input_parameters": {
                    f"{key[0]}:{key[1]}": value for key, value in fold_parameters[holdout].items()
                },
                "feature_names": names, "holdout_site": holdout,
                "train_record_ids": train_frame.record_id.tolist(),
            }, fold_dir / "model.joblib")
            result = {
                "model": arm, "holdout_site": holdout,
                "train": train_metrics, "holdout": hold_metrics,
                "train_ac_minus_holdout_ac": train_metrics["ac_auroc"] - hold_metrics["ac_auroc"],
                "n_iter": int(model.n_iter_[0]), "converged": not convergence,
                "convergence_warnings": convergence,
            }
            (fold_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
            results.append(result)
            print("LOSO", json.dumps(result), flush=True)
    rows = []
    for result in results:
        rows.append({
            "model": result["model"], "holdout_site": result["holdout_site"],
            **{f"train_{key}": value for key, value in result["train"].items()},
            **{f"holdout_{key}": value for key, value in result["holdout"].items()},
            "train_ac_minus_holdout_ac": result["train_ac_minus_holdout_ac"],
            "n_iter": result["n_iter"], "converged": result["converged"],
            "convergence_warnings": " | ".join(result["convergence_warnings"]),
        })
    fold_frame = pd.DataFrame(rows)
    fold_frame.to_csv(LOSO_OUT / "fold_metrics.csv", index=False, na_rep="N/A")
    aggregate = {}
    for arm in ARMS:
        group = fold_frame.loc[fold_frame.model == arm].set_index("holdout_site").loc[list(SITES)]
        aggregate[arm] = {
            "sites": {site: {
                "ac_auroc": float(group.loc[site, "holdout_ac_auroc"]),
                "age_weighted_auroc": float(group.loc[site, "holdout_age_weighted_auroc"]),
                "auroc": float(group.loc[site, "holdout_auroc"]),
                "auprc": float(group.loc[site, "holdout_auprc"]),
                "n": int(group.loc[site, "holdout_n"]),
                "positives": int(group.loc[site, "holdout_positives"]),
                "negatives": int(group.loc[site, "holdout_negatives"]),
                "eligible_ac_pairs": int(group.loc[site, "holdout_eligible_ac_pairs"]),
            } for site in SITES},
            "macro_ac_auroc": float(group.holdout_ac_auroc.mean()),
            "worst_site_ac_auroc": float(group.holdout_ac_auroc.min()),
            "macro_age_weighted_auroc": float(group.holdout_age_weighted_auroc.mean()),
            "macro_auroc": float(group.holdout_auroc.mean()),
            "macro_auprc": float(group.holdout_auprc.mean()),
        }
    (LOSO_OUT / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    old = json.loads((ROOT / "output/p5_loso_1103/aggregate_metrics.json").read_text())
    old_map = {"demo10": old["L0_demo10"], "compact30": old["L1_compact30"], "global59": old["L2_global59"]}
    lines = [
        "# Data2 LR LOSO results", "",
        "All models use the frozen P3 definitions and LR configuration. Typed selected-column scaling and P3 linear imputation/scaling were fitted only on the two outer training sites. No threshold, calibration, model selection, or holdout tuning was performed.", "",
        "| Model | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst AC | Macro AUROC | Macro AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = aggregate[arm]
        lines.append(
            f"| {arm} | {item['sites']['I0002']['ac_auroc']:.3f} | {item['sites']['I0006']['ac_auroc']:.3f} | "
            f"{item['sites']['S0001']['ac_auroc']:.3f} | {item['macro_ac_auroc']:.3f} | "
            f"{item['worst_site_ac_auroc']:.3f} | {item['macro_auroc']:.3f} | {item['macro_auprc']:.3f} |"
        )
    lines += ["", "## Descriptive old1103 comparison", "",
              "| Model | Old I0002/I0006/S0001 | Old Macro/Worst | Data2 Macro/Worst |",
              "|---|---:|---:|---:|"]
    for arm in ARMS:
        old_item, item = old_map[arm], aggregate[arm]
        lines.append(
            f"| {arm} | {old_item['sites']['I0002']['ac_auroc']:.3f} / {old_item['sites']['I0006']['ac_auroc']:.3f} / "
            f"{old_item['sites']['S0001']['ac_auroc']:.3f} | {old_item['macro_ac_auroc']:.3f} / "
            f"{old_item['worst_site_ac_auroc']:.3f} | {item['macro_ac_auroc']:.3f} / {item['worst_site_ac_auroc']:.3f} |"
        )
    lines += ["", "Old1103 to data2 simultaneously changes extractor, sample size, and site composition. These differences are descriptive and cannot be attributed until the new1103 bridge is complete."]
    warnings_frame = fold_frame.loc[~fold_frame["converged"]]
    if not warnings_frame.empty:
        warning_items = "; ".join(
            f"{row.model}/{row.holdout_site}: {row.convergence_warnings} (n_iter={int(row.n_iter)})"
            for row in warnings_frame.itertuples(index=False)
        )
        lines += ["", f"Convergence warning(s), retained without hyperparameter changes or reruns: {warning_items}."]
    (LOSO_OUT / "DATA2_LR_LOSO_RESULTS.md").write_text("\n".join(lines) + "\n")
    return fold_frame, aggregate


def make_registry(aggregate):
    REGISTRY_OUT.mkdir(parents=True, exist_ok=True)
    old = json.loads((ROOT / "output/p5_loso_1103/aggregate_metrics.json").read_text())
    p6agg = json.loads((ROOT / "output/p6_site_domain/lstm_balance/aggregate_metrics.json").read_text())
    entries = []
    def add(entry_id, dataset, model, feature_set, preprocessing, sampler, pos_weight,
            protocol, values, result_commit, result_path, role, checkpoint_rule="none"):
        entries.append({
            "id": entry_id, "dataset": dataset, "cache": "npz_new" if dataset.startswith("OLD1103") else "npz_data2",
            "extraction_version": "legacy_unversioned" if dataset.startswith("OLD1103") else "timegrid_v2.0.0",
            "model": model, "feature_set": feature_set, "preprocessing": preprocessing,
            "sampler": sampler, "pos_weight": pos_weight, "CV_protocol": protocol,
            "checkpoint_rule": checkpoint_rule, "seed": 7,
            "I0002_AC": values.get("sites", {}).get("I0002", {}).get("ac_auroc"),
            "I0006_AC": values.get("sites", {}).get("I0006", {}).get("ac_auroc"),
            "S0001_AC": values.get("sites", {}).get("S0001", {}).get("ac_auroc"),
            "macro_AC": values.get("macro_ac_auroc"), "worst_site_AC": values.get("worst_site_ac_auroc"),
            "result_commit": result_commit, "result_path": result_path, "role": role,
        })
    add("OLD-D", "OLD1103 / old npz_new", "elastic-net LR", "demo10", "typed_v1", "class_weight=balanced", None, "three-site LOSO", old["L0_demo10"], "69e2aecef497f879b6a9e9c1953474191b8e6d91", "output/p5_loso_1103", "historical LOSO benchmark")
    add("OLD-C", "OLD1103 / old npz_new", "elastic-net LR", "compact30", "typed_v1", "class_weight=balanced", None, "three-site LOSO", old["L1_compact30"], "69e2aecef497f879b6a9e9c1953474191b8e6d91", "output/p5_loso_1103", "historical LOSO benchmark")
    add("OLD-G", "OLD1103 / old npz_new", "elastic-net LR", "global59", "typed_v1", "class_weight=balanced", None, "three-site LOSO", old["L2_global59"], "69e2aecef497f879b6a9e9c1953474191b8e6d91", "output/p5_loso_1103", "historical LOSO benchmark")
    add("OLD-L0", "OLD1103 / old npz_new", "2-layer LSTM", "483+12+196", "legacy_clip", "global class-balanced", "empirical", "three-site LOSO fixed epoch6", old["L3_legacy_lstm"], "69e2aecef497f879b6a9e9c1953474191b8e6d91", "output/p5_loso_1103", "historical LOSO benchmark", "fixed_epoch6")
    add("OLD-L1", "OLD1103 / old npz_new", "2-layer LSTM", "483+12+196", "legacy_clip", "global class-balanced", 1.0, "three-site LOSO fixed epoch6", p6agg["B1_global_class_pos1"], "9eb1b8dc93e2d5905ae544a462f2dce71b1e1fe1", "output/p6_site_domain/lstm_balance", "historical LOSO benchmark", "fixed_epoch6")
    add("OLD-L2", "OLD1103 / old npz_new", "2-layer LSTM", "483+12+196", "legacy_clip", "site+class-balanced", 1.0, "three-site LOSO fixed epoch6", p6agg["B2_site_class_pos1"], "9eb1b8dc93e2d5905ae544a462f2dce71b1e1fe1", "output/p6_site_domain/lstm_balance", "historical LOSO benchmark", "fixed_epoch6")
    mixed = {"macro_ac_auroc": .701, "worst_site_ac_auroc": None, "sites": {}}
    add("P4-MIXED", "OLD1103 / old npz_new", "2-layer LSTM", "483+12+196", "legacy_clip", "global class-balanced", "empirical", "3x3 mixed-site repeated CV", mixed, "736c2dc", "output/p4_legacy_lstm_cv", "MIXED-SITE REFERENCE ONLY", "best validation AC")
    for arm, code in (("demo10", "DATA2-D"), ("compact30", "DATA2-C"), ("global59", "DATA2-G")):
        add(code, "data2 timegrid_v2", "elastic-net LR", arm, "typed_v1 selective exact adapter", "class_weight=balanced", None, "three-site LOSO", aggregate[arm], "self (commit containing registry)", "feat_input2/output/data2_lowcost/loso_lr", "current data2 low-cost baseline")
    payload = {
        "schema_version": 1, "generated_from_source_commit": git_head(),
        "interpretation_warning": "Old1103/data2 differences are descriptive; extractor, sample size, and site composition all change.",
        "entries": entries,
    }
    (REGISTRY_OUT / "baseline_registry.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Baseline registry", "", payload["interpretation_warning"], "",
             "| ID | Dataset | Model/features | Protocol | I0002 | I0006 | S0001 | Macro | Worst | Role |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    def value(x): return "N/A" if x is None else f"{x:.3f}"
    for item in entries:
        lines.append(
            f"| {item['id']} | {item['dataset']} | {item['model']} / {item['feature_set']} | {item['CV_protocol']} | "
            f"{value(item['I0002_AC'])} | {value(item['I0006_AC'])} | {value(item['S0001_AC'])} | "
            f"{value(item['macro_AC'])} | {value(item['worst_site_AC'])} | {item['role']} |"
        )
    lines += ["", "`P4-MIXED` is a MIXED-SITE REFERENCE ONLY and must not be interpreted as the same protocol as LOSO."]
    (REGISTRY_OUT / "BASELINE_REGISTRY.md").write_text("\n".join(lines) + "\n")


def final_report(manifest, site_summary, aggregate):
    counts = pd.read_csv(QC_OUT / "site_counts.csv")
    pairs = pd.read_csv(QC_OUT / "ac_pair_counts.csv")
    lines = [
        "# Data2 low-cost baselines", "", "## 1. Data2 integrity", "",
        "The complete local timegrid_v2 cache passed one-to-one manifest coverage, labels, sites, ages, version, dimensions, metadata alignment, and full finite-value checks: 6600 success, 0 failure.", "",
        "## 2. Composition", "", markdown_table(counts), "",
        "Eligible official gap=2 age-pair counts:", "", markdown_table(pairs, decimals=0), "",
        "## 3. Site decodability", "",
        "| Family | Balanced accuracy | SD |", "|---|---:|---:|",
    ]
    for row in site_summary.itertuples(index=False):
        lines.append(f"| {row.feature_family} | {row.site_balanced_accuracy_mean:.3f} | {row.std:.3f} |")
    lines += ["", "## 4. Three-site LOSO", "",
              "| Model | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst AC |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        item = aggregate[arm]
        lines.append(
            f"| {arm} | {item['sites']['I0002']['ac_auroc']:.3f} | {item['sites']['I0006']['ac_auroc']:.3f} | "
            f"{item['sites']['S0001']['ac_auroc']:.3f} | {item['macro_ac_auroc']:.3f} | {item['worst_site_ac_auroc']:.3f} |"
        )
    old = json.loads((ROOT / "output/p5_loso_1103/aggregate_metrics.json").read_text())
    lines += ["", "## 5. Descriptive old1103 comparison", "",
              "| Model | Old1103 Macro/Worst | Data2 Macro/Worst |", "|---|---:|---:|",
              f"| demo10 | {old['L0_demo10']['macro_ac_auroc']:.3f} / {old['L0_demo10']['worst_site_ac_auroc']:.3f} | {aggregate['demo10']['macro_ac_auroc']:.3f} / {aggregate['demo10']['worst_site_ac_auroc']:.3f} |",
              f"| compact30 | {old['L1_compact30']['macro_ac_auroc']:.3f} / {old['L1_compact30']['worst_site_ac_auroc']:.3f} | {aggregate['compact30']['macro_ac_auroc']:.3f} / {aggregate['compact30']['worst_site_ac_auroc']:.3f} |",
              f"| global59 | {old['L2_global59']['macro_ac_auroc']:.3f} / {old['L2_global59']['worst_site_ac_auroc']:.3f} | {aggregate['global59']['macro_ac_auroc']:.3f} / {aggregate['global59']['worst_site_ac_auroc']:.3f} |",
              "", "## 6. Limited conclusions", "",
              f"- demo10 is a modest population-only baseline (Macro AC {aggregate['demo10']['macro_ac_auroc']:.3f}); its I0006 holdout is {aggregate['demo10']['sites']['I0006']['ac_auroc']:.3f}.",
              f"- compact30 is strongest and most stable: Macro AC {aggregate['compact30']['macro_ac_auroc']:.3f}, Worst-site AC {aggregate['compact30']['worst_site_ac_auroc']:.3f}.",
              f"- global59 does not add overall cross-site AC over compact30: Macro delta {aggregate['global59']['macro_ac_auroc']-aggregate['compact30']['macro_ac_auroc']:+.3f}; its Macro AUPRC remains a secondary diagnostic.",
              "- Site decodability is diagnostic and is not evidence for automatic feature deletion.",
              "- Data2 supplies substantially more eligible age-matched pairs than old1103, but old-to-new differences are not causally attributable.",
              "", "## 7. Questions deferred until the new1103 bridge", "",
              "The comparison cannot determine whether timegrid_v2 itself improves performance, how much gain comes from 6600 records, whether B2 should be the final data2 LSTM protocol, whether typed beats legacy, or which 483-dimensional family should be removed. Old1103 to data2 changes extractor, sample size, and site composition simultaneously.",
              "", "No LSTM experiment was launched."]
    (OUT / "DATA2_LOWCOST_BASELINES.md").write_text("\n".join(lines) + "\n")



def bridge_qc(manifest, old, code_shas, source_hashes):
    QC_OUT.mkdir(parents=True, exist_ok=True)
    matched = old.merge(manifest, on="record_id", suffixes=("_old", "_new"), validate="one_to_one")
    matched.to_csv(QC_OUT / "matched_records.csv", index=False)
    counts=[]
    for site in SITES:
        x=manifest[manifest.site==site]
        counts.append({"site":site,"n":len(x),"patients":x.patient_id.nunique(),"positives":int(x.label.sum()),"negatives":int((x.label==0).sum()),"eligible_ac_pairs_gap2":p5.eligible_pairs(x.label,x.age)})
    pd.DataFrame(counts).to_csv(QC_OUT / "site_counts.csv", index=False)
    versions={"extraction_versions":sorted(manifest.extraction_version.unique()),"extraction_code_git_sha_values":sorted(code_shas),"source_sha256_json_values":[json.loads(v) for v in sorted(source_hashes)],"recovery_summary":{"existing":902,"success":201,"failed":0}}
    (QC_OUT / "extraction_versions.json").write_text(json.dumps(versions,indent=2)+"\n")
    lines=["# NEW1103 timegrid_v2 bridge QC","","- Exact coverage: 1103 NEW NPZ files matched one-to-one to the frozen P5 manifest.","- Patient IDs, labels, sites, and raw ages match for all records.","- Full streaming validation passed for 483/12/196 dimensions, metadata alignment, and finite core arrays.","- Extraction version is uniformly `timegrid_v2.0.0`.",f"- Extraction Git SHA values: `{', '.join(sorted(code_shas))}`; exact source hashes are in `extraction_versions.json`.","- Recovery summary: 902 existing + 201 generated + 0 failed.","","## Site counts","",markdown_table(pd.DataFrame(counts),decimals=0)]
    (QC_OUT / "BRIDGE_QC_REPORT.md").write_text("\n".join(lines)+"\n")


def paired_qc(old):
    out=OUT/"paired_qc"; out.mkdir(parents=True)
    slices={"eeg_spectral54":(0,54),"coherence360":(54,414),"bsr18":(414,432),"emg24":(432,456),"respiration14":(456,470),"stage_event13":(470,483)}
    rows=[]; long=[]; static_sum=np.zeros(196); static_changed=np.zeros(196,int)
    for i,row in enumerate(old.itertuples(index=False),1):
        with np.load(ROOT/row.npz_path,allow_pickle=False) as a,np.load(CACHE_ROOT/row.site/f"{row.record_id}.npz",allow_pickle=False) as b:
            osq,nsq=np.asarray(a["X_seq"],float),np.asarray(b["X_seq"],float); oec,nec=np.asarray(a["X_ecg"],float),np.asarray(b["X_ecg"],float); ost,nst=np.asarray(a["x_static"],float),np.asarray(b["x_static"],float)
            valid=np.asarray(b["stage_valid"],bool); channels=np.asarray(b["eeg_channel_available"],bool); clean=np.asarray(b["eeg_clean_subsegment_count"]); total=np.asarray(b["eeg_total_subsegment_count"]); hrv=np.asarray(b["hrv_success"],bool)
        k=min(len(osq),len(nsq)); ke=min(len(oec),len(nec)); sd=np.abs(ost-nst); static_sum+=sd; static_changed+=sd>1e-12
        item={"record_id":row.record_id,"patient_id":row.patient_id,"site":row.site,"old_epochs":len(osq),"new_epochs":len(nsq),"epoch_difference":len(nsq)-len(osq),"seq_mae_aligned_epochs":float(np.mean(np.abs(osq[:k]-nsq[:k]))),"ecg_mae_aligned_epochs":float(np.mean(np.abs(oec[:ke]-nec[:ke]))) if ke else math.nan,"static_mae":float(sd.mean()),"static_changed_dimensions":int((sd>1e-12).sum()),"stage_valid_ratio":float(valid.mean()),"unavailable_stage_ratio":float(1-valid.mean()),"aligned_stage_length":len(valid),"eeg_channels_available":int(channels.sum()),"eeg_clean_ratio":float(clean.sum()/total.sum()) if total.sum() else math.nan,"hrv_success_ratio":float(hrv.mean()) if len(hrv) else math.nan}
        for name,(start,stop) in slices.items():
            value=float(np.mean(np.abs(osq[:k,start:stop]-nsq[:k,start:stop]))); item[name+"_mae"]=value; long.append({"record_id":row.record_id,"site":row.site,"feature_family":name,"mean_absolute_difference":value})
        rows.append(item)
        if i%250==0: print(f"PAIRED {i}/1103",flush=True)
    rf=pd.DataFrame(rows); rf.to_csv(out/"record_level_changes.csv",index=False)
    ff=pd.DataFrame(long).groupby("feature_family").mean_absolute_difference.agg(["mean","median","std","min","max"]).reset_index(); ff.to_csv(out/"feature_family_changes.csv",index=False)
    sf=pd.DataFrame({"feature_index":range(196),"mean_absolute_difference":static_sum/len(old),"changed_records":static_changed}); sf.to_csv(out/"static_dimension_changes.csv",index=False)
    site=rf.groupby("site").agg(records=("record_id","size"),epoch_changed_records=("epoch_difference",lambda x:int((x!=0).sum())),epoch_difference_median=("epoch_difference","median"),epoch_difference_q1=("epoch_difference",lambda x:x.quantile(.25)),epoch_difference_q3=("epoch_difference",lambda x:x.quantile(.75)),seq_mae_median=("seq_mae_aligned_epochs","median"),static_mae_median=("static_mae","median")).reset_index(); site.to_csv(out/"site_level_changes.csv",index=False)
    top=sf.sort_values("mean_absolute_difference",ascending=False).head(15); unchanged=int((sf.changed_records==0).sum())
    lines=["# OLD vs NEW1103 input comparison","","Overlapping epochs are compared by index; this describes change and does not prove correctness.","",f"- Epoch count changed: {int((rf.epoch_difference!=0).sum())}/1103 records.",f"- Epoch difference median {rf.epoch_difference.median():.1f}, range {rf.epoch_difference.min()} to {rf.epoch_difference.max()}.",f"- Median aligned X_seq MAE: {rf.seq_mae_aligned_epochs.median():.6g}.",f"- Any static change: {int((rf.static_changed_dimensions>0).sum())}/1103 records; unchanged static dimensions: {unchanged}/196.","","## Site summary","",markdown_table(site),"","## Temporal family changes","",markdown_table(ff.sort_values("mean",ascending=False)),"","## Largest static dimension changes","",markdown_table(top)]
    (out/"OLD_NEW_INPUT_COMPARISON.md").write_text("\n".join(lines)+"\n")


def bridge_main():
    parser=argparse.ArgumentParser(); parser.add_argument("--smoke-only",action="store_true"); args=parser.parse_args()
    if (OUT/"lowcost_metadata.json").exists(): raise FileExistsError(f"Refusing to overwrite completed output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    old=pd.read_csv(ROOT/"output/p5_loso_1103/loso_manifest.csv",dtype={"record_id":str,"patient_id":str,"site":str})
    if len(old)!=1103 or old.record_id.nunique()!=1103 or old.patient_id.nunique()!=1103: raise RuntimeError("Invalid frozen P5 manifest")
    actual={p.stem for p in CACHE_ROOT.glob("*/*.npz")}
    if actual!=set(old.record_id): raise RuntimeError(f"OLD/NEW record mismatch missing={len(set(old.record_id)-actual)} extra={len(actual-set(old.record_id))}")
    records=[{"record_id":r.record_id,"site_id":r.site,"label":int(r.label)} for r in old.itertuples(index=False)]
    rules=fs.load_feature_rules(RULES_PATH); rules_by_key={(r["branch"],int(r["index"])):r for r in rules}
    smoke_test(records,rules,rules_by_key)
    if args.smoke_only: return
    manifest,availability,families,contributions,code_shas,source_hashes=full_scan(records,rules,rules_by_key)
    check=old.merge(manifest,on="record_id",suffixes=("_old","_new"),validate="one_to_one")
    if not (check.patient_id_old==check.patient_id_new).all() or not (check.site_old==check.site_new).all() or not (check.label_old==check.label_new).all() or not np.allclose(check.age_old,check.age_new): raise RuntimeError("OLD/NEW identity or truth mismatch")
    manifest=manifest.sort_values(["site","record_id"]).reset_index(drop=True); validate_patient_isolation(manifest)
    bridge_qc(manifest,old,code_shas,source_hashes); paired_qc(old)
    params=build_fold_parameters(contributions,rules_by_key); matrices=build_p3_matrices(records,params,rules_by_key)
    run_site_audit(manifest,families)
    source=SITE_OUT/"SITE_DECODABILITY_DATA2.md"; target=SITE_OUT/"SITE_DECODABILITY_NEW1103.md"; value=source.read_text().replace("on data2","on NEW1103 timegrid_v2").replace("All 6600 records","All 1103 records").replace("Data2 balanced accuracy","NEW1103 balanced accuracy").replace("extractor version, sample size, and site composition all changed","patients, labels, sites, and protocol are matched; extractor version changed"); target.write_text(value); source.unlink()
    _,aggregate=run_loso(manifest,matrices,params,rules)
    (LOSO_OUT/"data2_loso_manifest.csv").rename(LOSO_OUT/"new1103_loso_manifest.csv")
    source=LOSO_OUT/"DATA2_LR_LOSO_RESULTS.md"; target=LOSO_OUT/"NEW1103_LR_LOSO_RESULTS.md"; value=source.read_text().replace("Data2","NEW1103 timegrid_v2").replace("data2","NEW1103"); value=value.replace("Old1103 to NEW1103 simultaneously changes extractor, sample size, and site composition. These differences are descriptive and cannot be attributed until the new1103 bridge is complete.", "OLD1103 and NEW1103 use the same patients, labels, sites, folds, feature definitions, and LR protocol. The principal systematic change is extraction version; the observed differences remain descriptive rather than strict causal proof."); target.write_text(value); source.unlink()
    (OUT/"lowcost_metadata.json").write_text(json.dumps({"source_commit":git_head(),"branch":subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),"records":1103,"old_manifest_sha256":sha256(ROOT/"output/p5_loso_1103/loso_manifest.csv"),"extraction_code_sha_values":sorted(code_shas),"global59_status":"executed from embedded aligned stage metadata"},indent=2)+"\n")
    print("BRIDGE_LOWCOST_COMPLETE",json.dumps(aggregate),flush=True)
def data2_main_unused():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    records, events = validate_source_manifest()
    rules = fs.load_feature_rules(RULES_PATH)
    rules_by_key = {(rule["branch"], int(rule["index"])): rule for rule in rules}
    smoke_test(records, rules, rules_by_key)
    if args.smoke_only:
        return
    started = time.time()
    manifest, availability, families, contributions, code_shas, source_hashes = full_scan(
        records, rules, rules_by_key
    )
    validate_patient_isolation(manifest)
    manifest.to_csv(OUT / "data2_manifest.csv", index=False)
    write_qc(manifest, availability, code_shas, source_hashes, events)
    fold_parameters = build_fold_parameters(contributions, rules_by_key)
    matrices = build_p3_matrices(records, fold_parameters, rules_by_key)
    SUMMARY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SUMMARY_CACHE,
        record_id=manifest.record_id.to_numpy(str), patient_id=manifest.patient_id.to_numpy(str),
        site=manifest.site.to_numpy(str), label=manifest.label.to_numpy(np.int8), age=manifest.age.to_numpy(float),
        **{f"site_{key}": value for key, value in families.items()}, **matrices,
    )
    site_summary = run_site_audit(manifest, families)
    _, aggregate = run_loso(manifest, matrices, fold_parameters, rules)
    make_registry(aggregate)
    final_report(manifest, site_summary, aggregate)
    launch = next(item for item in events if item.get("event") == "launch")
    metadata = {
        "source_commit": git_head(), "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "dataset": DATASET, "npz_root": str(CACHE_ROOT), "records": 6600,
        "source_manifest_sha256": sha256(SOURCE_MANIFEST), "source_summary_sha256": sha256(SOURCE_SUMMARY),
        "feature_rules_sha256": sha256(RULES_PATH), "extraction_code_sha_values": sorted(code_shas),
        "extraction_source_sha256_json_values": [json.loads(value) for value in sorted(source_hashes)],
        "extraction_launch": launch, "global59_status": "executed from embedded aligned stage metadata",
        "summary_cache": str(SUMMARY_CACHE), "summary_cache_sha256": sha256(SUMMARY_CACHE),
        "summary_cache_committed": False, "elapsed_seconds": time.time() - started,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "platform": platform.platform()},
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print("DATA2_LOWCOST_COMPLETE", flush=True)


if __name__ == "__main__":
    bridge_main()
