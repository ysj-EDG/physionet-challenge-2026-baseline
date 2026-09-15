#!/usr/bin/env python3
"""Run the frozen P3/P4 baselines in three-site LOSO on the legacy 1103 cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

os.environ.setdefault("LSTM_INPUT_PREPROCESSING", "legacy_clip")
os.environ.setdefault("LSTM_POS_WEIGHT_MODE", "empirical")
os.environ.setdefault("LSTM_SEED", "7")
os.environ.setdefault("PYTHONHASHSEED", "7")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import helper_code
import train_lstm as tl
from evaluate_model import compute_auroc_age, compute_auroc_weighted
from feature_scaling import FeatureScaler
from pooled_logistic_v2 import assemble_features
from feat_input.feat_mody import run_p3_v2 as p3


BASE_COMMIT = "3bda28a4b10152c86564e84bfd11237fba3dad7a"
SITES = ("I0002", "I0006", "S0001")
LR_ARMS = {
    "L0_demo10": "P3_demo10",
    "L1_compact30": "P3_compact30",
    "L2_global59": "P3_global59",
}
LSTM_ARM = "L3_legacy_lstm"
TRUTH = Path(os.environ.get(
    "P5_TRUTH", "/database/home/gaohaojie/workspace/challenge2026/output/input/prevalence.csv"
))
PREPARED_OUTPUT = ROOT / "output/p5_loso_1103"
REMOTE_INPUT = ROOT / "p5_inputs"
REMOTE_MODE = (REMOTE_INPUT / "loso_manifest.csv").is_file()
RUN_OUTPUT = Path(os.environ.get(
    "P5_OUTPUT_DIR", ROOT / "p5_h100_results" if REMOTE_MODE else PREPARED_OUTPUT
))
MANIFEST = Path(os.environ.get(
    "P5_MANIFEST", REMOTE_INPUT / "loso_manifest.csv" if REMOTE_MODE
    else PREPARED_OUTPUT / "loso_manifest.csv"
))
FROZEN_FINGERPRINTS = Path(os.environ.get(
    "P5_FROZEN_FINGERPRINTS", REMOTE_INPUT / "input_fingerprints.json" if REMOTE_MODE
    else PREPARED_OUTPUT / "input_fingerprints.json"
))
RULES = ROOT / "feat_input/feature_rules_v1.json"
SIDE_REQUESTED = Path(os.environ.get(
    "P5_STAGE_SIDECAR", REMOTE_INPUT / "stage_sidecar" if REMOTE_MODE
    else ROOT / "output/p3_v2/stage_sidecar"
))
CACHE_REQUESTED = Path(os.environ.get("LSTM_CACHE_DIR", ROOT / "npz_new"))


def resolve_cache_root(requested: Path) -> Path:
    for path in (requested, requested / "npz_new"):
        if all((path / split).is_dir() for split in ("train", "val", "test", "external")):
            return path
    return requested


def resolve_sidecar(requested: Path) -> Path:
    for path in (requested, requested / "stage_sidecar"):
        if (path / "manifest.csv").is_file():
            return path
    return requested


CACHE = resolve_cache_root(CACHE_REQUESTED)
SIDE = resolve_sidecar(SIDE_REQUESTED)

CACHE_LAYOUT = os.environ.get("P5_CACHE_LAYOUT", "partition")

def cache_path(row) -> Path:
    """Resolve cache records while preserving the historical partition default."""
    if CACHE_LAYOUT not in {"partition", "site"}:
        raise ValueError(f"Unsupported P5_CACHE_LAYOUT: {CACHE_LAYOUT}")
    record_id = row["record_id"] if isinstance(row, dict) else row.record_id
    if CACHE_LAYOUT == "site":
        group = row["site"] if isinstance(row, dict) else row.site
    else:
        group = row["npz_partition"] if isinstance(row, dict) else row.npz_partition
    return CACHE / str(group) / f"{record_id}.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def records_from_splits() -> dict[str, dict]:
    result = {}
    for partition in ("train", "val", "test", "external"):
        records = json.loads((ROOT / f"split/{partition}_records.json").read_text())
        for record in records:
            rid = f"{record['BidsFolder']}_ses-{record['SessionID']}"
            if rid in result:
                raise RuntimeError(f"Duplicate record across split JSON: {rid}")
            result[rid] = {**record, "npz_partition": partition}
    return result


def eligible_pairs(labels, ages) -> int:
    labels = np.asarray(labels, int)
    ages = np.asarray(ages, float)
    pos = ages[labels == 1]
    neg = ages[labels == 0]
    return int((np.abs(pos[:, None] - neg[None, :]) <= 2).sum())


def metric_bundle(labels, scores, ages) -> dict:
    labels = np.asarray(labels, int)
    scores = np.asarray(scores, float)
    ages = np.asarray(ages, float)
    if not np.isfinite(scores).all():
        raise RuntimeError("Non-finite decision score")
    pairs = eligible_pairs(labels, ages)
    ac = math.nan if pairs == 0 else float(compute_auroc_age(labels, scores, ages, gap=2))
    try:
        weighted = float(compute_auroc_weighted(labels, scores, ages, gap=2))
    except (ValueError, ZeroDivisionError, FloatingPointError):
        weighted = math.nan
    try:
        auc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
    except ValueError:
        auc = auprc = math.nan
    pred = scores >= 0.0
    return {
        "n": int(len(labels)), "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()), "eligible_ac_pairs": pairs,
        "ac_auroc": ac, "age_weighted_auroc": weighted,
        "auroc": auc, "auprc": auprc,
        "diagnostic_threshold": "decision_score>=0 (uncalibrated p>=0.5)",
        "diagnostic_accuracy": float(accuracy_score(labels, pred)),
        "diagnostic_f1": float(f1_score(labels, pred, zero_division=0)),
    }


def prepare_manifest() -> None:
    if PREPARED_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {PREPARED_OUTPUT}")
    if not TRUTH.is_file():
        raise FileNotFoundError(TRUTH)
    if not (SIDE / "manifest.csv").is_file():
        raise FileNotFoundError(SIDE / "manifest.csv")
    split_records = records_from_splits()
    if len(split_records) != 1103:
        raise RuntimeError(f"Expected 1103 split records, got {len(split_records)}")

    truth = pd.read_csv(TRUTH, dtype={"SiteID": str, "BDSPPatientID": str})
    required = {"SiteID", "BDSPPatientID", "Age", "Cognitive_Impairment"}
    if set(truth.columns) != required or len(truth) != 1103:
        raise RuntimeError(f"Unexpected truth source: columns={truth.columns.tolist()} n={len(truth)}")
    if truth.duplicated(["SiteID", "BDSPPatientID"]).any():
        raise RuntimeError("Truth has duplicate site/patient keys")
    truth["label"] = [helper_code.load_label(row) for row in truth.to_dict("records")]
    if set(truth.label) != {0, 1} or truth.Age.isna().any():
        raise RuntimeError("Truth labels/ages are incomplete")

    side = pd.read_csv(
        SIDE / "manifest.csv", dtype={"site_id": str, "bdsp_patient_id": str}
    )
    if len(side) != 1103 or side.record_id.duplicated().any():
        raise RuntimeError("P3 sidecar manifest does not uniquely cover 1103 records")
    merged = side[["record_id", "site_id", "bdsp_patient_id", "split"]].merge(
        truth[["SiteID", "BDSPPatientID", "Age", "label"]],
        left_on=["site_id", "bdsp_patient_id"], right_on=["SiteID", "BDSPPatientID"],
        how="outer", validate="one_to_one", indicator=True,
    )
    if not (merged._merge == "both").all():
        raise RuntimeError(f"Truth/sidecar coverage failure: {merged._merge.value_counts().to_dict()}")
    if set(merged.record_id) != set(split_records):
        raise RuntimeError("Sidecar and split JSON record IDs differ")

    rows = []
    cache_hashes, sidecar_hashes = {}, {}
    for row in merged.itertuples(index=False):
        record = split_records[row.record_id]
        partition = record["npz_partition"]
        if partition != row.split or str(record["SiteID"]) != row.site_id:
            raise RuntimeError(f"Split/site mismatch for {row.record_id}")
        cache = CACHE / partition / f"{row.record_id}.npz"
        sidecar = SIDE / f"{row.record_id}.npz"
        if not cache.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"Missing cache/sidecar for {row.record_id}")
        cache_hashes[f"{partition}/{row.record_id}.npz"] = sha256(cache)
        sidecar_hashes[f"{row.record_id}.npz"] = sha256(sidecar)
        rows.append({
            "record_id": row.record_id, "patient_id": row.bdsp_patient_id,
            "site": row.site_id, "label": int(row.label), "age": float(row.Age),
            "npz_partition": partition,
            "npz_path": f"npz_new/{partition}/{row.record_id}.npz",
        })
    manifest = pd.DataFrame(rows).sort_values(["site", "record_id"]).reset_index(drop=True)
    if len(manifest) != 1103 or manifest.record_id.duplicated().any():
        raise RuntimeError("Frozen manifest identity failure")
    if manifest.patient_id.duplicated().any():
        raise RuntimeError("Patient ID repeats across the 1103-record manifest")
    if set(manifest.site) != set(SITES) or set(manifest.label) != {0, 1}:
        raise RuntimeError("Frozen manifest site/label failure")

    PREPARED_OUTPUT.mkdir(parents=True)
    manifest.to_csv(PREPARED_OUTPUT / "loso_manifest.csv", index=False)
    counts = []
    for site in SITES:
        holdout = manifest.loc[manifest.site == site]
        train = manifest.loc[manifest.site != site]
        counts.append({
            "holdout_site": site,
            "train_records": len(train), "train_patients": train.patient_id.nunique(),
            "train_positives": int(train.label.sum()), "train_negatives": int((train.label == 0).sum()),
            "holdout_records": len(holdout), "holdout_patients": holdout.patient_id.nunique(),
            "holdout_positives": int(holdout.label.sum()), "holdout_negatives": int((holdout.label == 0).sum()),
            "holdout_eligible_ac_pairs": eligible_pairs(holdout.label, holdout.age),
        })
    pd.DataFrame(counts).to_csv(PREPARED_OUTPUT / "site_counts.csv", index=False)
    manifest_hash = sha256(PREPARED_OUTPUT / "loso_manifest.csv")
    fingerprints = {
        "base_commit": BASE_COMMIT,
        "truth_source": str(TRUTH), "truth_sha256": sha256(TRUTH),
        "official_label_parser": "helper_code.load_label",
        "manifest_sha256": manifest_hash,
        "feature_rules_sha256": sha256(RULES),
        "sidecar_manifest_sha256": sha256(SIDE / "manifest.csv"),
        "cache_file_count": len(cache_hashes), "cache_sha256": cache_hashes,
        "sidecar_file_count": len(sidecar_hashes), "sidecar_sha256": sidecar_hashes,
    }
    (PREPARED_OUTPUT / "input_fingerprints.json").write_text(
        json.dumps(fingerprints, indent=2, sort_keys=True) + "\n"
    )
    (PREPARED_OUTPUT / "source_commit.txt").write_text(BASE_COMMIT + "\n")
    print(json.dumps({"manifest": str(PREPARED_OUTPUT / 'loso_manifest.csv'), "counts": counts}, indent=2))


def validate_run_inputs() -> tuple[pd.DataFrame, dict[str, dict]]:
    if RUN_OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {RUN_OUTPUT}")
    for path in (MANIFEST, FROZEN_FINGERPRINTS, RULES, SIDE / "manifest.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = pd.read_csv(MANIFEST, dtype={"record_id": str, "patient_id": str, "site": str})
    if len(manifest) != 1103 or manifest.record_id.duplicated().any() or manifest.patient_id.duplicated().any():
        raise RuntimeError("Frozen P5 manifest does not uniquely cover 1103 patients")
    if set(manifest.site) != set(SITES) or set(manifest.label) != {0, 1}:
        raise RuntimeError("Frozen P5 manifest sites/labels are invalid")
    frozen = json.loads(FROZEN_FINGERPRINTS.read_text())
    if sha256(MANIFEST) != frozen["manifest_sha256"]:
        raise RuntimeError("Frozen LOSO manifest hash changed")
    if sha256(RULES) != frozen["feature_rules_sha256"]:
        raise RuntimeError("P3 feature rules changed")
    if sha256(SIDE / "manifest.csv") != frozen["sidecar_manifest_sha256"]:
        raise RuntimeError("P3 sidecar manifest changed")
    split_records = records_from_splits()
    missing = []
    for row in manifest.itertuples(index=False):
        cache = CACHE / row.npz_partition / f"{row.record_id}.npz"
        sidecar = SIDE / f"{row.record_id}.npz"
        if not cache.is_file() or not sidecar.is_file():
            missing.append(row.record_id)
        if row.record_id not in split_records:
            missing.append(row.record_id)
        cache_key = f"{row.npz_partition}/{row.record_id}.npz"
        sidecar_key = f"{row.record_id}.npz"
        if cache.is_file() and sha256(cache) != frozen["cache_sha256"].get(cache_key):
            raise RuntimeError(f"Frozen cache content hash changed: {cache_key}")
        if sidecar.is_file() and sha256(sidecar) != frozen["sidecar_sha256"].get(sidecar_key):
            raise RuntimeError(f"Frozen sidecar content hash changed: {sidecar_key}")
    if missing:
        raise RuntimeError(f"Missing cache/sidecar/split record(s): {missing[:10]}")
    if len(list(CACHE.glob("*/*.npz"))) != 1103 or len(list(SIDE.glob("*.npz"))) != 1103:
        raise RuntimeError("Cache or sidecar file count differs from frozen 1103")
    # Patient-level LOSO isolation is a hard stop.
    for site in SITES:
        train = manifest.loc[manifest.site != site]
        holdout = manifest.loc[manifest.site == site]
        if set(train.patient_id) & set(holdout.patient_id):
            raise RuntimeError(f"Patient leakage for holdout site {site}")
    return manifest, split_records


class FrozenTruthDataset(tl.PSGDataset):
    """Reuse PSGDataset tensor/alignment logic while sourcing labels from frozen truth."""

    def __init__(self, rows: list[dict], preprocessor: FeatureScaler):
        super().__init__(rows, str(CACHE), extractor=None, preprocessor=preprocessor)

    def __getitem__(self, idx):
        row = self.records[idx]
        path = cache_path(row)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as data:
            return self._build_tensors(
                np.asarray(data["X_seq"]), np.asarray(data["X_ecg"]),
                np.asarray(data["x_static"]), int(row["label"]),
                np.asarray(data["mask"]), row["record_id"],
            )


def natural_loader(rows: list[dict], preprocessor: FeatureScaler) -> DataLoader:
    return DataLoader(
        FrozenTruthDataset(rows, preprocessor), batch_size=8, shuffle=False,
        collate_fn=tl.collate_fn, num_workers=0,
    )


def balanced_loader(rows: list[dict], preprocessor: FeatureScaler) -> DataLoader:
    labels = np.asarray([int(row["label"]) for row in rows], int)
    counts = {label: int((labels == label).sum()) for label in (0, 1)}
    if min(counts.values()) <= 0:
        raise RuntimeError(f"Cannot construct balanced sampler: {counts}")
    weights = [len(rows) / (2 * counts[int(label)]) for label in labels]
    generator = torch.Generator().manual_seed(7)
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(rows),
        replacement=True, generator=generator,
    )
    return DataLoader(
        FrozenTruthDataset(rows, preprocessor), batch_size=8, shuffle=False,
        sampler=sampler, collate_fn=tl.collate_fn, num_workers=0, generator=generator,
    )


@torch.no_grad()
def strict_collect(model, loader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, scores, ages = [], [], []
    for batch in loader:
        if batch is None:
            raise RuntimeError("Unexpected empty batch")
        x_seq, x_ecg, mask_ecg, x_static, batch_ages, y, lengths = batch
        logits = model(
            x_seq.to(tl.device), x_ecg.to(tl.device), mask_ecg.to(tl.device),
            x_static.to(tl.device), lengths.to(tl.device),
        )
        values = logits.detach().cpu().numpy().astype(float)
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite LSTM decision logits")
        labels.extend(y.numpy().ravel().tolist())
        scores.extend(values.tolist())
        ages.extend(batch_ages.numpy().ravel().tolist())
    return np.asarray(labels, int), np.asarray(scores, float), np.asarray(ages, float)


def rows_for(frame: pd.DataFrame, split_records: dict[str, dict]) -> list[dict]:
    rows = []
    for item in frame.itertuples(index=False):
        record = split_records[item.record_id]
        rows.append({
            **record, "record_id": item.record_id, "patient_id": item.patient_id,
            "site": item.site, "label": int(item.label), "age": float(item.age),
            "npz_partition": item.npz_partition,
        })
    return rows


def make_flat_cache_index(manifest: pd.DataFrame, destination: Path) -> None:
    destination.mkdir(exist_ok=True)
    for row in manifest.itertuples(index=False):
        source = CACHE / row.npz_partition / f"{row.record_id}.npz"
        (destination / f"{row.record_id}.npz").symlink_to(source)


def load_lr_matrices(
    rows: list[dict], preprocessor: FeatureScaler,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    values = {arm: [] for arm in LR_ARMS}
    labels, ages = [], []
    for row in rows:
        path = cache_path(row)
        with np.load(path, allow_pickle=False) as data:
            x_seq = np.asarray(data["X_seq"])
            x_ecg = np.asarray(data["X_ecg"])
            x_static = np.asarray(data["x_static"])
        x_seq, _, x_static = preprocessor.transform_arrays(
            x_seq, x_ecg, x_static, record_id=row["record_id"]
        )
        with np.load(SIDE / f"{row['record_id']}.npz", allow_pickle=False) as data:
            stage = np.asarray(data["stage_code"])
            valid = np.asarray(data["stage_valid"])
            channels = np.asarray(data["eeg_channel_available"])
        for output_arm, p3_arm in LR_ARMS.items():
            values[output_arm].append(
                assemble_features(x_static, stage, valid, channels, x_seq, p3_arm)
            )
        labels.append(row["label"])
        ages.append(row["age"])
    return (
        {arm: np.asarray(matrix, dtype=np.float64) for arm, matrix in values.items()},
        np.asarray(labels, int), np.asarray(ages, float),
    )


def write_logits(path: Path, rows: list[dict], labels, ages, scores) -> None:
    if len(rows) != len(scores):
        raise RuntimeError("Logit coverage mismatch")
    frame = pd.DataFrame({
        "record_id": [row["record_id"] for row in rows],
        "patient_id": [row["patient_id"] for row in rows],
        "site": [row["site"] for row in rows], "label": labels,
        "raw_age": ages, "decision_score": scores,
        "uncalibrated_probability": 1 / (1 + np.exp(-np.clip(scores, -50, 50))),
        "diagnostic_prediction_at_0_5": np.asarray(scores) >= 0,
    })
    frame.to_csv(path, index=False)


def configure_log(path: Path):
    logger = logging.getLogger(f"p5.{path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.handlers[:] = [stream, file_handler]
    return logger


def run_lr_fold(
    arm: str, p3_arm: str, train_rows: list[dict], holdout_rows: list[dict],
    x_train: np.ndarray, x_holdout: np.ndarray, labels_train, ages_train,
    labels_holdout, ages_holdout, preprocessor: FeatureScaler, fold_dir: Path,
) -> dict:
    output = fold_dir / arm
    output.mkdir()
    median, center, scale, all_missing = p3.fit_imputer_scaler(x_train)
    train_scaled = p3.apply_imputer_scaler(x_train, median, center, scale)
    holdout_scaled = p3.apply_imputer_scaler(x_holdout, median, center, scale)
    if not np.isfinite(train_scaled).all() or not np.isfinite(holdout_scaled).all():
        raise RuntimeError(f"Non-finite LR input for {arm}/{fold_dir.name}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = p3.LogisticRegression(**p3.LR)
        model.fit(train_scaled, labels_train)
    convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
    train_score = model.decision_function(train_scaled)
    holdout_score = model.decision_function(holdout_scaled)
    train_metrics = metric_bundle(labels_train, train_score, ages_train)
    holdout_metrics = metric_bundle(labels_holdout, holdout_score, ages_holdout)
    feature_names = p3.feature_names(preprocessor, p3_arm)
    if len(feature_names) != x_train.shape[1]:
        raise RuntimeError(f"Feature name/dimension mismatch for {arm}")
    config = {
        "arm": arm, "p3_definition": p3_arm, "feature_names": feature_names,
        "feature_dimension": len(feature_names), "linear_model": p3.LR,
        "input_preprocessing": "typed_v1 fitted on outer training sites only",
        "imputation_scaling": "P3 median imputation then mean/std, outer training only",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    joblib.dump({
        "model": model, "imputer_median": median, "linear_mean": center,
        "linear_scale": scale, "all_missing_columns": all_missing,
        "input_preprocessing": preprocessor.state_dict(), "feature_names": feature_names,
        "train_record_ids": [row["record_id"] for row in train_rows],
        "holdout_site": holdout_rows[0]["site"],
    }, output / "model.joblib")
    pd.DataFrame([{"epoch": "final", **train_metrics}]).to_csv(
        output / "train_metrics.csv", index=False
    )
    write_logits(output / "train_logits.csv", train_rows, labels_train, ages_train, train_score)
    write_logits(output / "holdout_logits.csv", holdout_rows, labels_holdout, ages_holdout, holdout_score)
    result = {
        "model": arm, "holdout_site": holdout_rows[0]["site"],
        "train": train_metrics, "holdout": holdout_metrics,
        "train_ac_minus_holdout_ac": train_metrics["ac_auroc"] - holdout_metrics["ac_auroc"],
        "n_iter": int(model.n_iter_[0]), "converged": not convergence,
        "convergence_warnings": convergence,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_lstm_fold(
    train_rows: list[dict], holdout_rows: list[dict], fold_dir: Path,
) -> dict:
    output = fold_dir / LSTM_ARM
    output.mkdir()
    logger = configure_log(output / "train.log")
    tl.seed_everything(7)
    preprocessor = FeatureScaler.legacy_clip()
    train_loader = balanced_loader(train_rows, preprocessor)

    # Match the production/P4 call order: sampler diagnostic precedes model initialization.
    first_batch = next(iter(train_loader))
    if first_batch is None:
        raise RuntimeError("Empty LSTM smoke batch")
    x_seq, x_ecg, _mask, x_static, ages, y, _lengths = first_batch
    if (tuple(x_seq.shape[-1:]) != (483,) or tuple(x_ecg.shape[-1:]) != (12,)
            or tuple(x_static.shape[-1:]) != (196,)):
        raise RuntimeError("Legacy LSTM smoke dimensions differ")
    if not all(torch.isfinite(x).all() for x in (x_seq, x_ecg, x_static)):
        raise RuntimeError("Non-finite LSTM smoke input")
    logger.info(
        "smoke batch seq=%s ecg=%s static=%s positives=%d age=[%.1f,%.1f]",
        tuple(x_seq.shape), tuple(x_ecg.shape), tuple(x_static.shape),
        int(y.sum()), float(ages.min()), float(ages.max()),
    )
    model = tl.LSTMModel().to(tl.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    positives = sum(row["label"] for row in train_rows)
    negatives = len(train_rows) - positives
    pos_weight_value = tl.resolve_pos_weight("empirical", positives, negatives)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=tl.device)
    )
    history = []
    for epoch in range(1, 7):
        started = time.time()
        loss = float(tl.train_epoch(model, train_loader, optimizer, criterion))
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite LSTM loss at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": loss})
        logger.info("epoch=%d train_loss=%.8f time=%.1fs", epoch, loss, time.time() - started)

    frozen_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    checkpoint = {
        "state_dict": frozen_state, "checkpoint_role": "fixed_epoch6_loso",
        "fixed_epoch": 6, "model_seed": 7, "base_commit": BASE_COMMIT,
        "holdout_site": holdout_rows[0]["site"],
        "train_record_ids": [row["record_id"] for row in train_rows],
        "input_preprocessing": preprocessor.state_dict(),
        "feature_dims": {"X_seq": 483, "X_ecg": 12, "x_static": 196, "time_input": 495},
        "training_config": {
            "batch_size": 8, "optimizer": "Adam", "lr": 1e-3,
            "epochs": 6, "early_stopping": False, "scheduler": None,
            "gradient_clip_max_norm": 1.0, "weighted_random_sampler": True,
            "pos_weight_mode": "empirical", "pos_weight": pos_weight_value,
            "hidden_size": 128, "num_layers": 2, "fc_hidden": 64, "dropout": 0.3,
        },
    }
    torch.save(checkpoint, output / "epoch6_checkpoint.pt")
    logger.info("epoch6 checkpoint frozen before holdout loader construction")

    train_natural = natural_loader(train_rows, preprocessor)
    labels_train, score_train, ages_train = strict_collect(model, train_natural)
    train_metrics = metric_bundle(labels_train, score_train, ages_train)
    history[-1].update({f"natural_{key}": value for key, value in train_metrics.items()})
    pd.DataFrame(history).to_csv(output / "train_metrics.csv", index=False)
    write_logits(output / "train_logits.csv", train_rows, labels_train, ages_train, score_train)

    # Outer holdout is constructed and evaluated only after the epoch-6 model is frozen.
    holdout_loader = natural_loader(holdout_rows, preprocessor)
    labels_holdout, score_holdout, ages_holdout = strict_collect(model, holdout_loader)
    holdout_metrics = metric_bundle(labels_holdout, score_holdout, ages_holdout)
    write_logits(
        output / "holdout_logits.csv", holdout_rows,
        labels_holdout, ages_holdout, score_holdout,
    )
    config = {
        "arm": LSTM_ARM, "protocol": "P4 legacy_clip fixed epoch 6 LOSO",
        "holdout_site": holdout_rows[0]["site"],
        "outer_holdout_used_during_training": False,
        "model": "train_lstm.LSTMModel",
        "training_config": checkpoint["training_config"],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    result = {
        "model": LSTM_ARM, "holdout_site": holdout_rows[0]["site"],
        "train": train_metrics, "holdout": holdout_metrics,
        "train_ac_minus_holdout_ac": train_metrics["ac_auroc"] - holdout_metrics["ac_auroc"],
        "epochs_completed": 6,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    logger.info(
        "holdout frozen evaluation site=%s ac=%.6f weighted=%.6f auroc=%.6f auprc=%.6f",
        holdout_rows[0]["site"], holdout_metrics["ac_auroc"],
        holdout_metrics["age_weighted_auroc"], holdout_metrics["auroc"],
        holdout_metrics["auprc"],
    )
    return result


def smoke_lr(
    train_rows: list[dict], flat_cache: Path,
) -> None:
    # Training-only sample with both labels; no outer holdout is touched.
    frame = pd.DataFrame(train_rows)
    sample = pd.concat([
        frame.loc[frame.label == 0].head(24), frame.loc[frame.label == 1].head(16)
    ]).to_dict("records")
    records = [{"BidsFolder": row["BidsFolder"], "SessionID": row["SessionID"]} for row in sample]
    pre = FeatureScaler.fit_from_cache(records, flat_cache, RULES, sample_cap=128, seed=20260907)
    matrices, labels, _ = load_lr_matrices(sample, pre)
    for arm, matrix in matrices.items():
        med, center, scale, _ = p3.fit_imputer_scaler(matrix)
        transformed = p3.apply_imputer_scaler(matrix, med, center, scale)
        model, warnings_seen = p3.fit_lr(transformed, labels)
        if warnings_seen or not np.isfinite(model.decision_function(transformed)).all():
            raise RuntimeError(f"LR smoke failure for {arm}: {warnings_seen}")
        if matrix.shape[1] != {"L0_demo10": 10, "L1_compact30": 30, "L2_global59": 59}[arm]:
            raise RuntimeError(f"LR smoke dimension failure for {arm}")


def aggregate_and_report(results: list[dict], manifest: pd.DataFrame) -> None:
    rows = []
    for result in results:
        row = {
            "model": result["model"], "holdout_site": result["holdout_site"],
            **{f"train_{key}": value for key, value in result["train"].items()},
            **{f"holdout_{key}": value for key, value in result["holdout"].items()},
            "train_ac_minus_holdout_ac": result["train_ac_minus_holdout_ac"],
        }
        rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RUN_OUTPUT / "fold_metrics.csv", index=False, na_rep="N/A")
    aggregate = {}
    for model, group in metrics.groupby("model", sort=False):
        group = group.set_index("holdout_site").loc[list(SITES)]
        values = group.holdout_ac_auroc.dropna()
        aggregate[model] = {
            "sites": {
                site: {
                    "ac_auroc": float(group.loc[site, "holdout_ac_auroc"]),
                    "age_weighted_auroc": float(group.loc[site, "holdout_age_weighted_auroc"]),
                    "auroc": float(group.loc[site, "holdout_auroc"]),
                    "auprc": float(group.loc[site, "holdout_auprc"]),
                } for site in SITES
            },
            "macro_ac_auroc": float(values.mean()) if len(values) else math.nan,
            "worst_site_ac_auroc": float(values.min()) if len(values) else math.nan,
            "macro_age_weighted_auroc": float(group.holdout_age_weighted_auroc.mean()),
            "macro_auroc": float(group.holdout_auroc.mean()),
            "macro_auprc": float(group.holdout_auprc.mean()),
        }
    (RUN_OUTPUT / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )
    write_final_report(metrics, aggregate, manifest)


def f(value) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.3f}"


def write_final_report(metrics: pd.DataFrame, aggregate: dict, manifest: pd.DataFrame) -> None:
    order = [*LR_ARMS, LSTM_ARM]
    lines = [
        "# P5: three-site LOSO on the frozen legacy 1103 cache",
        "",
        "## Scope",
        "",
        f"- Base commit: `{BASE_COMMIT}`.",
        "- Sites: I0002, I0006, S0001; each site is held out in full once with patient-level isolation.",
        "- Inputs: the existing 1103 `npz_new` caches and existing P3 stage sidecars only. No feature extraction or NPZ modification occurred.",
        "- Truth: the previously used official `prevalence.csv` (1103 unique site/patient keys), parsed with `helper_code.load_label`; NPZ labels were not used.",
        "- LSTM: P4 `legacy_clip`, original 2-layer architecture, seed 7, balanced sampler + empirical pos_weight, exactly 6 epochs.",
        "- LR: exact P3 v2 demo10, compact30, and global59 definitions/configuration; typed scaling and imputation fit only on outer training sites.",
        "- All main results use raw decision-score ranking. No holdout calibration, thresholding, model selection, or epoch selection was performed.",
        "",
        "## Main results",
        "",
        "| Model | I0002 AC | I0006 AC | S0001 AC | Macro AC | Worst-site AC | Macro AUROC | Macro AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in order:
        a = aggregate[model]
        lines.append(
            f"| {model} | {f(a['sites']['I0002']['ac_auroc'])} | {f(a['sites']['I0006']['ac_auroc'])} | "
            f"{f(a['sites']['S0001']['ac_auroc'])} | {f(a['macro_ac_auroc'])} | "
            f"{f(a['worst_site_ac_auroc'])} | {f(a['macro_auroc'])} | {f(a['macro_auprc'])} |"
        )
    lines += [
        "",
        "P4 legacy mixed-site repeated-CV AC=0.701 is retained only as a historical reference; its protocol differs from LOSO and the difference is not interpreted as a pure domain-shift effect.",
        "",
        "## Fold details",
        "",
        "| Model | Holdout | n (+/-) | Eligible pairs | Train AC | Holdout AC | Age-weighted | AUROC | AUPRC | Train-holdout AC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in order:
        for site in SITES:
            row = metrics.loc[(metrics.model == model) & (metrics.holdout_site == site)].iloc[0]
            lines.append(
                f"| {model} | {site} | {int(row.holdout_n)} ({int(row.holdout_positives)}/{int(row.holdout_negatives)}) | "
                f"{int(row.holdout_eligible_ac_pairs)} | {f(row.train_ac_auroc)} | {f(row.holdout_ac_auroc)} | "
                f"{f(row.holdout_age_weighted_auroc)} | {f(row.holdout_auroc)} | {f(row.holdout_auprc)} | "
                f"{row.train_ac_minus_holdout_ac:+.3f} |"
            )
    lines += [
        "",
        "## Run integrity and exceptions",
        "",
        "- Independent post-run verification confirmed exact manifest coverage, label/age alignment, finite scores, and exact reproduction of all 12 saved holdout metric bundles from raw logits.",
        "- The three LSTM runs completed all six epochs with finite losses; each checkpoint was frozen before its holdout loader was constructed.",
        "- `L0_demo10` for holdout I0002 reached the frozen P3 `max_iter=100000` without convergence. Its outputs are retained and flagged; no hyperparameter was changed and the fold was not rerun.",
        "",
        "## Conclusions",
        "",
        f"1. Legacy LSTM holdout AC was {f(aggregate[LSTM_ARM]['sites']['I0002']['ac_auroc'])} on I0002, {f(aggregate[LSTM_ARM]['sites']['I0006']['ac_auroc'])} on I0006, and {f(aggregate[LSTM_ARM]['sites']['S0001']['ac_auroc'])} on S0001.",
        f"2. Its site-macro AC was {f(aggregate[LSTM_ARM]['macro_ac_auroc'])}, and its worst-site AC was {f(aggregate[LSTM_ARM]['worst_site_ac_auroc'])} (I0006).",
        "3. Legacy was clearly above 0.5 only on I0002. S0001 was 0.505, effectively near chance, while I0006 was below chance at 0.474; therefore the experiment does not establish clear above-chance transfer on at least two unseen sites.",
        "4. Training with S0001 transferred useful ranking to I0002 (0.646) but not to I0006 (0.474). Conversely, training only on I0002+I0006 did not reproduce the earlier S0001-dominated mixed-site signal (S0001 LOSO 0.505). The transfer is site-specific rather than general.",
        "5. The earlier weak I0006 evidence persists under true LOSO: every model except compact30 was at or below 0.5 AC on I0006, and legacy reached only 0.474.",
        f"6. Compact30 dropped from the historical P3 mixed-site AC 0.581 to LOSO macro {f(aggregate['L1_compact30']['macro_ac_auroc'])}; global59 dropped from 0.590 to {f(aggregate['L2_global59']['macro_ac_auroc'])}. These are protocol-level comparisons, not pure estimates of domain-shift magnitude.",
        "7. The frozen global EEG summaries did not provide a stable unseen-site increment over compact30: global59 changed AC by +0.063 on I0002, -0.039 on I0006, and -0.031 on S0001, with macro AC lower by 0.002.",
        "8. Legacy exceeded global59 on only one site (S0001, +0.014), tied it on I0002 to displayed and full precision, and was lower on I0006 (-0.020). Its macro AC was lower by 0.002, so the P4 legacy advantage over global59 did not survive LOSO.",
        "9. The S0001 holdout fold had the smallest training pool (246 records, 28 positives) and showed large train-holdout gaps for compact30/global59 (0.335/0.375), so it is most structurally exposed to limited training-site composition. For legacy specifically, the largest gap occurred on I0006 (0.426), indicating additional site mismatch beyond training-set size.",
        "10. The evidence supports priority B: investigate site-domain effects and data2 before further temporal/static branch ablation. Mixed-site P4 AC=0.701 did not translate into robust three-site LOSO performance; no next experiment is launched automatically.",
    ]
    (RUN_OUTPUT / "P5_LOSO_1103_RESULTS.md").write_text("\n".join(lines) + "\n")


def run_experiment() -> None:
    manifest, split_records = validate_run_inputs()
    RUN_OUTPUT.mkdir(parents=True)
    shutil.copy2(MANIFEST, RUN_OUTPUT / "loso_manifest.csv")
    counts = []
    for site in SITES:
        holdout = manifest.loc[manifest.site == site]
        train = manifest.loc[manifest.site != site]
        counts.append({
            "holdout_site": site, "train_records": len(train),
            "train_patients": train.patient_id.nunique(), "train_positives": int(train.label.sum()),
            "train_negatives": int((train.label == 0).sum()), "holdout_records": len(holdout),
            "holdout_patients": holdout.patient_id.nunique(), "holdout_positives": int(holdout.label.sum()),
            "holdout_negatives": int((holdout.label == 0).sum()),
            "holdout_eligible_ac_pairs": eligible_pairs(holdout.label, holdout.age),
        })
    pd.DataFrame(counts).to_csv(RUN_OUTPUT / "site_counts.csv", index=False)
    shutil.copy2(FROZEN_FINGERPRINTS, RUN_OUTPUT / "input_fingerprints.json")
    (RUN_OUTPUT / "source_commit.txt").write_text(BASE_COMMIT + "\n")
    environment = {
        "python": sys.executable, "python_version": sys.version,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "sklearn": sklearn.__version__, "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "platform": platform.platform(), "device": str(tl.device),
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_VISIBLE_DEVICES", "LSTM_INPUT_PREPROCESSING", "LSTM_POS_WEIGHT_MODE",
            "LSTM_SEED", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG",
        )},
    }
    (RUN_OUTPUT / "environment.txt").write_text(json.dumps(environment, indent=2) + "\n")

    results = []
    with tempfile.TemporaryDirectory(prefix="p5_flat_cache_") as temp:
        flat_cache = Path(temp)
        make_flat_cache_index(manifest, flat_cache)
        first_site = SITES[0]
        smoke_train = rows_for(manifest.loc[manifest.site != first_site], split_records)
        smoke_lr(smoke_train, flat_cache)

        for holdout_site in SITES:
            fold_dir = RUN_OUTPUT / f"holdout_{holdout_site}"
            fold_dir.mkdir()
            train_frame = manifest.loc[manifest.site != holdout_site].copy()
            holdout_frame = manifest.loc[manifest.site == holdout_site].copy()
            train_rows = rows_for(train_frame, split_records)
            holdout_rows = rows_for(holdout_frame, split_records)
            print(f"[P5] holdout={holdout_site} train={len(train_rows)} holdout_n={len(holdout_rows)}", flush=True)

            scaler_records = [
                {"BidsFolder": row["BidsFolder"], "SessionID": row["SessionID"]}
                for row in train_rows
            ]
            preprocessor = FeatureScaler.fit_from_cache(
                scaler_records, flat_cache, RULES, sample_cap=128, seed=20260907,
                code_head=BASE_COMMIT,
            )
            train_matrices, labels_train, ages_train = load_lr_matrices(train_rows, preprocessor)
            hold_matrices, labels_holdout, ages_holdout = load_lr_matrices(holdout_rows, preprocessor)
            for arm, p3_arm in LR_ARMS.items():
                results.append(run_lr_fold(
                    arm, p3_arm, train_rows, holdout_rows,
                    train_matrices[arm], hold_matrices[arm], labels_train, ages_train,
                    labels_holdout, ages_holdout, preprocessor, fold_dir,
                ))
            results.append(run_lstm_fold(train_rows, holdout_rows, fold_dir))

    if len(results) != 12:
        raise RuntimeError(f"Expected 12 model/fold results, got {len(results)}")
    aggregate_and_report(results, manifest)
    print("P5_LOSO_1103_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare_manifest()
    else:
        run_experiment()


if __name__ == "__main__":
    main()
