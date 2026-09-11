#!/usr/bin/env python
"""Run legacy_clip LSTM on the frozen P3 outer folds without resplitting."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("LSTM_INPUT_PREPROCESSING", "legacy_clip")
os.environ.setdefault("LSTM_POS_WEIGHT_MODE", "empirical")
os.environ.setdefault("LSTM_SEED", "7")
os.environ.setdefault("PYTHONHASHSEED", "7")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_lstm as tl
from evaluate_model import compute_auroc_age
from feature_scaling import FeatureScaler

BASE_COMMIT = "498380f1bee20afdbb78c55ca076262e6cd0a182"
FOLD_MANIFEST = Path(os.environ.get(
    "P4_FOLD_MANIFEST", ROOT / "p4_inputs" / "cv_fold_manifest.csv"
))
VAL_IDENTITIES = Path(os.environ.get(
    "P4_VAL_IDENTITIES", ROOT / "feat_input" / "p4_val_identity.csv"
))
CACHE_REQUESTED = Path(os.environ.get("LSTM_CACHE_DIR", ROOT / "npz_new"))
CACHE = next(
    (path for path in (CACHE_REQUESTED, CACHE_REQUESTED / "npz_new")
     if (path / "train").is_dir() and (path / "val").is_dir()),
    CACHE_REQUESTED,
)
SPLIT = ROOT / "split"
OUTPUT = Path(os.environ.get("P4_OUTPUT_DIR", ROOT / "p4_h100_results"))
MODEL_SEED = 7
OUTER_SEEDS = (7, 17, 29)
ARMS = ("best_val_ac", "fixed_epoch6")


def record_id(record):
    return f"{record['BidsFolder']}_ses-{record['SessionID']}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        h.update(name.encode())
        h.update(state[name].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_records(name):
    return json.loads((SPLIT / f"{name}_records.json").read_text())


def cache_path(split, rid):
    return CACHE / split / f"{rid}.npz"


def metrics(labels, logits, ages):
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=float)
    ages = np.asarray(ages, dtype=float)
    if not np.isfinite(logits).all():
        raise RuntimeError("Non-finite decision logits")
    pos_age = ages[labels == 1]
    neg_age = ages[labels == 0]
    pairs = int((np.abs(pos_age[:, None] - neg_age[None, :]) <= 2).sum())
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "eligible_age_pairs_gap2": pairs,
        "ac_auroc": float(compute_auroc_age(labels, logits, ages, gap=2)),
        "auroc": float(roc_auc_score(labels, logits)),
        "auprc": float(average_precision_score(labels, logits)),
    }


def collect(model, loader):
    y, logits, ages = tl.collect_outputs(model, loader)
    return y.astype(int), logits.astype(float), ages.astype(float)


def make_loader(records, split, preprocessor, sampler=None):
    dataset = tl.PSGDataset(
        records, str(CACHE / split), extractor=None, preprocessor=preprocessor
    )
    return DataLoader(
        dataset, batch_size=tl.BATCH_SIZE,
        shuffle=False, sampler=sampler, collate_fn=tl.collate_fn,
        num_workers=0,
    )


def configure_fold_log(path):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if not any(getattr(h, "_p4_console", False) for h in root.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        console._p4_console = True
        root.addHandler(console)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return handler


def remove_log_handler(handler):
    root = logging.getLogger()
    root.removeHandler(handler)
    handler.close()


def validate_inputs():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT}")
    if not FOLD_MANIFEST.is_file():
        raise FileNotFoundError(FOLD_MANIFEST)
    if not VAL_IDENTITIES.is_file():
        raise FileNotFoundError(VAL_IDENTITIES)
    if tl.INPUT_PREPROCESSING != "legacy_clip":
        raise RuntimeError(f"Expected legacy_clip, got {tl.INPUT_PREPROCESSING}")
    if tl.POS_WEIGHT_MODE != "empirical" or tl.SEED != MODEL_SEED:
        raise RuntimeError(
            f"Expected empirical/seed7, got {tl.POS_WEIGHT_MODE}/{tl.SEED}"
        )
    expected = {
        "seq": 483, "ecg": 12, "static": 196, "input": 495,
        "hidden": 128, "layers": 2, "fc": 64, "batch": 8,
        "epochs": 80, "patience": 15,
    }
    actual = {
        "seq": tl.SEQ_DIM, "ecg": tl.ECG_DIM, "static": tl.STATIC_DIM,
        "input": tl.INPUT_DIM, "hidden": tl.HIDDEN_DIM,
        "layers": tl.NUM_LAYERS, "fc": tl.FC_HIDDEN,
        "batch": tl.BATCH_SIZE, "epochs": tl.EPOCHS,
        "patience": tl.PATIENCE,
    }
    if actual != expected or tl.LR != 1e-3 or tl.DROPOUT != 0.3:
        raise RuntimeError(f"Frozen LSTM configuration differs: {actual}")

    manifest = pd.read_csv(FOLD_MANIFEST)
    required = {
        "seed", "fold", "role", "record_id", "bdsp_patient_id",
        "site_id", "label", "raw_age",
    }
    if set(manifest.columns) != required:
        raise RuntimeError(f"Unexpected fold manifest schema: {manifest.columns}")
    if set(manifest.seed.unique()) != set(OUTER_SEEDS):
        raise RuntimeError("Outer seeds differ from 7/17/29")
    base_ids = set(manifest.record_id)
    base_records = {record_id(r): r for r in load_records("train")}
    if len(base_ids) != 733 or base_ids != set(base_records):
        raise RuntimeError("P3 base records differ from frozen train split")
    for seed in OUTER_SEEDS:
        holdout_union = []
        for fold in range(3):
            block = manifest[(manifest.seed == seed) & (manifest.fold == fold)]
            train = block[block.role == "train"]
            hold = block[block.role == "holdout"]
            if len(train) + len(hold) != 733:
                raise RuntimeError(f"Bad fold size: {seed}/{fold}")
            if set(train.record_id) & set(hold.record_id):
                raise RuntimeError(f"Record overlap: {seed}/{fold}")
            if set(train.bdsp_patient_id.astype(str)) & set(hold.bdsp_patient_id.astype(str)):
                raise RuntimeError(f"BDSPPatientID overlap: {seed}/{fold}")
            holdout_union.extend(hold.record_id.tolist())
        if len(holdout_union) != 733 or len(set(holdout_union)) != 733:
            raise RuntimeError(f"Holdout coverage failure for seed {seed}")

    val = pd.read_csv(VAL_IDENTITIES)
    if set(val.columns) != {"record_id", "bdsp_patient_id"} or len(val) != 158:
        raise RuntimeError("Invalid frozen val identity file")
    base_patients = set(manifest.bdsp_patient_id.astype(str))
    val_patients = set(val.bdsp_patient_id.astype(str))
    overlap = sorted(base_patients & val_patients)
    if overlap:
        raise RuntimeError(f"Base/val BDSPPatientID overlap: {overlap[:5]}")
    val_records = {record_id(r): r for r in load_records("val")}
    if set(val.record_id) != set(val_records):
        raise RuntimeError("Val identity records differ from fixed val split")

    missing = []
    for rid in sorted(base_ids):
        if not cache_path("train", rid).is_file():
            missing.append(str(cache_path("train", rid)))
    for rid in sorted(val.record_id):
        if not cache_path("val", rid).is_file():
            missing.append(str(cache_path("val", rid)))
    if missing:
        raise RuntimeError(f"Frozen cache misses: {len(missing)}, first={missing[0]}")
    return manifest, base_records, val_records


def preflight_batch(manifest, base_records):
    first = manifest[
        (manifest.seed == 7) & (manifest.fold == 0) & (manifest.role == "train")
    ]
    records = [base_records[rid] for rid in first.record_id]
    pre = FeatureScaler.legacy_clip()
    loader = make_loader(records[:8], "train", pre)
    batch = next(iter(loader))
    if batch is None:
        raise RuntimeError("Empty preflight batch")
    tl.seed_everything(MODEL_SEED)
    model = tl.LSTMModel().to(tl.device)
    xseq, xecg, mask, static, ages, labels, lengths = batch
    with torch.no_grad():
        logits = model(
            xseq.to(tl.device), xecg.to(tl.device), mask.to(tl.device),
            static.to(tl.device), lengths.to(tl.device),
        )
    if logits.shape != (len(records[:8]),) or not torch.isfinite(logits).all():
        raise RuntimeError("One-batch forward preflight failed")
    return {
        "batch_size": len(records[:8]), "x_seq_shape": list(xseq.shape),
        "x_ecg_shape": list(xecg.shape), "x_static_shape": list(static.shape),
        "finite_logits": True,
    }


def prediction_frame(records, y, ages, logits, calibrator):
    return pd.DataFrame({
        "record_id": [record_id(r) for r in records],
        "label": np.asarray(y, dtype=int),
        "raw_age": np.asarray(ages, dtype=float),
        "decision_logit": np.asarray(logits, dtype=float),
        "sigmoid_probability": tl.sigmoid_np(logits),
        "val_platt_probability": tl.apply_calibrator(logits, calibrator),
    })


def evaluate_validation_and_train(state, model, val_loader, train_loader, train_records):
    model.load_state_dict(state)
    val_y, val_logits, val_ages = collect(model, val_loader)
    calibrator = tl.fit_platt_calibrator(val_logits, val_y)
    val_prob = tl.apply_calibrator(val_logits, calibrator)
    threshold, threshold_tpr, threshold_fpr = tl.select_youden_threshold(
        val_y, val_prob
    )
    train_y, train_logits, train_ages = collect(model, train_loader)
    result = {
        "validation": metrics(val_y, val_logits, val_ages),
        "natural_outer_train": metrics(train_y, train_logits, train_ages),
        "calibrator": calibrator,
        "threshold": threshold,
        "threshold_tpr": threshold_tpr,
        "threshold_fpr": threshold_fpr,
    }
    return result, prediction_frame(
        train_records, train_y, train_ages, train_logits, calibrator
    )


def evaluate_holdout(state, model, holdout_loader, holdout_records, calibrator):
    model.load_state_dict(state)
    y, logits, ages = collect(model, holdout_loader)
    return (
        metrics(y, logits, ages),
        prediction_frame(holdout_records, y, ages, logits, calibrator),
    )


def run_fold(seed, fold, manifest, base_records, val_records):
    fold_dir = OUTPUT / "seed7_model" / f"outer_seed{seed}_fold{fold}"
    fold_dir.mkdir(parents=True)
    log_handler = configure_fold_log(fold_dir / "train.log")
    log = logging.getLogger("p4_legacy_lstm_cv")
    try:
        block = manifest[(manifest.seed == seed) & (manifest.fold == fold)]
        train_rows = block[block.role == "train"]
        holdout_rows = block[block.role == "holdout"]
        block.to_csv(fold_dir / "frozen_outer_fold_manifest.csv", index=False)
        train_records = [base_records[rid] for rid in train_rows.record_id]
        holdout_records = [base_records[rid] for rid in holdout_rows.record_id]
        fixed_val_records = list(val_records.values())
        log.info(
            "START outer_seed=%d fold=%d train=%d holdout=%d fixed_val=%d model_seed=7",
            seed, fold, len(train_records), len(holdout_records), len(fixed_val_records),
        )
        tl.seed_everything(MODEL_SEED)
        pre = FeatureScaler.legacy_clip()
        generator = torch.Generator().manual_seed(MODEL_SEED)
        sampler = tl.build_balanced_sampler(
            train_records, str(CACHE / "train"), generator=generator
        )
        train_dataset = tl.PSGDataset(
            train_records, str(CACHE / "train"), extractor=None, preprocessor=pre
        )
        train_loader = DataLoader(
            train_dataset, batch_size=tl.BATCH_SIZE, sampler=sampler,
            shuffle=(sampler is None), collate_fn=tl.collate_fn,
            num_workers=0, generator=generator,
        )
        val_loader = make_loader(fixed_val_records, "val", pre)

        # Preserve the production entry's first-batch diagnostic call order.
        first_batch = next(iter(train_loader))
        if first_batch is None:
            raise RuntimeError("First sampled training batch is empty")
        for name, tensor in zip(
            ("X_seq", "X_ecg", "x_static"),
            (first_batch[0], first_batch[1], first_batch[3]),
        ):
            if not torch.isfinite(tensor).all():
                raise RuntimeError(f"Non-finite first-batch {name}")
            log.info(
                "first_batch %s shape=%s min=%.4f max=%.4f",
                name, tuple(tensor.shape), tensor.min().item(), tensor.max().item(),
            )

        model = tl.LSTMModel().to(tl.device)
        initial_hash = state_hash(clone_state(model))
        n_pos, n_neg, n_missing = tl.count_cached_labels(
            train_records, str(CACHE / "train")
        )
        if n_missing or n_pos <= 0 or n_neg <= 0:
            raise RuntimeError(
                f"Invalid outer training cache labels: pos={n_pos} neg={n_neg} "
                f"missing={n_missing}"
            )
        pos_weight_value = tl.resolve_pos_weight("empirical", n_pos, n_neg)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight_value], device=tl.device)
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=8
        )
        best_score = -float("inf")
        best_state = None
        epoch6_state = None
        best_epoch = 0
        best_val = None
        patience_counter = 0
        history = []
        stop_epoch = 0
        for epoch in range(1, 81):
            started = time.time()
            loss = tl.train_epoch(model, train_loader, optimizer, criterion)
            val_y, val_logits, val_ages = collect(model, val_loader)
            val_metrics = metrics(val_y, val_logits, val_ages)
            selection_score = val_metrics["ac_auroc"]
            scheduler.step(selection_score)
            if epoch == 6:
                epoch6_state = clone_state(model)
            improved = best_state is None or selection_score > best_score
            if improved:
                best_score = selection_score
                best_state = clone_state(model)
                best_epoch = epoch
                best_val = val_metrics
                patience_counter = 0
            else:
                patience_counter += 1
            row = {
                "epoch": epoch, "train_loss": float(loss),
                "val_ac_auroc": val_metrics["ac_auroc"],
                "val_auroc": val_metrics["auroc"],
                "val_auprc": val_metrics["auprc"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "selection_improved": bool(improved),
                "elapsed_sec": float(time.time() - started),
            }
            history.append(row)
            log.info(
                "epoch=%d loss=%.6f val_ac=%.6f val_auroc=%.6f "
                "val_auprc=%.6f lr=%.8g improved=%s time=%.1fs",
                epoch, loss, val_metrics["ac_auroc"], val_metrics["auroc"],
                val_metrics["auprc"], optimizer.param_groups[0]["lr"],
                improved, row["elapsed_sec"],
            )
            stop_epoch = epoch
            if patience_counter >= 15:
                log.info("early_stop epoch=%d best_epoch=%d", epoch, best_epoch)
                break
        if best_state is None or epoch6_state is None:
            raise RuntimeError("Missing best or epoch-6 frozen state")
        pd.DataFrame(history).to_csv(fold_dir / "val_by_epoch_metrics.csv", index=False)

        # Derive checkpoint metadata without constructing an outer holdout loader.
        base_checkpoint = {
            "model_type": "lstm", "base_commit": BASE_COMMIT,
            "outer_seed": seed, "outer_fold": fold, "model_seed": MODEL_SEED,
            "input_preprocessing": pre.state_dict(),
            "input_preprocessing_mode": "legacy_clip",
            "pos_weight_mode": "empirical", "pos_weight": pos_weight_value,
            "train_label_counts": {"positive": n_pos, "negative": n_neg},
            "best_epoch": best_epoch, "stop_epoch": stop_epoch,
            "selection_metric": "val_age_conditioned_auroc_gap2",
            "initial_model_hash": initial_hash,
            "feature_dims": {"X_seq": 483, "X_ecg": 12, "x_static": 196},
            "training_config": {
                "batch_size": 8, "optimizer": "Adam", "lr": 1e-3,
                "max_epochs": 80, "early_stopping_patience": 15,
                "scheduler": "ReduceLROnPlateau", "scheduler_factor": 0.5,
                "scheduler_patience": 8, "gradient_clip_max_norm": 1.0,
                "weighted_random_sampler": True,
            },
            "outer_train_ids": train_rows.record_id.tolist(),
            "model_selection_val_ids": list(val_records.keys()),
        }
        natural_train_loader = make_loader(train_records, "train", pre)
        results = {
            "outer_seed": seed, "outer_fold": fold,
            "model_seed": MODEL_SEED, "best_epoch": best_epoch,
            "stop_epoch": stop_epoch, "best_selection_val": best_val,
            "pos_weight": pos_weight_value, "initial_model_hash": initial_hash,
            "checkpoints": {},
        }
        frozen = (("best_val_ac", best_state), ("fixed_epoch6", epoch6_state))
        for role, state in frozen:
            result, train_predictions = evaluate_validation_and_train(
                state, model, val_loader, natural_train_loader, train_records
            )
            results["checkpoints"][role] = result
            train_predictions.to_csv(
                fold_dir / f"{role}_outer_train_logits.csv", index=False
            )
            checkpoint = {
                **base_checkpoint, "checkpoint_role": role,
                "state_dict": state, "calibrator": result["calibrator"],
                "threshold": result["threshold"],
                "validation": result["validation"],
                "state_dict_sha256": state_hash(state),
            }
            if role == "fixed_epoch6":
                checkpoint["fixed_epoch"] = 6
            torch.save(checkpoint, fold_dir / f"{role}.pt")

        log.info(
            "Both complete checkpoints frozen; constructing outer holdout loader now"
        )
        holdout_loader = make_loader(holdout_records, "train", pre)
        for role, state in frozen:
            result = results["checkpoints"][role]
            holdout_metrics, holdout_predictions = evaluate_holdout(
                state, model, holdout_loader, holdout_records, result["calibrator"]
            )
            result["outer_holdout"] = holdout_metrics
            holdout_predictions.to_csv(
                fold_dir / f"{role}_outer_holdout_logits.csv", index=False
            )
            log.info(
                "%s outer_holdout ac=%.6f auroc=%.6f auprc=%.6f",
                role, result["outer_holdout"]["ac_auroc"],
                result["outer_holdout"]["auroc"],
                result["outer_holdout"]["auprc"],
            )
        (fold_dir / "fold_metrics.json").write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n"
        )
    finally:
        remove_log_handler(log_handler)


def main():
    manifest, base_records, val_records = validate_inputs()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    preflight = preflight_batch(manifest, base_records)
    OUTPUT.mkdir(parents=True)
    identity = {
        "base_commit": BASE_COMMIT,
        "fold_manifest": str(FOLD_MANIFEST),
        "fold_manifest_sha256": sha256(FOLD_MANIFEST),
        "val_identities": str(VAL_IDENTITIES),
        "val_identities_sha256": sha256(VAL_IDENTITIES),
        "cache": str(CACHE), "cache_npz_count": len(list(CACHE.rglob("*.npz"))),
        "model_seed": MODEL_SEED, "outer_seeds": list(OUTER_SEEDS),
        "python": sys.executable, "python_version": sys.version,
        "platform": platform.platform(), "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment": {
            name: os.environ.get(name) for name in (
                "CUDA_VISIBLE_DEVICES", "LSTM_INPUT_PREPROCESSING",
                "LSTM_POS_WEIGHT_MODE", "LSTM_SEED", "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
        "source_sha256": {
            name: sha256(ROOT / name) for name in (
                "train_lstm.py", "feature_scaling.py",
                "feat_input/feat_mody/run_p4_legacy_lstm_cv.py",
            )
        },
        "preflight": preflight,
        "val_base_bdsp_overlap": 0,
    }
    (OUTPUT / "run_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    for seed in OUTER_SEEDS:
        for fold in range(3):
            run_fold(seed, fold, manifest, base_records, val_records)
    print("P4_LEGACY_LSTM_CV_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
