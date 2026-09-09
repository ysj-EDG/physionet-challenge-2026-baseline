"""Train the fixed P2 static logistic-regression baselines in one process."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

from evaluate_model import compute_auroc_age
from feature_scaling import FeatureScaler, SCALER_VERSION
from static_logistic import MODEL_TYPE, STATIC_DIM, StaticLogisticModel


LOGGER = logging.getLogger("train_static_lr")
SEQ_DIM = 483
ECG_DIM = 12
SEED = 7
COMPACT_INDICES = [
    11, 12, 13, 14, 16, 17, 18, 19, 20, 32,
    34, 87, 96, 97, 98, 126, 127, 141, 168, 173,
]
FEATURE_SETS = {
    "I_LR_demo10": list(range(10)),
    "J_LR_compact30": list(range(10)) + COMPACT_INDICES,
    "K_LR_static196": list(range(STATIC_DIM)),
}
EXPECTED_COMPACT_NAMES = [
    "caisr_sleep_tst_sec",
    "caisr_sleep_se",
    "caisr_sleep_sol_sec",
    "caisr_sleep_rem_latency_sec",
    "caisr_sleep_waso_sec",
    "caisr_sleep_n1_pct",
    "caisr_sleep_n2_pct",
    "caisr_sleep_n3_pct",
    "caisr_sleep_rem_pct",
    "caisr_sleep_transition_rate",
    "caisr_sleep_short_bout_ratio",
    "caisr_sleep_mean_stage_entropy",
    "caisr_arousal_arousal_index",
    "caisr_arousal_arousal_burden_ratio",
    "caisr_arousal_mean_arousal_duration",
    "caisr_respiratory_respiratory_event_index",
    "caisr_respiratory_respiratory_burden_ratio",
    "caisr_respiratory_mean_resp_duration",
    "caisr_limb_limb_movement_index",
    "caisr_limb_plmi",
]
LR_CONFIG = {
    "penalty": "elasticnet",
    "solver": "saga",
    "l1_ratio": 0.4,
    "C": 0.03,
    "class_weight": "balanced",
    "max_iter": 10000,
    "tol": 1e-4,
    "random_state": SEED,
    "fit_intercept": True,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_key(record: dict) -> str:
    return f"{record['BidsFolder']}_ses-{record['SessionID']}"


def load_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    keys = [record_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate records in {path}")
    return records


def load_static_split(
    records: list[dict],
    cache_dir: Path,
    preprocessor: FeatureScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, labels, ages = [], [], []
    placeholder_seq = np.zeros((1, SEQ_DIM), dtype=np.float32)
    placeholder_ecg = np.zeros((0, ECG_DIM), dtype=np.float32)
    for record in records:
        rid = record_key(record)
        path = cache_dir / f"{rid}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen cache: {path}")
        with np.load(path, allow_pickle=False) as data:
            x_static = np.asarray(data["x_static"])
            label = int(np.asarray(data["y"]).item())
        if x_static.shape != (STATIC_DIM,):
            raise RuntimeError(f"{rid}: invalid x_static shape {x_static.shape}")
        if label not in {0, 1}:
            raise RuntimeError(f"{rid}: non-binary training label {label}")
        raw_age = float(x_static[0])
        _, _, transformed = preprocessor.transform_arrays(
            placeholder_seq,
            placeholder_ecg,
            x_static,
            record_id=rid,
            fallback_sequence=True,
        )
        features.append(transformed)
        labels.append(label)
        ages.append(raw_age)
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.shape != (len(records), STATIC_DIM) or not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"Invalid transformed static matrix {matrix.shape}")
    return matrix, np.asarray(labels, dtype=int), np.asarray(ages, dtype=float)


def fit_calibrator(logits: np.ndarray, labels: np.ndarray) -> dict:
    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(np.asarray(logits).reshape(-1, 1), labels)
    return {
        "method": "platt_logistic",
        "coef": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
    }


def apply_calibrator(logits: np.ndarray, calibrator: dict) -> np.ndarray:
    values = calibrator["coef"] * np.asarray(logits) + calibrator["intercept"]
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5, 0.0, 0.0
    indices = np.flatnonzero(finite)
    best = indices[int(np.argmax(tpr[finite] - fpr[finite]))]
    return float(thresholds[best]), float(tpr[best]), float(fpr[best])


def metrics(labels: np.ndarray, logits: np.ndarray, ages: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(labels, logits)),
        "age_conditioned_auroc": float(compute_auroc_age(labels, logits, ages, gap=2)),
        "auprc": float(average_precision_score(labels, logits)),
    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    root = Path(__file__).resolve().parent
    cache_root = Path(os.environ.get("LSTM_CACHE_DIR", root / "npz_new")).resolve()
    splits_root = Path(os.environ.get("LSTM_SPLITS_DIR", root / "split")).resolve()
    source_path = Path(os.environ["LSTM_PREPROCESSOR_STATE_CHECKPOINT"]).resolve()
    output_root = Path(
        os.environ.get("P2_OUTPUT_DIR", root / "output" / "p2_model_controls" / "seed7")
    ).resolve()
    for arm in FEATURE_SETS:
        if (output_root / arm).exists():
            raise FileExistsError(f"Refusing to overwrite: {output_root / arm}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    preprocessor = FeatureScaler.from_state_dict(
        source_checkpoint["input_preprocessing"]
    ).with_runtime_config(clip_z=None, mask_config={})
    if preprocessor.mode != SCALER_VERSION:
        raise RuntimeError(f"Expected {SCALER_VERSION}, got {preprocessor.mode}")
    static_rules = {
        int(rule["index"]): rule["name"]
        for rule in preprocessor.rules
        if rule["branch"] == "x_static"
    }
    if len(static_rules) != STATIC_DIM:
        raise RuntimeError(f"Expected {STATIC_DIM} static rules, got {len(static_rules)}")
    actual_compact_names = [static_rules[index] for index in COMPACT_INDICES]
    if actual_compact_names != EXPECTED_COMPACT_NAMES:
        raise RuntimeError(
            "Frozen compact feature mapping mismatch: "
            f"expected={EXPECTED_COMPACT_NAMES} actual={actual_compact_names}"
        )

    train_records = load_records(splits_root / "train_records.json")
    val_records = load_records(splits_root / "val_records.json")
    X_train, y_train, age_train = load_static_split(
        train_records, cache_root / "train", preprocessor
    )
    X_val, y_val, age_val = load_static_split(
        val_records, cache_root / "val", preprocessor
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    if not np.all(np.isfinite(X_train_scaled)) or not np.all(np.isfinite(X_val_scaled)):
        raise RuntimeError("StandardScaler produced non-finite values")

    try:
        code_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.SubprocessError):
        code_head = None

    for arm, selected_indices in FEATURE_SETS.items():
        arm_dir = output_root / arm
        arm_dir.mkdir()
        selected_names = [static_rules[index] for index in selected_indices]
        classifier = LogisticRegression(**LR_CONFIG)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(X_train_scaled[:, selected_indices], y_train)
        convergence_warnings = [
            str(item.message)
            for item in caught
            if issubclass(item.category, ConvergenceWarning)
        ]
        train_logits = classifier.decision_function(
            X_train_scaled[:, selected_indices]
        )
        val_logits = classifier.decision_function(X_val_scaled[:, selected_indices])
        calibrator = fit_calibrator(val_logits, y_val)
        val_probability = apply_calibrator(val_logits, calibrator)
        threshold, threshold_tpr, threshold_fpr = select_threshold(
            y_val, val_probability
        )
        coefficients = np.asarray(classifier.coef_[0], dtype=np.float64)
        intercept = float(classifier.intercept_[0])
        network = StaticLogisticModel(
            selected_indices,
            scaler.mean_,
            scaler.scale_,
            coefficients,
            intercept,
        )
        train_summary = metrics(y_train, train_logits, age_train)
        val_summary = metrics(y_val, val_logits, age_val)
        nonzero = int(np.count_nonzero(coefficients))
        checkpoint = {
            "model_type": MODEL_TYPE,
            "state_dict": network.state_dict(),
            "seed": SEED,
            "best_epoch": None,
            "feature_dims": {
                "X_seq": SEQ_DIM,
                "X_ecg": ECG_DIM,
                "x_static": STATIC_DIM,
            },
            "input_preprocessing": preprocessor.state_dict(),
            "input_preprocessing_source": {
                "checkpoint": str(source_path),
                "sha256": sha256(source_path),
            },
            "selected_indices": list(selected_indices),
            "selected_names": selected_names,
            "linear_scaler": {
                "type": "sklearn.preprocessing.StandardScaler",
                "fit_split": "train",
                "fit_record_count": len(train_records),
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "var": scaler.var_.tolist(),
                "n_samples_seen": int(scaler.n_samples_seen_),
            },
            "linear_model": {
                "type": "sklearn.linear_model.LogisticRegression",
                "config": LR_CONFIG,
                "coefficients": coefficients.tolist(),
                "intercept": intercept,
                "nonzero_coefficients": nonzero,
                "n_iter": np.asarray(classifier.n_iter_, dtype=int).tolist(),
                "converged": not convergence_warnings,
                "convergence_warnings": convergence_warnings,
            },
            "calibrator": calibrator,
            "threshold": threshold,
            "threshold_source": "validation_youden_calibrated_probability",
            "train": train_summary,
            "validation": {
                **val_summary,
                "threshold_tpr": threshold_tpr,
                "threshold_fpr": threshold_fpr,
                "calibrated_probability_mean": float(np.mean(val_probability)),
            },
            "code_head": code_head,
        }
        model_path = arm_dir / "lstm_model.pt"
        torch.save(checkpoint, model_path)
        log_lines = [
            f"arm={arm}",
            f"model_type={MODEL_TYPE}",
            f"selected_dimension={len(selected_indices)}",
            f"selected_indices={selected_indices}",
            f"selected_names={selected_names}",
            f"train_records={len(train_records)} val_records={len(val_records)}",
            f"train_metrics={train_summary}",
            f"validation_metrics={val_summary}",
            f"calibrator={calibrator}",
            f"threshold={threshold} tpr={threshold_tpr} fpr={threshold_fpr}",
            f"n_iter={classifier.n_iter_.tolist()} converged={not convergence_warnings}",
            f"nonzero_coefficients={nonzero}",
            f"convergence_warnings={convergence_warnings}",
            f"model_path={model_path}",
        ]
        (arm_dir / "train.log").write_text(
            "\n".join(log_lines) + "\n", encoding="utf-8"
        )
        LOGGER.info(
            "%s saved: dim=%d nonzero=%d n_iter=%s val_AC=%.4f val_AUROC=%.4f",
            arm,
            len(selected_indices),
            nonzero,
            classifier.n_iter_.tolist(),
            val_summary["age_conditioned_auroc"],
            val_summary["auroc"],
        )


if __name__ == "__main__":
    main()
