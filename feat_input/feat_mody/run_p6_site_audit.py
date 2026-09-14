#!/usr/bin/env python3
"""P6-A: decode acquisition site from frozen legacy npz_new features."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import warnings

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pooled_logistic_v2 import COMPACT_INDICES, assemble_features

BASE_COMMIT = "69e2aecef497f879b6a9e9c1953474191b8e6d91"
MANIFEST = ROOT / "output/p5_loso_1103/loso_manifest.csv"
FINGERPRINTS = ROOT / "output/p5_loso_1103/input_fingerprints.json"
CACHE = ROOT / "npz_new"
SIDECAR = ROOT / "output/p3_v2/stage_sidecar"
OUT = ROOT / "output/p6_site_domain/site_audit"
SITES = ("I0002", "I0006", "S0001")
FAMILIES = (
    "demo10", "compact30", "static196", "eeg_spectral54", "coherence360",
    "bsr18", "emg24", "respiration14", "stage_event13", "ecg12", "global59",
)
LR_CONFIG = {
    "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 5000,
    "class_weight": "balanced", "random_state": 7,
}


def summarize_continuous(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        return np.full(2 * x.shape[1], np.nan, dtype=np.float64)
    return np.r_[np.nanmedian(x, axis=0), np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)]


def validate_manifest() -> pd.DataFrame:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    manifest = pd.read_csv(MANIFEST, dtype={"record_id": str, "patient_id": str, "site": str})
    if len(manifest) != 1103 or manifest.record_id.nunique() != 1103 or manifest.patient_id.nunique() != 1103:
        raise RuntimeError("P5 manifest identity mismatch")
    if set(manifest.site) != set(SITES):
        raise RuntimeError("Incomplete site labels")
    frozen = json.loads(FINGERPRINTS.read_text())
    missing = []
    for row in manifest.itertuples(index=False):
        path = ROOT / row.npz_path
        if path != CACHE / row.npz_partition / f"{row.record_id}.npz" or not path.is_file():
            missing.append(row.record_id)
    if missing or len(list(CACHE.glob("*/*.npz"))) != 1103:
        raise RuntimeError(f"Frozen legacy cache mismatch: missing={missing[:5]}")
    if frozen.get("cache_file_count") != 1103:
        raise RuntimeError("P5 cache fingerprint count mismatch")
    return manifest


def load_patient_features(manifest: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    values = {name: [] for name in FAMILIES}
    for row in manifest.itertuples(index=False):
        with np.load(ROOT / row.npz_path, allow_pickle=False) as z:
            seq = np.asarray(z["X_seq"], dtype=np.float64)
            ecg = np.asarray(z["X_ecg"], dtype=np.float64)
            static = np.asarray(z["x_static"], dtype=np.float64)
        if seq.ndim != 2 or seq.shape[1] != 483 or ecg.ndim != 2 or ecg.shape[1] != 12 or static.shape != (196,):
            raise RuntimeError(f"Legacy cache shape mismatch: {row.record_id}")
        values["demo10"].append(static[:10])
        values["compact30"].append(np.r_[static[:10], static[COMPACT_INDICES]])
        values["static196"].append(static)
        values["eeg_spectral54"].append(summarize_continuous(seq[:, 0:54]))
        values["coherence360"].append(summarize_continuous(seq[:, 54:414]))
        values["bsr18"].append(summarize_continuous(seq[:, 414:432]))
        values["emg24"].append(summarize_continuous(seq[:, 432:456]))
        values["respiration14"].append(summarize_continuous(seq[:, 456:470]))
        values["stage_event13"].append(np.nanmean(seq[:, 470:483], axis=0))
        values["ecg12"].append(summarize_continuous(ecg))
        side = SIDECAR / f"{row.record_id}.npz"
        if not side.is_file():
            raise RuntimeError(f"Missing P3 sidecar for global59: {row.record_id}")
        with np.load(side, allow_pickle=False) as s:
            global59 = assemble_features(
                static, np.asarray(s["stage_code"]), np.asarray(s["stage_valid"]),
                np.asarray(s["eeg_channel_available"]), seq, "P3_global59",
            )
        values["global59"].append(global59)
    matrices = {name: np.asarray(rows, dtype=np.float64) for name, rows in values.items()}
    expected = {
        "demo10": 10, "compact30": 30, "static196": 196, "eeg_spectral54": 108,
        "coherence360": 720, "bsr18": 36, "emg24": 48, "respiration14": 28,
        "stage_event13": 13, "ecg12": 24, "global59": 59,
    }
    for name, matrix in matrices.items():
        if matrix.shape != (1103, expected[name]):
            raise RuntimeError(f"Feature dimension mismatch: {name} {matrix.shape}")
    return matrices, manifest.site.map({site: i for i, site in enumerate(SITES)}).to_numpy(int)


def fit_impute_scale(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.zeros(x_train.shape[1], dtype=np.float64)
    for column in range(x_train.shape[1]):
        finite = np.isfinite(x_train[:, column])
        med[column] = np.median(x_train[finite, column]) if finite.any() else 0.0
    train = np.where(np.isfinite(x_train), x_train, med)
    test = np.where(np.isfinite(x_test), x_test, med)
    scaler = StandardScaler().fit(train)
    return scaler.transform(train), scaler.transform(test), med


def main() -> None:
    manifest = validate_manifest()
    OUT.mkdir(parents=True)
    confusion_dir = OUT / "site_confusion_matrices"
    confusion_dir.mkdir()
    matrices, labels = load_patient_features(manifest)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    rows, summaries = [], []
    for family in FAMILIES:
        matrix = matrices[family]
        aggregate_cm = np.zeros((3, 3), dtype=int)
        convergence_messages = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(matrix, labels)):
            train, test, _ = fit_impute_scale(matrix[train_idx], matrix[test_idx])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model = LogisticRegression(**LR_CONFIG).fit(train, labels[train_idx])
            messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
            convergence_messages.extend(messages)
            pred = model.predict(test)
            prob = model.predict_proba(test)
            cm = confusion_matrix(labels[test_idx], pred, labels=np.arange(3))
            aggregate_cm += cm
            rows.append({
                "feature_family": family, "feature_dimension": matrix.shape[1], "fold": fold,
                "balanced_accuracy": balanced_accuracy_score(labels[test_idx], pred),
                "macro_f1": f1_score(labels[test_idx], pred, average="macro", zero_division=0),
                "macro_ovr_auroc": roc_auc_score(labels[test_idx], prob, average="macro", multi_class="ovr"),
                "train_n": len(train_idx), "holdout_n": len(test_idx),
                "converged": not messages, "n_iter": int(model.n_iter_.max()),
            })
        family_rows = pd.DataFrame([row for row in rows if row["feature_family"] == family])
        summaries.append({
            "feature_family": family, "feature_dimension": matrix.shape[1],
            "mean_balanced_accuracy": family_rows.balanced_accuracy.mean(),
            "sd_balanced_accuracy": family_rows.balanced_accuracy.std(ddof=1),
            "min_balanced_accuracy": family_rows.balanced_accuracy.min(),
            "max_balanced_accuracy": family_rows.balanced_accuracy.max(),
            "mean_macro_f1": family_rows.macro_f1.mean(),
            "mean_macro_ovr_auroc": family_rows.macro_ovr_auroc.mean(),
            "all_folds_converged": not convergence_messages,
            "convergence_warnings": " | ".join(sorted(set(convergence_messages))),
        })
        pd.DataFrame(aggregate_cm, index=SITES, columns=SITES).rename_axis("true_site").to_csv(
            confusion_dir / f"{family}.csv"
        )
        print(json.dumps(summaries[-1]), flush=True)
    fold = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries).sort_values("mean_balanced_accuracy", ascending=False)
    fold.to_csv(OUT / "site_feature_cv_metrics.csv", index=False)
    summary.to_csv(OUT / "site_feature_summary.csv", index=False)
    lines = [
        "# P6-A site decodability audit", "",
        f"Base commit: `{BASE_COMMIT}`. Input was exclusively the frozen 1103-record `npz_new` manifest from P5.", "",
        "Each temporal continuous family was summarized by per-column median and IQR; stage/event columns used whole-night means. Five-fold site-stratified CV used fixed seed 7. Median imputation and StandardScaler were fitted within each training fold, followed by fixed L2 logistic regression (`C=1`, balanced class weights, no tuning).", "",
        "| Feature family | Dim | Balanced accuracy mean | SD | Min-max | Macro F1 | Macro OVR AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| {row.feature_family} | {row.feature_dimension} | {row.mean_balanced_accuracy:.3f} | {row.sd_balanced_accuracy:.3f} | {row.min_balanced_accuracy:.3f}-{row.max_balanced_accuracy:.3f} | {row.mean_macro_f1:.3f} | {row.mean_macro_ovr_auroc:.3f} |")
    by_name = summary.set_index("feature_family")
    lines += [
        "", "## Diagnostic conclusions", "",
        "1. Respiration was the most site-decodable individual family "
        f"(balanced accuracy {by_name.loc['respiration14', 'mean_balanced_accuracy']:.3f}); "
        "global59, EMG, static196, and compact30 also carried strong site signatures.",
        f"2. Compact30 ({by_name.loc['compact30', 'mean_balanced_accuracy']:.3f}) was only "
        f"slightly less site-decodable than static196 ({by_name.loc['static196', 'mean_balanced_accuracy']:.3f}); "
        "compactness did not remove most site information.",
        f"3. Coherence360 was clearly site-decodable ({by_name.loc['coherence360', 'mean_balanced_accuracy']:.3f}) "
        "but was not the strongest family despite its 720 summary dimensions.",
        f"4. EEG spectral features also showed a clear site signature ({by_name.loc['eeg_spectral54', 'mean_balanced_accuracy']:.3f}), "
        "similar in magnitude to coherence.",
        "5. Several high-dimensional physiological families carry site information consistent with a domain-shift concern, but this audit does not establish causality for P5 LOSO failure.",
        "", "Random balanced-accuracy reference is approximately 0.333. High site decodability does not by itself justify feature removal.",
    ]
    (OUT / "SITE_DECODABILITY_AUDIT.md").write_text("\n".join(lines) + "\n")
    print("P6_SITE_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
