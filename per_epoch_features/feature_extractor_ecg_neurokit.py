#!/usr/bin/env python
"""
ECG HRV 特征提取器 — 基于 neurokit2 + Kubios 风格间期校正 + 5 分钟分段。

管道路径:
    原始 ECG → 5min 分段 → ecg_clean → ecg_peaks
    → signal_fixpeaks(Kubios) → hrv_time/freq/nonlinear/symbolic
    → 每段 11 维 → 跨段 mean+std → 22 维
"""

import logging
import warnings
import numpy as np
import pandas as pd

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

import neurokit2 as nk


logger = logging.getLogger("ecg_neurokit")

EPS = 1e-12
WINDOW_SEC = 300  # 5 分钟
MIN_WINDOW_SEC = 270  # 最小窗口时长

# 11 个 5min-window ECG/HRV 特征。
BASE_MODEL_COLUMNS = [
    "model_HRV_MedianNN",
    "model_HRV_MCVNN",
    "model_HRV_CVNN",
    "model_HRV_CVSD",
    "model_HRV_pNN20",
    "model_log_HRV_LF",
    "model_log_HRV_HF",
    "model_HRV_HF_rel_LFHF",
    "model_HRV_SD1SD2",
    "model_HRV_Symbolic_EqualProb4_0V",
    "model_HRV_Symbolic_EqualProb4_2UV",
]


MODEL_COLUMNS = BASE_MODEL_COLUMNS

ECG_NEUROKIT_WINDOW_DIM = len(MODEL_COLUMNS)  # 11
ECG_NEUROKIT_FEATURE_DIM = len(MODEL_COLUMNS) * 2  # mean + std = 22


# ============================================================================
# 5 分钟 HRV 提取核心
# ============================================================================

def _value(df: pd.DataFrame, column: str) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty or column not in df.columns:
        return np.nan
    v = df.iloc[0][column]
    return float(v) if pd.notna(v) else np.nan




def _artifact_count(artifacts: dict) -> int:
    if not isinstance(artifacts, dict):
        return 0
    count = 0
    for v in artifacts.values():
        try:
            count += len(v)
        except TypeError:
            pass
    return count


def _sd1sd2_from_rpeaks(rpeaks, sampling_rate) -> float:
    """Compute only the Poincare SD1/SD2 ratio used by the model.

    This is equivalent to NeuroKit's private Poincare calculation inside
    ``nk.hrv_nonlinear``.  Calling the public function also calculates DFA,
    several entropy families, fractal dimensions and other unused metrics.
    """
    rpeaks = np.asarray(rpeaks, dtype=float).reshape(-1)
    if len(rpeaks) < 3 or sampling_rate <= 0:
        return np.nan

    rri = np.diff(rpeaks) / float(sampling_rate) * 1000.0
    if len(rri) < 2 or not np.all(np.isfinite(rri)):
        return np.nan

    rri_n = rri[:-1]
    rri_plus = rri[1:]
    x1 = (rri_n - rri_plus) / np.sqrt(2.0)
    x2 = (rri_n + rri_plus) / np.sqrt(2.0)
    sd1 = float(np.std(x1, ddof=1))
    sd2 = float(np.std(x2, ddof=1))
    return sd1 / sd2 if sd2 > 0 else np.nan


def extract_5min_hrv(ecg_1d, sampling_rate, apply_artifact_correction=True):
    """
    从 5 分钟 ECG 波形提取 11 维 HRV 特征。

    Returns
    -------
    (features_36, qc_dict) or (None, None)
    """
    ecg_1d = np.asarray(ecg_1d, dtype=float).reshape(-1)
    if len(ecg_1d) < int(sampling_rate * 30):
        return None, None

    warnings.filterwarnings("ignore")

    # A malformed or near-flat ECG window must not invalidate the full record.
    # NeuroKit may raise before it can return an empty peak vector, so treat the
    # window as unavailable and let the caller keep the record without ECG HRV.
    try:
        ecg_clean = nk.ecg_clean(ecg_1d, sampling_rate=sampling_rate, method="neurokit")
        _, peak_info = nk.ecg_peaks(
            ecg_clean, sampling_rate=sampling_rate, method="neurokit",
            correct_artifacts=False, show=False,
        )
    except Exception as exc:
        logger.debug("ECG clean/peak detection failed: %s", exc)
        return None, None
    rpeaks_raw = peak_info["ECG_R_Peaks"]
    rpeaks_raw = np.unique(np.asarray(rpeaks_raw, dtype=int))
    rpeaks_raw.sort()

    if len(rpeaks_raw) < 30:
        return None, None

    # ---- Kubios 风格间期校正 ----
    rpeaks_used = rpeaks_raw.copy()
    if apply_artifact_correction:
        try:
            _, rpeaks_used = nk.signal_fixpeaks(
                rpeaks_raw, sampling_rate=sampling_rate,
                method="Kubios", iterative=True, show=False,
            )
            rpeaks_used = np.unique(np.asarray(rpeaks_used, dtype=int))
            rpeaks_used.sort()
        except Exception:
            rpeaks_used = rpeaks_raw.copy()

    if len(rpeaks_used) < 30:
        return None, None

    # ---- 三域 HRV ----
    try:
        hrv_time = nk.hrv_time(rpeaks_used, sampling_rate=sampling_rate, show=False)
    except (IndexError, Exception):
        return None, None
    try:
        hrv_freq = nk.hrv_frequency(
            rpeaks_used, sampling_rate=sampling_rate,
            psd_method="welch", interpolation_rate=4,
            normalize=False, show=False, silent=True,
        )
    except Exception:
        hrv_freq = pd.DataFrame([{}])
    try:
        hrv_sd1sd2 = _sd1sd2_from_rpeaks(rpeaks_used, sampling_rate)
    except Exception:
        hrv_sd1sd2 = np.nan

    # 符号动力学
    try:
        hrv_symbolic = nk.hrv_symbolic(
            rpeaks_used, sampling_rate=sampling_rate,
            quantization_level_equal_prob=(4,),
            quantization_level_max_min=(),
            sigma_rate=(),
        )
    except Exception:
        hrv_symbolic = pd.DataFrame([{}])

    # ---- 频域派生 ----
    lf = _value(hrv_freq, "HRV_LF")
    hf = _value(hrv_freq, "HRV_HF")

    base_features = [
        _value(hrv_time, "HRV_MedianNN"),
        _value(hrv_time, "HRV_MCVNN"),
        _value(hrv_time, "HRV_CVNN"),
        _value(hrv_time, "HRV_CVSD"),
        _value(hrv_time, "HRV_pNN20"),
        np.log(lf + EPS) if np.isfinite(lf) else np.nan,
        np.log(hf + EPS) if np.isfinite(hf) else np.nan,
        hf / (lf + hf + EPS) if np.isfinite(lf) and np.isfinite(hf) else np.nan,
        hrv_sd1sd2,
        _value(hrv_symbolic, "HRV_Symbolic_EqualProb4_0V"),
        _value(hrv_symbolic, "HRV_Symbolic_EqualProb4_2UV"),
    ]
    features = np.asarray(base_features, dtype=np.float32)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0), None


# ============================================================================
# ECGNeurokitMixin
# ============================================================================

class ECGNeurokitMixin:
    """
    使用 neurokit2 + Kubios 校正从 ECG 提取 HRV 特征 (5 分钟分段)。
    """

    def extract_ecg_neurokit(self, ecg_sig, fs):
        """
        从整夜 ECG 提取聚合 HRV 特征 (22 维)。

        Returns
        -------
        features : (22,) ndarray
            11 特征 × 2 统计量 (mean, std)，跨 5 分钟窗口聚合。
        """
        sig = np.asarray(ecg_sig, dtype=float).reshape(-1)
        if np.isnan(sig).any():
            sig = np.nan_to_num(sig, nan=0.0)

        if len(sig) < int(fs * 30):
            logger.debug("ECG signal too short (%d samples @ %dHz), returning zeros", len(sig), fs)
            return np.zeros(ECG_NEUROKIT_FEATURE_DIM, dtype=np.float32)

        window_samples = int(WINDOW_SEC * fs)
        n_windows = max(1, len(sig) // window_samples)

        all_features = []
        for i in range(n_windows):
            start = i * window_samples
            end = min(start + window_samples, len(sig))
            seg = sig[start:end]

            if len(seg) < int(fs * 60):
                continue

            f11, _ = extract_5min_hrv(seg, fs)
            if f11 is not None:
                all_features.append(f11)

        if len(all_features) == 0:
            logger.debug("ECG: 0 valid 5-min windows extracted")
            return np.zeros(ECG_NEUROKIT_FEATURE_DIM, dtype=np.float32)

        logger.debug("ECG: %d valid 5-min windows extracted", len(all_features))

        all_features = np.stack(all_features, axis=0)  # (N_win, 11)
        win_mean = np.mean(all_features, axis=0)
        win_std = np.std(all_features, axis=0)

        return np.concatenate([win_mean, win_std]).astype(np.float32)

    @staticmethod
    def ecg_neurokit_feature_names():
        names = []
        for stat in ["mean", "std"]:
            for col in MODEL_COLUMNS:
                names.append(f"ecgnk_{col}_{stat}")
        return names
