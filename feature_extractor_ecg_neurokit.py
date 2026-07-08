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

# 11 个核心特征
MODEL_COLUMNS = [
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

ECG_NEUROKIT_FEATURE_DIM = len(MODEL_COLUMNS) * 2  # mean + std = 22


# ============================================================================
# 5 分钟 HRV 提取核心
# ============================================================================

def _value(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
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


def extract_5min_hrv(ecg_1d, sampling_rate, apply_artifact_correction=True):
    """
    从 5 分钟 ECG 波形提取 11 维 HRV 特征。

    Returns
    -------
    (features_11, qc_dict) or (None, None)
    """
    try:
        ecg_1d = np.asarray(ecg_1d, dtype=float).reshape(-1)
        if len(ecg_1d) < int(sampling_rate * 30):
            return None, None
        ecg_1d = np.nan_to_num(ecg_1d, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.any(np.isfinite(ecg_1d)) or np.nanstd(ecg_1d) < EPS:
            return None, None

        # ECG 清洗
        ecg_clean = nk.ecg_clean(ecg_1d, sampling_rate=sampling_rate, method="neurokit")
        ecg_clean = np.asarray(ecg_clean, dtype=float).reshape(-1)
        if len(ecg_clean) == 0 or np.nanstd(ecg_clean) < EPS:
            return None, None

        # R 峰检测. NeuroKit may raise IndexError on flat/noisy windows with no QRS candidates.
        _, peak_info = nk.ecg_peaks(
            ecg_clean, sampling_rate=sampling_rate, method="neurokit",
            correct_artifacts=False, show=False,
        )
        rpeaks_raw = peak_info.get("ECG_R_Peaks", [])
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
        except Exception:
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
            hrv_nonlinear = nk.hrv_nonlinear(rpeaks_used, sampling_rate=sampling_rate, show=False)
        except Exception:
            hrv_nonlinear = pd.DataFrame([{}])

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

        features = np.array([
            _value(hrv_time, "HRV_MedianNN"),
            _value(hrv_time, "HRV_MCVNN"),
            _value(hrv_time, "HRV_CVNN"),
            _value(hrv_time, "HRV_CVSD"),
            _value(hrv_time, "HRV_pNN20"),
            np.log(lf + EPS) if np.isfinite(lf) else np.nan,
            np.log(hf + EPS) if np.isfinite(hf) else np.nan,
            hf / (lf + hf + EPS) if np.isfinite(lf) and np.isfinite(hf) else np.nan,
            _value(hrv_nonlinear, "HRV_SD1SD2"),
            _value(hrv_symbolic, "HRV_Symbolic_EqualProb4_0V"),
            _value(hrv_symbolic, "HRV_Symbolic_EqualProb4_2UV"),
        ], dtype=np.float32)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features, None
    except Exception as exc:
        logger.debug("Skipping ECG HRV window after extraction failure: %s", exc)
        return None, None


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
