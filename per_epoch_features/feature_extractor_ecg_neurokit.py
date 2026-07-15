#!/usr/bin/env python
"""
ECG HRV 特征提取器 — 基于 neurokit2 + Kubios 风格间期校正 + 5 分钟分段。

管道路径:
    原始 ECG → 5min 分段 → ecg_clean → ecg_peaks
    → signal_fixpeaks(Kubios) → hrv_time/freq/nonlinear/symbolic
    → 每段 36 维 → 跨段 mean+std → 72 维
"""

import logging
import warnings
import numpy as np
import pandas as pd

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

import neurokit2 as nk

try:
    from hrvanalysis.extract_features import (
        get_time_domain_features,
        get_frequency_domain_features,
        get_poincare_plot_features,
        get_sampen,
        get_csi_cvi_features,
    )
except Exception as exc:  # pragma: no cover - handled at runtime for cache jobs
    get_time_domain_features = None
    get_frequency_domain_features = None
    get_poincare_plot_features = None
    get_sampen = None
    get_csi_cvi_features = None
    _HRVANALYSIS_IMPORT_ERROR = exc
else:
    _HRVANALYSIS_IMPORT_ERROR = None

logger = logging.getLogger("ecg_neurokit")

EPS = 1e-12
WINDOW_SEC = 300  # 5 分钟
MIN_WINDOW_SEC = 270  # 最小窗口时长

# 36 个 5min-window ECG/HRV 特征: 原 11 维 + hrv-analysis 公式迁移 25 维。
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

HRV_ANALYSIS_COLUMNS = [
    "hrva_mean_nn",
    "hrva_sdnn",
    "hrva_rmssd",
    "hrva_sdsd",
    "hrva_nn50",
    "hrva_pnn50",
    "hrva_nn20",
    "hrva_mean_hr",
    "hrva_min_hr",
    "hrva_max_hr",
    "hrva_std_hr",
    "hrva_range_nn",
    "hrva_total_power",
    "hrva_vlf",
    "hrva_lf_hf_ratio",
    "hrva_lfnu",
    "hrva_hfnu",
    "hrva_sd1",
    "hrva_sd2",
    "hrva_sampen",
    "hrva_csi",
    "hrva_cvi",
    "hrva_modified_csi",
    "hrva_dfa_alpha1",
    "hrva_dfa_alpha2",
]

MODEL_COLUMNS = BASE_MODEL_COLUMNS + HRV_ANALYSIS_COLUMNS

ECG_NEUROKIT_WINDOW_DIM = len(MODEL_COLUMNS)  # 36
ECG_NEUROKIT_FEATURE_DIM = len(MODEL_COLUMNS) * 2  # mean + std = 72


# ============================================================================
# 5 分钟 HRV 提取核心
# ============================================================================

def _value(df: pd.DataFrame, column: str) -> float:
    if not isinstance(df, pd.DataFrame) or df.empty or column not in df.columns:
        return np.nan
    v = df.iloc[0][column]
    return float(v) if pd.notna(v) else np.nan


def _dict_value(d: dict, key: str) -> float:
    if not isinstance(d, dict):
        return np.nan
    v = d.get(key, np.nan)
    return float(v) if v is not None and np.isfinite(v) else np.nan


def _rpeaks_to_nn_ms(rpeaks, sampling_rate):
    rpeaks = np.asarray(rpeaks, dtype=float).ravel()
    if len(rpeaks) < 2:
        return np.array([], dtype=float)
    nn = np.diff(rpeaks) * 1000.0 / float(sampling_rate)
    nn = nn[np.isfinite(nn)]
    return nn[(nn >= 300.0) & (nn <= 2000.0)]


def _dfa_alpha(nn, scale_min, scale_max):
    nn = np.asarray(nn, dtype=float).ravel()
    n = len(nn)
    if n < max(scale_min * 2, scale_min + 2):
        return np.nan
    y = np.cumsum(nn - np.mean(nn))
    ns = np.arange(scale_min, min(scale_max + 1, n // 2 + 1))
    fluctuations = []
    scales = []
    for scale in ns:
        n_segments = n // scale
        if n_segments < 2:
            continue
        rms_vals = []
        for i in range(n_segments):
            seg = y[i * scale:(i + 1) * scale]
            x = np.arange(len(seg), dtype=float)
            try:
                coeff = np.polyfit(x, seg, 1)
            except Exception:
                continue
            trend = np.polyval(coeff, x)
            rms_vals.append(float(np.sqrt(np.mean((seg - trend) ** 2))))
        if rms_vals:
            f = float(np.mean(rms_vals))
            if f > 0:
                scales.append(scale)
                fluctuations.append(f)
    if len(scales) < 2:
        return np.nan
    return float(np.polyfit(np.log10(scales), np.log10(fluctuations), 1)[0])


def _extract_hrvanalysis_features(rpeaks, sampling_rate):
    features = {col: np.nan for col in HRV_ANALYSIS_COLUMNS}
    if _HRVANALYSIS_IMPORT_ERROR is not None:
        logger.debug("hrvanalysis unavailable: %s", _HRVANALYSIS_IMPORT_ERROR)
        return features

    nn = _rpeaks_to_nn_ms(rpeaks, sampling_rate)
    if len(nn) < 3:
        return features
    nn_list = nn.tolist()

    try:
        td = get_time_domain_features(nn_list)
    except Exception:
        td = {}
    features.update({
        "hrva_mean_nn": _dict_value(td, "mean_nni"),
        "hrva_sdnn": _dict_value(td, "sdnn"),
        "hrva_rmssd": _dict_value(td, "rmssd"),
        "hrva_sdsd": _dict_value(td, "sdsd"),
        "hrva_nn50": _dict_value(td, "nni_50"),
        "hrva_pnn50": _dict_value(td, "pnni_50"),
        "hrva_nn20": _dict_value(td, "nni_20"),
        "hrva_mean_hr": _dict_value(td, "mean_hr"),
        "hrva_min_hr": _dict_value(td, "min_hr"),
        "hrva_max_hr": _dict_value(td, "max_hr"),
        "hrva_std_hr": _dict_value(td, "std_hr"),
        "hrva_range_nn": _dict_value(td, "range_nni"),
    })

    try:
        fd = get_frequency_domain_features(nn_list, method="welch", sampling_frequency=4)
    except Exception:
        fd = {}
    features.update({
        "hrva_total_power": _dict_value(fd, "total_power"),
        "hrva_vlf": _dict_value(fd, "vlf"),
        "hrva_lf_hf_ratio": _dict_value(fd, "lf_hf_ratio"),
        "hrva_lfnu": _dict_value(fd, "lfnu"),
        "hrva_hfnu": _dict_value(fd, "hfnu"),
    })

    try:
        pc = get_poincare_plot_features(nn_list)
    except Exception:
        pc = {}
    features.update({
        "hrva_sd1": _dict_value(pc, "sd1"),
        "hrva_sd2": _dict_value(pc, "sd2"),
    })

    try:
        se = get_sampen(nn_list)
    except Exception:
        se = {}
    features["hrva_sampen"] = _dict_value(se, "sampen")

    try:
        csi = get_csi_cvi_features(nn_list)
    except Exception:
        csi = {}
    features.update({
        "hrva_csi": _dict_value(csi, "csi"),
        "hrva_cvi": _dict_value(csi, "cvi"),
        "hrva_modified_csi": _dict_value(csi, "Modified_csi"),
    })

    features["hrva_dfa_alpha1"] = _dfa_alpha(nn, 4, 16)
    features["hrva_dfa_alpha2"] = _dfa_alpha(nn, 16, 64)
    return features


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
    从 5 分钟 ECG 波形提取 36 维 HRV 特征。

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
    hrv_analysis = _extract_hrvanalysis_features(rpeaks_used, sampling_rate)
    features = np.array(
        base_features + [hrv_analysis[col] for col in HRV_ANALYSIS_COLUMNS],
        dtype=np.float32,
    )

    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features, None


# ============================================================================
# ECGNeurokitMixin
# ============================================================================

class ECGNeurokitMixin:
    """
    使用 neurokit2 + Kubios 校正从 ECG 提取 HRV 特征 (5 分钟分段)。
    """

    def extract_ecg_neurokit(self, ecg_sig, fs):
        """
        从整夜 ECG 提取聚合 HRV 特征 (72 维)。

        Returns
        -------
        features : (72,) ndarray
            36 特征 × 2 统计量 (mean, std)，跨 5 分钟窗口聚合。
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

            f36, _ = extract_5min_hrv(seg, fs)
            if f36 is not None:
                all_features.append(f36)

        if len(all_features) == 0:
            logger.debug("ECG: 0 valid 5-min windows extracted")
            return np.zeros(ECG_NEUROKIT_FEATURE_DIM, dtype=np.float32)

        logger.debug("ECG: %d valid 5-min windows extracted", len(all_features))

        all_features = np.stack(all_features, axis=0)  # (N_win, 36)
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
