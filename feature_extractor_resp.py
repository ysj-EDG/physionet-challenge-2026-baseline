#!/usr/bin/env python
"""
呼吸特征提取器 — 基于 Hilbert 包络 + 局部基线 + 30s epoch 分析。

管道路径:
    气流/胸带/腹带 → 25Hz重采样 → 0.05-1Hz带通 → Hilbert包络
    → 120s滚动80%分位数基线 → 30s分段 → 周期检测+幅度特征
    → 胸腹联合(相关/时滞/反常运动) → 跨epoch mean+std → 28维
"""

import logging
import numpy as np
from fractions import Fraction
from scipy.signal import (
    butter, correlate, detrend, find_peaks, hilbert,
    resample_poly, sosfiltfilt,
)

logger = logging.getLogger("resp")

EPS = 1e-12

# 14 个模型特征 (per epoch)
MODEL_COLS = [
    "airflow_rate_bpm",
    "airflow_cycle_cv",
    "airflow_amp_local_norm",
    "airflow_amp_iqr_over_median",
    "airflow_reduction30_fraction",
    "airflow_near_absent_fraction",
    "airflow_longest_reduction_sec",
    "thorax_amp_local_norm",
    "thorax_reduction30_fraction",
    "abdomen_amp_local_norm",
    "abdomen_reduction30_fraction",
    "thorax_abd_corr",
    "thorax_abd_lag_sec",
    "thorax_abd_paradox_fraction",
]

RESP_FEATURE_DIM = len(MODEL_COLS) * 2  # mean + std = 28


# ============================================================
# 基础函数
# ============================================================

def _fill_nan_linear(x):
    x = np.asarray(x, dtype=float).copy()
    nan_mask = ~np.isfinite(x)
    if not nan_mask.any():
        return x
    valid_mask = ~nan_mask
    if valid_mask.sum() < 10:
        raise ValueError("呼吸信号有效采样点过少。")
    idx = np.arange(len(x))
    x[nan_mask] = np.interp(idx[nan_mask], idx[valid_mask], x[valid_mask])
    return x


def _resample_to(x, old_fs, new_fs):
    x = np.asarray(x, dtype=float)
    if np.isclose(old_fs, new_fs):
        return x
    ratio = Fraction(float(new_fs) / float(old_fs)).limit_denominator(1000)
    return resample_poly(x, up=ratio.numerator, down=ratio.denominator)


def _bandpass(x, fs, lowcut=0.05, highcut=1.0, order=4):
    nyquist = fs / 2.0
    highcut = min(highcut, 0.45 * fs)
    sos = butter(order, [lowcut / nyquist, highcut / nyquist],
                 btype="bandpass", output="sos")
    return sosfiltfilt(sos, x)


def _lowpass(x, fs, highcut=0.4, order=3):
    nyquist = fs / 2.0
    highcut = min(highcut, 0.45 * fs)
    sos = butter(order, highcut / nyquist, btype="lowpass", output="sos")
    return sosfiltfilt(sos, x)


def _rolling_quantile(x, fs, window_sec=120, q=0.80):
    import pandas as pd
    window_samples = max(3, int(round(window_sec * fs)))
    baseline = (
        pd.Series(x)
        .rolling(window=window_samples, center=True,
                 min_periods=max(10, window_samples // 4))
        .quantile(q).bfill().ffill().to_numpy()
    )
    positive = x[x > 0]
    floor = np.quantile(positive, 0.05) if len(positive) > 0 else EPS
    return np.maximum(baseline, max(floor, EPS))


def _find_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts, ends))


def _longest_run_sec(mask, fs):
    runs = _find_runs(mask)
    if len(runs) == 0:
        return 0.0
    return float(max((end - start) / fs for start, end in runs))


def _safe_cv(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    return float(np.std(values, ddof=1) / (np.mean(values) + EPS))


def _safe_corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 5 or np.std(x) < EPS or np.std(y) < EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _lag_at_max_corr(x, y, fs, max_lag_sec=2.0):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 5 or np.std(x) < EPS or np.std(y) < EPS:
        return np.nan
    x = (x - np.mean(x)) / (np.std(x) + EPS)
    y = (y - np.mean(y)) / (np.std(y) + EPS)
    corr = correlate(x, y, mode="full", method="auto")
    lags = np.arange(-len(y) + 1, len(x))
    max_lag_samples = int(round(max_lag_sec * fs))
    keep = np.abs(lags) <= max_lag_samples
    best_idx = np.argmax(corr[keep])
    return float(lags[keep][best_idx] / fs)


# ============================================================
# 整夜预处理: 单通道
# ============================================================

def _preprocess_resp_channel(signal, fs, target_fs=25):
    signal = _fill_nan_linear(signal)
    signal = _resample_to(signal, fs, target_fs)
    signal = detrend(signal, type="constant")
    clean = _bandpass(signal, target_fs, 0.05, 1.0)
    envelope = np.abs(hilbert(clean))
    envelope = _lowpass(envelope, target_fs, 0.4)
    envelope = np.maximum(envelope, 0)
    baseline = _rolling_quantile(envelope, target_fs, 120, 0.80)
    relative_amp = envelope / (baseline + EPS)
    robust_scale = np.median(np.abs(clean - np.median(clean)))
    peaks, _ = find_peaks(
        clean,
        distance=max(1, int(round(1.2 * target_fs))),
        prominence=max(0.10 * robust_scale, EPS),
    )
    intervals_sec = np.diff(peaks) / target_fs
    interval_centers = ((peaks[:-1] + peaks[1:]) // 2).astype(int)
    valid = (intervals_sec >= 1.5) & (intervals_sec <= 12.0)
    intervals_sec = intervals_sec[valid]
    interval_centers = interval_centers[valid]
    reduction30_mask = relative_amp < 0.70
    global_lowflow_runs = [
        (s, e) for s, e in _find_runs(reduction30_mask)
        if (e - s) / target_fs >= 10.0
    ]
    return {
        "clean": clean, "envelope": envelope, "baseline": baseline,
        "relative_amp": relative_amp, "peaks": peaks,
        "intervals_sec": intervals_sec, "interval_centers": interval_centers,
        "reduction30_mask": reduction30_mask,
        "global_lowflow_runs": global_lowflow_runs,
    }


# ============================================================
# 单 epoch 提取
# ============================================================

def _extract_resp_epoch(proc, start_s, end_s, fs, prefix, global_runs):
    rel_amp = proc["relative_amp"][start_s:end_s]
    interval_mask = (proc["interval_centers"] >= start_s) & (proc["interval_centers"] < end_s)
    intervals = proc["intervals_sec"][interval_mask]
    rate_bpm = 60.0 / np.median(intervals) if len(intervals) > 0 else np.nan
    candidate_overlap = sum(
        rs < end_s and re > start_s for rs, re in global_runs
    )
    amp_iqr = np.percentile(rel_amp, 75) - np.percentile(rel_amp, 25)
    return np.array([
        float(rate_bpm),
        _safe_cv(intervals),
        float(np.median(rel_amp)),
        float(amp_iqr / (np.median(rel_amp) + EPS)),
        float(np.mean(rel_amp < 0.70)),
        float(np.mean(rel_amp < 0.10)),
        _longest_run_sec(rel_amp < 0.70, fs),
        int(candidate_overlap),
    ], dtype=np.float32)


def _extract_thorax_abd_epoch(thorax_proc, abdomen_proc, start_s, end_s, fs):
    thorax = thorax_proc["clean"][start_s:end_s]
    abdomen = abdomen_proc["clean"][start_s:end_s]
    mini_samples = int(round(5.0 * fs))
    mini_corrs = []
    for st in range(0, len(thorax), mini_samples):
        ed = min(st + mini_samples, len(thorax))
        c = _safe_corr(thorax[st:ed], abdomen[st:ed])
        if np.isfinite(c):
            mini_corrs.append(c)
    paradox = float(np.mean(np.asarray(mini_corrs) < -0.25)) if mini_corrs else np.nan
    thorax_rel = thorax_proc["relative_amp"][start_s:end_s]
    abdomen_rel = abdomen_proc["relative_amp"][start_s:end_s]
    return np.array([
        _safe_corr(thorax, abdomen),
        _lag_at_max_corr(thorax, abdomen, fs, 2.0),
        paradox,
        float(np.log(np.median(thorax_rel) + EPS) - np.log(np.median(abdomen_rel) + EPS)),
    ], dtype=np.float32)


# ============================================================
# EMGMixin — 集成到 FeatureExtractor
# ============================================================

class RespMixin:
    """
    呼吸特征提取器 (28 维): 气流(7) + 胸带(2) + 腹带(2) + 胸腹联合(3), mean+std.
    """

    def extract_resp_features(self, phys_data, phys_fs):
        """从原始通道字典提取呼吸特征 (28 维)."""
        def _pick(candidates):
            for c in candidates:
                if c in phys_data and phys_data[c] is not None and len(phys_data[c]) > 1:
                    return phys_data[c], float(phys_fs.get(c, 200.0))
            c_lower = {k.lower().strip(): k for k in phys_data}
            for c in candidates:
                key = c_lower.get(c.lower().strip())
                if key and phys_data[key] is not None and len(phys_data[key]) > 1:
                    return phys_data[key], float(phys_fs.get(key, 200.0))
            return None, None

        airflow_sig, airflow_fs = _pick([
            'airflow', 'flow', 'ptaf', 'nasal', 'cannula', 'npt', 'nasal_pressure',
        ])
        thorax_sig, thorax_fs = _pick([
            'chest', 'thorax', 'thor', 'thoracic',
        ])
        abdomen_sig, abdomen_fs = _pick([
            'abdomen', 'abd', 'abdominal', 'abdo',
        ])

        logger.debug("Resp channels found: airflow=%s, thorax=%s, abdomen=%s",
                     airflow_sig is not None, thorax_sig is not None, abdomen_sig is not None)

        # 预处理各通道
        processed = {}
        if airflow_sig is not None:
            processed["airflow"] = _preprocess_resp_channel(airflow_sig, airflow_fs)
        if thorax_sig is not None:
            processed["thorax"] = _preprocess_resp_channel(thorax_sig, thorax_fs)
        if abdomen_sig is not None:
            processed["abdomen"] = _preprocess_resp_channel(abdomen_sig, abdomen_fs)

        if not processed:
            return np.zeros(RESP_FEATURE_DIM, dtype=np.float32)

        # 胸腹方向校正
        if "thorax" in processed and "abdomen" in processed:
            whole_corr = _safe_corr(
                processed["thorax"]["clean"], processed["abdomen"]["clean"])
            if np.isfinite(whole_corr) and whole_corr < 0:
                processed["abdomen"]["clean"] *= -1

        target_fs = 25
        epoch_samples = int(round(30 * target_fs))
        n_epochs = min(len(p["clean"]) // epoch_samples for p in processed.values())

        all_feats = []
        for ep in range(n_epochs):
            start_s = ep * epoch_samples
            end_s = start_s + epoch_samples
            row = []

            for ch in ["airflow", "thorax", "abdomen"]:
                if ch in processed:
                    p = processed[ch]
                    f8 = _extract_resp_epoch(p, start_s, end_s, target_fs, ch,
                                             p.get("global_lowflow_runs", []))
                    if ch == "airflow":
                        row.extend(f8[:7])   # 7 model features
                    else:
                        row.extend([f8[2], f8[4]])  # amp_local_norm, reduction30_fraction
                else:
                    if ch == "airflow":
                        row.extend([0.0] * 7)
                    else:
                        row.extend([0.0, 0.0])

            if "thorax" in processed and "abdomen" in processed:
                ta = _extract_thorax_abd_epoch(
                    processed["thorax"], processed["abdomen"],
                    start_s, end_s, target_fs)
                row.extend(ta[:3])  # corr, lag, paradox
            else:
                row.extend([0.0, 0.0, 0.0])

            all_feats.append(row)

        if not all_feats:
            return np.zeros(RESP_FEATURE_DIM, dtype=np.float32)

        all_feats = np.asarray(all_feats, dtype=np.float32)
        all_feats = np.nan_to_num(all_feats, nan=0.0)
        feat_mean = np.mean(all_feats, axis=0)
        feat_std = np.std(all_feats, axis=0)
        return np.concatenate([feat_mean, feat_std]).astype(np.float32)

    @staticmethod
    def resp_feature_names():
        names = []
        for stat in ["mean", "std"]:
            for c in MODEL_COLS:
                names.append(f"resp_{c}_{stat}")
        return names
