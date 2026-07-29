#!/usr/bin/env python
"""
EMG 特征提取器 — 基于全夜包络基线 + 30s epoch burst/tonic 分析。

管道路径:
    原始 EMG → NaN 插值与去中心 → 10–90 Hz 带通 → 整流与低通包络
    → 全夜包络 20% 分位数基线 → 30s 分段 → burst/tonic + 频谱特征
    → 按全夜或睡眠阶段跨 epoch 聚合

输出:
    features: (42,) — chin overall/REM/Wake (24)
    + left leg overall (8) + right leg overall (8)
    + 双侧腿相关性与不对称性 (2)

通道缺失:
    缺失或不足 30 秒的通道使用对应维度的零值；双侧腿任一通道缺失时，
    相关性与不对称性均置零。
"""

import logging
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch

logger = logging.getLogger("emg")

EPS = 1e-12

# ============================================================================
# 基础工具
# ============================================================================

def _fill_nan_linear(x):
    """
    对 EMG 中的 NaN 和无穷值进行线性插值。

    有效采样点少于 10 个时抛出 ValueError，避免由过少数据构造信号。
    """
    x = np.asarray(x, dtype=float).copy()
    nan_mask = ~np.isfinite(x)
    if not nan_mask.any():
        return x
    valid = ~nan_mask
    if valid.sum() < 10:
        raise ValueError("EMG 有效采样点过少，无法插值。")
    idx = np.arange(len(x))
    x[nan_mask] = np.interp(idx[nan_mask], idx[valid], x[valid])
    return x


def _butter_filter(x, fs, lowcut=None, highcut=None, order=4):
    """使用零相位 Butterworth 滤波器执行带通、高通或低通滤波。"""
    nyq = fs / 2.0
    if lowcut is not None and highcut is not None:
        sos = butter(order, [lowcut / nyq, highcut / nyq], btype="bandpass", output="sos")
    elif lowcut is not None:
        sos = butter(order, lowcut / nyq, btype="highpass", output="sos")
    elif highcut is not None:
        sos = butter(order, highcut / nyq, btype="lowpass", output="sos")
    else:
        return x.copy()
    return sosfiltfilt(sos, x)


def _apply_notch(x, fs, line_freq=None, quality_factor=30.0):
    """对指定工频执行零相位陷波；频率不可用时返回信号副本。"""
    if line_freq is None:
        return x.copy()
    nyq = fs / 2.0
    if line_freq >= nyq * 0.95:
        return x.copy()
    b, a = iirnotch(w0=line_freq / nyq, Q=quality_factor)
    return filtfilt(b, a, x)


def _find_runs(mask):
    """查找布尔掩码中连续 True 区间，返回左闭右开的索引对。"""
    mask = np.asarray(mask, dtype=bool)
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return list(zip(starts, ends))


def _merge_runs(runs, max_gap_samples):
    """合并间隔不超过指定采样点数的相邻活动区间。"""
    if len(runs) == 0:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= max_gap_samples:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [tuple(x) for x in merged]


def _select_duration_runs(runs, fs, min_sec, max_sec):
    """保留持续时间位于闭区间 ``[min_sec, max_sec]`` 内的活动段。"""
    return [(s, e) for s, e in runs if min_sec <= (e - s) / fs <= max_sec]


def _bandpower(signal, fs, f_low, f_high):
    """使用 Welch 功率谱计算指定闭合频带内的积分功率。"""
    signal = np.asarray(signal, dtype=float)
    if len(signal) < int(fs * 2):
        return np.nan
    nperseg = min(len(signal), int(fs * 4))
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg,
                       noverlap=nperseg // 2, detrend="constant", scaling="density")
    keep = (freqs >= f_low) & (freqs <= f_high)
    if keep.sum() < 2:
        return np.nan
    return float(np.trapz(psd[keep], freqs[keep]))


# ============================================================================
# 全夜预处理
# ============================================================================

def _preprocess_emg(emg, fs, lowcut=10.0, highcut=90.0, baseline_quantile=0.20):
    """
    对单通道全夜 EMG 进行滤波、整流和包络基线估计。

    返回带通信号、非负低通包络，以及包络指定分位数对应的全夜基线。
    高频截止频率根据采样率自适应限制为 ``0.45 * fs``。
    """
    emg = _fill_nan_linear(emg)
    emg_centered = emg - np.median(emg)
    usable_highcut = min(highcut, 0.45 * fs)
    emg_filtered = _butter_filter(emg_centered, fs, lowcut=lowcut,
                                  highcut=usable_highcut, order=4)
    emg_rectified = np.abs(emg_filtered)
    envelope_cutoff = min(5.0, 0.20 * fs)
    emg_envelope = _butter_filter(emg_rectified, fs, highcut=envelope_cutoff, order=4)
    emg_envelope = np.maximum(emg_envelope, 0)
    baseline = float(np.quantile(emg_envelope, baseline_quantile))
    if baseline <= EPS:
        baseline = float(np.median(emg_envelope) + EPS)
    return emg_filtered, emg_envelope, baseline


# ============================================================================
# 单 epoch 特征提取
# ============================================================================

def _extract_emg_epoch(emg_filt, envelope, fs, baseline, mode="chin",
                       activity_mult=2.0, merge_gap_sec=0.10):
    """
    从一个 30 秒 EMG epoch 提取 8 维活动、爆发和频谱特征。

    下颏模式保留 0.10–5.00 秒的爆发，腿部模式保留 0.50–10.00 秒
    的爆发；间隔不超过 ``merge_gap_sec`` 的活动段会先合并。

    Returns
    -------
    features : (8,) ndarray
        依次为归一化 log RMS、包络 IQR、活动比例、tonic 比例、
        每分钟 burst 数、burst 占空比、3 秒 mini-epoch phasic 比例，
        以及 20 Hz 上下频带功率比的对数。
    """
    envelope = np.asarray(envelope, dtype=float)
    emg_filt = np.asarray(emg_filt, dtype=float)
    duration_sec = len(envelope) / fs

    threshold = baseline * activity_mult
    activity_mask = envelope > threshold
    runs = _find_runs(activity_mask)
    runs = _merge_runs(runs, max_gap_samples=int(round(merge_gap_sec * fs)))

    if mode == "chin":
        bursts = _select_duration_runs(runs, fs, 0.10, 5.00)
    else:
        bursts = _select_duration_runs(runs, fs, 0.50, 10.00)

    tonic_runs = [(s, e) for s, e in runs if (e - s) / fs >= 5.0]

    burst_durations = np.array([(e - s) / fs for s, e in bursts], dtype=float)
    tonic_sec = float(sum((e - s) / fs for s, e in tonic_runs))
    burst_sec = float(burst_durations.sum())

    # ---- 频谱功率比 ----
    low_power = _bandpower(emg_filt, fs, 10.0, 20.0)
    high_upper = min(55.0, 0.40 * fs)
    high_power = _bandpower(emg_filt, fs, 20.0, high_upper)
    log_hf_lf = float(np.log((high_power + EPS) / (low_power + EPS))
                      if low_power > 0 else 0.0)

    # ---- 3 秒 mini-epoch phasic 比例（下颏和双侧腿统一计算）----
    phasic_fraction = np.nan
    if abs(duration_sec - 30.0) < 1.0:
        mini_samples = int(round(3.0 * fs))
        active_mini = 0
        for mi in range(10):
            ms, me = mi * mini_samples, min((mi + 1) * mini_samples, len(envelope))
            if any(s < me and e > ms for s, e in bursts):
                active_mini += 1
        phasic_fraction = active_mini / 10.0

    rms = float(np.sqrt(np.mean(emg_filt ** 2)))
    env_median = float(np.median(envelope))
    env_iqr = float(np.percentile(envelope, 75) - np.percentile(envelope, 25))

    return np.array([
        float(np.log((rms + EPS) / (baseline + EPS))),   # log_rms_norm
        float(env_iqr / (env_median + EPS)),              # envelope_iqr_norm
        float(np.mean(activity_mask)),                    # active_fraction
        float(tonic_sec / duration_sec),                   # tonic_fraction
        float(len(bursts) / (duration_sec / 60.0)),        # burst_rate_per_min
        float(burst_sec / duration_sec),                   # burst_duty_cycle
        float(phasic_fraction) if np.isfinite(phasic_fraction) else 0.0,
        float(log_hf_lf),                                  # log_hf_lf_ratio
    ], dtype=np.float32)


# ============================================================================
# 全夜提取 + 聚合
# ============================================================================

def _extract_emg_channel(emg_sig, fs, stages, mode="chin"):
    """
    提取单通道全夜 EMG，并按 epoch 聚合 8 维特征。

    Parameters
    ----------
    emg_sig : ndarray
        单通道连续 EMG 信号。
    fs : float
        采样率（Hz）。
    stages : array-like or None
        睡眠分期数组（1=N3、2=N2、3=N1、4=REM、5=Wake）。
    mode : {"chin", "leg"}, default="chin"
        控制 burst 持续时间筛选范围。

    Returns
    -------
    overall, rem_mean, wake_mean : tuple of (8,) ndarray
        全部、REM 和 Wake epoch 的均值；阶段无有效 epoch 时回退到全夜均值。
    """
    sig = _fill_nan_linear(emg_sig)
    emg_filt, envelope, baseline = _preprocess_emg(sig, fs)

    epoch_samples = int(round(30 * fs))
    n_epochs = len(sig) // epoch_samples
    if n_epochs < 1:
        return np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)

    all_feat = []
    wake_feat = []
    rem_feat = []
    nrem_feat = []

    for ep in range(n_epochs):
        start = ep * epoch_samples
        f_epoch = emg_filt[start:start + epoch_samples]
        e_epoch = envelope[start:start + epoch_samples]
        try:
            f8 = _extract_emg_epoch(f_epoch, e_epoch, fs, baseline, mode=mode)
        except Exception:
            continue

        all_feat.append(f8)
        if stages is not None and ep < len(stages):
            s = int(stages[ep])
            if s == 5:
                wake_feat.append(f8)
            elif s == 4:
                rem_feat.append(f8)
            elif s in (1, 2, 3):
                nrem_feat.append(f8)

    if not all_feat:
        return np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)

    overall = np.mean(all_feat, axis=0)
    rem_mean = np.mean(rem_feat, axis=0) if rem_feat else overall
    wake_mean = np.mean(wake_feat, axis=0) if wake_feat else overall
    return overall, rem_mean, wake_mean


EMG_FEATURE_NAMES = [
    "log_rms_norm", "envelope_iqr_norm", "active_fraction", "tonic_fraction",
    "burst_rate_per_min", "burst_duty_cycle", "phasic_mini_epoch_fraction", "log_hf_lf_ratio",
]

# 总维度: chin overall/REM/Wake (24) + lleg (8) + rleg (8)
#          + leg correlation/asymmetry (2) = 42。
EMG_FEATURE_DIM = 8 * 3 + 8 + 8 + 2  # 42


class EMGMixin:
    """
    从下颏与双侧胫前肌 EMG 提取全夜及分睡眠阶段特征。

    单通道活动阈值基于全夜包络基线，最终输出 42 维固定长度向量。
    """

    # ========================================================================
    # 公有 API
    # ========================================================================

    def extract_emg_features(self, phys_data, phys_fs, stages=None):
        """
        从原始通道字典提取 EMG 特征 (42 维)。

        Parameters
        ----------
        phys_data : dict {channel_label: signal}
            原始生理信号字典。
        phys_fs : dict {channel_label: fs}
            各通道对应的采样率字典。
        stages : ndarray or None
            CAISR 睡眠分期（1=N3、2=N2、3=N1、4=REM、5=Wake）。

        Returns
        -------
        features : (42,) ndarray
            chin (24) + left leg (8) + right leg (8) + bilateral (2)。
        """
        from scipy.stats import pearsonr

        def _pick_channel(candidates):
            for c in candidates:
                if c in phys_data and phys_data[c] is not None and len(phys_data[c]) > 1:
                    return phys_data[c], float(phys_fs.get(c, 200.0))
            # 回退：对去除首尾空格并转为小写的通道名进行匹配。
            c_lower = {k.lower().strip(): k for k in phys_data}
            for c in candidates:
                key = c_lower.get(c)
                if key and phys_data[key] is not None and len(phys_data[key]) > 1:
                    return phys_data[key], float(phys_fs.get(key, 200.0))
            return None, None

        chin_sig, chin_fs = _pick_channel([
            'chin1-chin2', 'chin', 'chin1', 'emg', 'chin_emg', 'chin emg',
        ])
        lleg_sig, lleg_fs = _pick_channel([
            'lat', 'lleg', 'l leg', 'leg1', 'left_leg', 'plml',
        ])
        rleg_sig, rleg_fs = _pick_channel([
            'rat', 'rleg', 'r leg', 'leg2', 'right_leg', 'plmr',
        ])

        logger.debug("EMG channels found: chin=%s, lleg=%s, rleg=%s",
                     chin_sig is not None, lleg_sig is not None, rleg_sig is not None)

        features = []

        for name, (sig, fs) in [("chin", (chin_sig, chin_fs)),
                                 ("lleg", (lleg_sig, lleg_fs)),
                                 ("rleg", (rleg_sig, rleg_fs))]:
            if sig is None or len(sig) < int(fs * 30):
                if name == "chin":
                    features.extend([0.0] * 24)
                else:
                    features.extend([0.0] * 8)
                continue

            mode = "chin" if name == "chin" else "leg"
            overall, rem_mean, wake_mean = _extract_emg_channel(sig, fs, stages, mode=mode)

            if name == "chin":
                features.extend(overall.tolist())
                features.extend(rem_mean.tolist())
                features.extend(wake_mean.tolist())
            else:
                features.extend(overall.tolist())

        # ---- 双侧腿 EMG 的 epoch 功率相关性与归一化不对称性 ----
        if lleg_sig is not None and rleg_sig is not None:
            l_sig, l_fs = lleg_sig, lleg_fs
            r_sig, r_fs = rleg_sig, rleg_fs
            l_power = _emg_epoch_power_series(l_sig, l_fs)
            r_power = _emg_epoch_power_series(r_sig, r_fs)
            n = min(len(l_power), len(r_power))
            if n >= 5:
                corr = float(pearsonr(l_power[:n], r_power[:n])[0])
                asym = float(np.mean(np.abs(l_power[:n] - r_power[:n])) /
                             (np.mean(l_power[:n] + r_power[:n]) + EPS))
            else:
                corr, asym = 0.0, 0.0
        else:
            corr, asym = 0.0, 0.0
        features.extend([corr, asym])

        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def emg_feature_names():
        """返回与 42 维 EMG 输出顺序一致的特征名称列表。"""
        names = []
        for ch in ["chin", "lleg", "rleg"]:
            if ch == "chin":
                for agg in ["overall", "rem", "wake"]:
                    for f in EMG_FEATURE_NAMES:
                        names.append(f"emg_{ch}_{f}_{agg}")
            else:
                for f in EMG_FEATURE_NAMES:
                    names.append(f"emg_{ch}_{f}_overall")
        names.append("emg_leg_correlation")
        names.append("emg_leg_asymmetry")
        return names


# ============================================================================
# 双侧腿辅助特征
# ============================================================================

def _emg_epoch_power_series(emg_sig, fs):
    """
    计算整夜 EMG 的逐 30 秒 epoch 方差序列，用于双侧腿相关性分析。

    仅使用完整的 30 秒 epoch。
    """
    sig = _fill_nan_linear(emg_sig)
    epoch_samples = int(round(30 * fs))
    n_epochs = len(sig) // epoch_samples
    powers = []
    for ep in range(n_epochs):
        seg = sig[ep * epoch_samples:(ep + 1) * epoch_samples]
        powers.append(float(np.var(seg)))
    return np.array(powers, dtype=float)
