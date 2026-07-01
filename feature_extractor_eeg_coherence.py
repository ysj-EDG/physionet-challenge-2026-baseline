#!/usr/bin/env python
"""
EEG 相干特征提取器。

基于 ref/eeg_sleep_features.py 的 eeg_segment_coherence()，
从 15 对导联的幅度平方相干谱中提取 24 个标量特征，共 360 维。

管道路径:
    原始 EEG → 重采样 100Hz → 30s 分段 → PSD + 相干谱
    → 每导联对每 epoch 提取 24 特征 → 跨 epoch 取均值 → 360 维
"""

import os
import sys
import numpy as np

# 确保 ref 目录在 path 中
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REF_DIR = os.path.join(_SCRIPT_DIR, "ref")
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)

from eeg_sleep_features import eeg_segment_coherence, MAX_CHANNELS, N_PAIRS, N_FFT_BINS

# ============================================================================
# 频带定义 (bin 索引, 100 bins / 0-50Hz / 0.5Hz 分辨率)
# ============================================================================
# bin i → 频率 i * 0.5 Hz
BAND_BINS = {
    "delta": (1, 8),        # 0.5–4 Hz
    "theta": (8, 16),       # 4–8 Hz
    "alpha": (16, 26),      # 8–13 Hz
    "sigma": (22, 32),      # 11–16 Hz
    "beta":  (32, 60),      # 16–30 Hz
    "low_sigma":  (22, 27), # 11–13.5 Hz
    "high_sigma": (27, 32), # 13.5–16 Hz
    "low":   (1, 16),       # 0.5–8 Hz (delta+theta)
    "high":  (32, 60),      # 16–30 Hz (beta)
    "full":  (1, 60),       # 0.5–30 Hz
}

# 15 导联对名称
PAIR_NAMES = [
    "F3-F4", "F3-C3", "F3-C4", "F3-O1", "F3-O2",
    "F4-C3", "F4-C4", "F4-O1", "F4-O2",
    "C3-C4", "C3-O1", "C3-O2",
    "C4-O1", "C4-O2",
    "O1-O2",
]

# 每导联对特征数
FEATURES_PER_PAIR = 24
# 总维度
COHERENCE_FEATURE_DIM = N_PAIRS * FEATURES_PER_PAIR  # 360
SPECTRAL_FEATURE_DIM = MAX_CHANNELS * 9               # 54 (6通道 × 9 PSD特征)
EEG_FEATURE_DIM = SPECTRAL_FEATURE_DIM + COHERENCE_FEATURE_DIM  # 414


class EEGCoherenceMixin:
    """
    从 EEG 信号中提取导联间相干特征。

    依赖 ref/eeg_sleep_features.py 提供 eeg_segment_coherence()。
    """

    @staticmethod
    def _bin_mask(lo_bin, hi_bin):
        """生成频带布尔掩码 (长度 100)."""
        mask = np.zeros(N_FFT_BINS, dtype=bool)
        mask[lo_bin:hi_bin] = True
        return mask

    @staticmethod
    def _band_mean(coh_spectrum, lo_bin, hi_bin):
        """频带内平均相干度."""
        return float(np.mean(coh_spectrum[lo_bin:hi_bin]))

    @staticmethod
    def _band_auc(coh_spectrum, lo_bin, hi_bin):
        """频带内积分相干度 (梯形法则 AUC)."""
        vals = coh_spectrum[lo_bin:hi_bin]
        return float(np.trapz(vals, dx=0.5))

    @staticmethod
    def _band_iqr(coh_spectrum, lo_bin, hi_bin):
        """频带内相干度 IQR."""
        vals = coh_spectrum[lo_bin:hi_bin]
        if len(vals) < 2:
            return 0.0
        return float(np.quantile(vals, 0.75) - np.quantile(vals, 0.25))

    @staticmethod
    def _safe_div(num, den):
        return float(num) / float(den) if den > 0 else 0.0

    @classmethod
    def _extract_coh_features_from_spectrum(cls, coh):
        """
        从单个导联对的相干谱 (100 bins) 提取 24 维特征。

        Parameters
        ----------
        coh : (100,) ndarray — 幅度平方相干值

        Returns
        -------
        features : (24,) ndarray
        """
        b = BAND_BINS
        # --- 频带平均相干度 (5) ---
        f_mean_delta = cls._band_mean(coh, *b["delta"])
        f_mean_theta = cls._band_mean(coh, *b["theta"])
        f_mean_alpha = cls._band_mean(coh, *b["alpha"])
        f_mean_sigma = cls._band_mean(coh, *b["sigma"])
        f_mean_beta  = cls._band_mean(coh, *b["beta"])

        # --- 频带积分相干度 (5) ---
        f_auc_delta = cls._band_auc(coh, *b["delta"])
        f_auc_theta = cls._band_auc(coh, *b["theta"])
        f_auc_alpha = cls._band_auc(coh, *b["alpha"])
        f_auc_sigma = cls._band_auc(coh, *b["sigma"])
        f_auc_beta  = cls._band_auc(coh, *b["beta"])

        # --- 频带内稳定性 IQR (5) ---
        f_iqr_delta = cls._band_iqr(coh, *b["delta"])
        f_iqr_theta = cls._band_iqr(coh, *b["theta"])
        f_iqr_alpha = cls._band_iqr(coh, *b["alpha"])
        f_iqr_sigma = cls._band_iqr(coh, *b["sigma"])
        f_iqr_beta  = cls._band_iqr(coh, *b["beta"])

        # --- 比例 (4) ---
        f_sigma_delta_ratio = cls._safe_div(f_mean_sigma, f_mean_delta)
        f_alpha_delta_ratio = cls._safe_div(f_mean_alpha, f_mean_delta)
        f_beta_delta_ratio  = cls._safe_div(f_mean_beta, f_mean_delta)
        f_high_low_ratio    = cls._safe_div(f_mean_beta,
                                            cls._band_mean(coh, *b["low"]))

        # --- 全谱形状 (3) ---
        full_lo, full_hi = b["full"]
        freqs = np.arange(full_lo, full_hi) * 0.5  # 频率 (Hz)
        vals = coh[full_lo:full_hi]
        total = float(np.sum(vals))
        if total > 0:
            centroid = float(np.sum(freqs * vals) / total)
            prob = np.maximum(vals / total, 1e-12)
            entropy = float(-np.sum(prob * np.log(prob)) / np.log(len(freqs)))
            # 带宽: 以 centroid 为中心的加权标准差
            bandwidth = float(np.sqrt(np.sum(vals * (freqs - centroid) ** 2) / total))
        else:
            centroid, entropy, bandwidth = 0.0, 0.0, 0.0

        # --- Sigma 细分 (2) ---
        f_low_sigma_mean  = cls._band_mean(coh, *b["low_sigma"])
        f_high_sigma_mean = cls._band_mean(coh, *b["high_sigma"])

        return np.array([
            f_mean_delta, f_mean_theta, f_mean_alpha, f_mean_sigma, f_mean_beta,
            f_auc_delta, f_auc_theta, f_auc_alpha, f_auc_sigma, f_auc_beta,
            f_iqr_delta, f_iqr_theta, f_iqr_alpha, f_iqr_sigma, f_iqr_beta,
            f_sigma_delta_ratio, f_alpha_delta_ratio, f_beta_delta_ratio, f_high_low_ratio,
            centroid, entropy, bandwidth,
            f_low_sigma_mean, f_high_sigma_mean,
        ], dtype=np.float32)

    # ====================================================================
    # 公有方法
    # ====================================================================

    def extract_eeg_coherence(self, eeg_data, fs, thr=100.0, dthr=45.0):
        """
        从原始 EEG 提取相干特征 (360 维)。

        Parameters
        ----------
        eeg_data : ndarray (n_channels, n_samples) or (n_samples,)
            原始 EEG 信号。支持任意通道数，自动选取最佳 6 导联。
        fs : float or int
            原始采样率 (Hz)。
        thr : float
            振幅伪迹阈值 (uV)。
        dthr : float
            差分伪迹阈值 (uV)。

        Returns
        -------
        features : (360,) ndarray
            15 导联对 × 24 特征，跨 epoch 均值。
        """
        # 使用 ref 中的 eeg_segment_coherence 获取每 epoch 相干谱
        epoch_features, pvalues = eeg_segment_coherence(
            eeg_data, fs, n_seg=None, thr=thr, dthr=dthr,
        )
        # epoch_features: (N_epochs, 1554)
        # 后 1500 维 = 15 导联对 × 100 频点

        n_epochs = epoch_features.shape[0]
        if n_epochs == 0:
            return np.zeros(EEG_FEATURE_DIM, dtype=np.float32)

        # ---- 频谱特征: 前 54 维跨 epoch 均值 ----
        spectral = np.mean(epoch_features[:, :SPECTRAL_FEATURE_DIM], axis=0)  # (54,)

        # ---- 相干特征: 每 epoch 每导联对 24 维 ----
        all_pair_features = []
        for ep in range(n_epochs):
            coh_start = 54  # 前 54 维是 PSD 特征
            pair_features_this_epoch = []
            for pair_idx in range(N_PAIRS):
                start = coh_start + pair_idx * N_FFT_BINS
                coh_spectrum = epoch_features[ep, start:start + N_FFT_BINS]
                f24 = self._extract_coh_features_from_spectrum(coh_spectrum)
                pair_features_this_epoch.append(f24)
            all_pair_features.append(np.stack(pair_features_this_epoch))  # (15, 24)

        # 跨 epoch 聚合: 均值
        all_pair_features = np.stack(all_pair_features, axis=0)  # (N_ep, 15, 24)
        coherence = np.mean(all_pair_features, axis=0).ravel()  # (360,)

        return np.concatenate([spectral, coherence]).astype(np.float32)

    def extract_eeg_coherence_from_processed(self, processed_channels, processed_fs):
        """
        从已标准化的通道字典中提取 EEG 相干特征。

        Parameters
        ----------
        processed_channels : dict {label: signal_array}
            已重命名并双极构建后的通道字典。
        processed_fs : dict {label: fs}
            对应的采样率。

        Returns
        -------
        features : (360,) ndarray
        """
        EEG_CH_ORDER = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1']
        eeg_signals = []
        fs_val = None

        for ch in EEG_CH_ORDER:
            if ch in processed_channels and processed_channels[ch] is not None:
                sig = processed_channels[ch]
                if len(sig) > 1:
                    eeg_signals.append(sig)
                    if fs_val is None:
                        fs_val = processed_fs.get(ch, 200.0)

        if len(eeg_signals) < 2:
            return np.zeros(COHERENCE_FEATURE_DIM, dtype=np.float32)

        # 堆叠为 (n_chan, n_samples)
        eeg_data = np.stack(eeg_signals, axis=0)
        return self.extract_eeg_coherence(eeg_data, fs_val)

    @staticmethod
    def coherence_feature_names():
        """返回 360 维特征的名称列表。"""
        stat_names = [
            "mean_delta", "mean_theta", "mean_alpha", "mean_sigma", "mean_beta",
            "auc_delta", "auc_theta", "auc_alpha", "auc_sigma", "auc_beta",
            "iqr_delta", "iqr_theta", "iqr_alpha", "iqr_sigma", "iqr_beta",
            "sigma_delta_ratio", "alpha_delta_ratio", "beta_delta_ratio", "high_low_ratio",
            "spectral_centroid", "spectral_entropy", "spectral_bandwidth",
            "low_sigma_mean", "high_sigma_mean",
        ]
        names = []
        for pair in PAIR_NAMES:
            for stat in stat_names:
                names.append(f"coh_{pair}_{stat}")
        return names
