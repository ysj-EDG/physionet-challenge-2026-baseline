#!/usr/bin/env python
"""
EEG 爆发抑制比（Burst Suppression Ratio, BSR）特征提取器。

基于 NumPy 将每个 30 秒 EEG epoch 划分为完整的 2 秒子段，计算各子段
的峰峰值，并统计峰峰值低于不同振幅阈值的子段比例。

输出:
    X_bsr: (N_epochs, 18) — 3 个振幅阈值 × 6 个 EEG 通道

特征顺序:
    按阈值优先排列，即依次输出 5、10、20 µV 阈值下 6 个通道的 BSR。

实现说明:
    本实现仅依赖 NumPy，不依赖外部 BSR 包或绝对路径。
"""

import numpy as np

# ============================================================================
# BSR 参数与特征定义
# ============================================================================

# 每个输出 epoch 的固定时长（秒）。
EPOCH_SEC = 30
# 用于判定抑制子段的峰峰值阈值（µV）。
BSR_THRESHOLDS = (5, 10, 20)
# 输出特征采用的标准 EEG 通道顺序。
BSR_CHANNELS = ("F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1")
BSR_FEATURE_DIM = len(BSR_THRESHOLDS) * len(BSR_CHANNELS)


# ============================================================================
# BSR 计算核心
# ============================================================================

def eeg_bsr(epochs, thresholds=(5, 10, 20)):
    """
    从 2 秒子段计算各 EEG 通道在不同阈值下的爆发抑制比。

    Parameters
    ----------
    epochs : ndarray, shape (n_subepochs, n_channels, n_samples)
        同一分析窗口内的 EEG 子段。
    thresholds : sequence of float, default=(5, 10, 20)
        抑制判定的峰峰值阈值（µV）。

    Returns
    -------
    bsr : ndarray, shape (n_channels, n_thresholds)
        每个通道中峰峰值低于相应阈值的子段百分比，范围为 0–100。
    """
    n_epochs, n_chans = epochs.shape[:2]
    thresholds = np.asarray(thresholds)
    # 每个 2 秒子段、每个通道沿时间轴计算峰峰值。
    ptp = np.ptp(epochs, axis=-1)
    bsr = np.zeros((n_chans, len(thresholds)))

    for i, threshold in enumerate(thresholds):
        # 抑制子段数除以子段总数，转换为百分比。
        bsr[:, i] = np.sum(ptp < threshold, axis=0) / n_epochs * 100.0

    return bsr


# ============================================================================
# 公有 API
# ============================================================================

def extract_bsr_30s(
    eeg_data,
    n_epochs,
    fs=200.0,
    channel_available=None,
):
    """
    提取与 30 秒 EEG epoch 对齐的 18 维 BSR 特征。

    Parameters
    ----------
    eeg_data : ndarray, shape (6, n_samples)
        按 ``BSR_CHANNELS`` 顺序排列的连续 EEG 信号。
    n_epochs : int
        需要输出的 30 秒 epoch 数量。
    fs : float, default=200.0
        EEG 采样率（Hz）。

    Returns
    -------
    features : ndarray, shape (n_epochs, 18)
        每行包含 3 个阈值 × 6 个通道的 BSR，数据类型为 float32。
        无完整 2 秒子段的 epoch 返回全零特征。
    """
    eeg_data = np.asarray(eeg_data, dtype=float)

    if eeg_data.ndim != 2:
        raise ValueError(
            f"eeg_data must be 2D, got shape {eeg_data.shape}"
        )

    n_channels = eeg_data.shape[0]

    if channel_available is None:
        channel_available = np.ones(
            n_channels,
            dtype=bool,
        )
    else:
        channel_available = np.asarray(
            channel_available,
            dtype=bool,
        ).reshape(-1)

        if channel_available.shape != (n_channels,):
            raise ValueError(
                "channel_available shape "
                f"{channel_available.shape}; "
                f"expected ({n_channels},)"
            )

    epoch_samples = int(round(EPOCH_SEC * fs))
    subepoch_samples = int(round(2.0 * fs))

    rows = []
    for epoch_index in range(int(n_epochs)):
        segment = eeg_data[:, epoch_index * epoch_samples:(epoch_index + 1) * epoch_samples]
        n_subepochs = segment.shape[1] // subepoch_samples
        if n_subepochs == 0:
            rows.append(np.zeros(BSR_FEATURE_DIM, dtype=np.float32))
            continue

        # 仅保留完整的 2 秒子段，并整理为
        # (n_subepochs, n_channels, n_samples) 供 eeg_bsr() 计算。
        epochs_2s = segment[:, :n_subepochs * subepoch_samples].reshape(
            segment.shape[0], n_subepochs, subepoch_samples,
        ).transpose(1, 0, 2)
        bsr = eeg_bsr(epochs_2s, thresholds=BSR_THRESHOLDS)
        bsr[~channel_available, :] = 0.0
        # 转置后按“阈值优先、通道其次”的顺序展平。
        rows.append(np.asarray(bsr, dtype=np.float32).T.ravel())

    if not rows:
        return np.zeros((0, BSR_FEATURE_DIM), dtype=np.float32)
    return np.nan_to_num(np.stack(rows, axis=0).astype(np.float32), nan=0.0)


def bsr_feature_names():
    """
    返回与 BSR 输出列顺序一致的 18 个特征名称。

    名称按阈值优先、通道其次排列。
    """
    return [
        f"eeg_bsr_{threshold}uv_{channel}"
        for threshold in BSR_THRESHOLDS
        for channel in BSR_CHANNELS
    ]
