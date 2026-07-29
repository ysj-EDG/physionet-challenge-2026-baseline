#!/usr/bin/env python
"""
30 秒 epoch 睡眠分期与事件标签特征提取器。

将 CAISR 睡眠分期编码为 one-hot，并计算觉醒、呼吸事件和肢体运动
在每个 epoch 内的时间覆盖比例，再跨 epoch 计算 mean 和 std。

输出:
    features: (26,) — 13 个逐 epoch 标签特征 × 2 个统计量

逐 epoch 特征:
    睡眠分期 one-hot (5) + arousal (1) + 呼吸事件亚型 (5)
    + isolated limb movement (1) + periodic limb movement (1)

聚合含义:
    mean 表示各阶段出现比例或各事件的平均时间覆盖比例，std 表示其
    在不同 epoch 之间的波动。
"""

import numpy as np

# ============================================================================
# Epoch 与标签定义
# ============================================================================

EPOCH_SEC = 30.0
EVENT_ONEHOT_FEATURE_DIM = 26

# 睡眠分期名称及 CAISR 编码，输出顺序为 N1、N2、N3、REM、Wake。
STAGE_LABELS = [("N1", 3), ("N2", 2), ("N3", 1), ("REM", 4), ("Wake", 5)]


# ============================================================================
# 事件覆盖比例
# ============================================================================

def _event_fraction_in_epoch(event_starts_sec, event_ends_sec,
                             epoch_start, epoch_end):
    """
    计算一个 epoch 内事件区间覆盖的时间比例。

    Parameters
    ----------
    event_starts_sec, event_ends_sec : 1D array-like
        各事件的起止时间（秒），采用一一对应的区间表示。
    epoch_start, epoch_end : float
        当前 epoch 的起止时间（秒）。

    Returns
    -------
    fraction : float
        事件覆盖时长占 30 秒 epoch 的比例，并限制在 0–1。
    """
    if len(event_starts_sec) == 0:
        return 0.0
    overlap = 0.0
    for s, e in zip(event_starts_sec, event_ends_sec):
        overlap += max(0.0, min(e, epoch_end) - max(s, epoch_start))
    return min(1.0, overlap / EPOCH_SEC)


def _event_fractions_by_epoch(event_starts_sec, event_ends_sec, n_epochs):
    """
    批量计算有序、不重叠事件区间在所有 epoch 内的覆盖比例。

    通过 ``searchsorted`` 计算各 epoch 边界处的累计覆盖时长，避免为
    每个 epoch 重复遍历全部事件。

    Returns
    -------
    fractions : (n_epochs,) ndarray
        每个 30 秒 epoch 的事件覆盖比例，数据类型为 float32。
    """
    if n_epochs <= 0:
        return np.zeros(0, dtype=np.float32)

    starts = np.asarray(event_starts_sec, dtype=float).reshape(-1)
    ends = np.asarray(event_ends_sec, dtype=float).reshape(-1)
    if len(starts) == 0:
        return np.zeros(n_epochs, dtype=np.float32)
    if len(starts) != len(ends):
        raise ValueError("event starts and ends must have the same length")

    boundaries = np.arange(n_epochs + 1, dtype=float) * EPOCH_SEC
    durations = ends - starts
    cumulative = np.concatenate([[0.0], np.cumsum(durations)])

    # 统计每个 epoch 边界之前已经开始的事件区间数量。
    started = np.searchsorted(starts, boundaries, side="left")
    covered = cumulative[started].copy()
    has_started = started > 0
    last_interval = started[has_started] - 1
    covered[has_started] -= np.maximum(
        ends[last_interval] - boundaries[has_started], 0.0,
    )

    fractions = np.diff(covered) / EPOCH_SEC
    return np.clip(fractions, 0.0, 1.0).astype(np.float32)


# ============================================================================
# EventOneHotMixin — 集成到 FeatureExtractor
# ============================================================================

class EventOneHotMixin:
    """
    从 CAISR 标注提取逐 epoch 阶段与事件特征，并聚合为 26 维向量。

    睡眠分期使用 one-hot 编码，事件标签使用当前 epoch 内的覆盖比例。
    """

    # ========================================================================
    # 公有 API
    # ========================================================================

    def extract_event_onehot(self, algo_data):
        """
        从 CAISR 算法输出字典提取睡眠分期与事件标签特征。

        Parameters
        ----------
        algo_data : dict
            CAISR 标注信号字典，睡眠分期键为 ``stage_caisr``；可选事件键
            包括 ``arousal_caisr``、``resp_caisr`` 和 ``limb_caisr``。

        Returns
        -------
        features : (26,) ndarray
            13 个逐 epoch 标签特征 × mean/std，数据类型为 float32。
            睡眠分期缺失或无有效分期时返回全零向量。
        """
        if not algo_data or 'stage_caisr' not in algo_data:
            return np.zeros(EVENT_ONEHOT_FEATURE_DIM, dtype=np.float32)

        raw_stages = np.asarray(algo_data['stage_caisr'], dtype=float).reshape(-1)
        # 仅保留 CAISR 编码 1–5 的有效分期，并以其数量确定记录时长。
        valid = np.isin(raw_stages, [1, 2, 3, 4, 5])
        stages = raw_stages[valid].astype(int)
        n_epochs = len(stages)
        if n_epochs == 0:
            return np.zeros(EVENT_ONEHOT_FEATURE_DIM, dtype=np.float32)

        trt_sec = n_epochs * EPOCH_SEC

        # ---- 将离散标签信号转换为连续事件区间 ----
        def _segments_from_binary(signal, dt_sec):
            """将二值序列转换为以秒表示的左闭右开连续区间。"""
            sig = np.asarray(signal, dtype=float).reshape(-1)
            binary = sig > 0
            edges = np.diff(binary.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0].astype(float) * dt_sec
            ends = np.where(edges == -1)[0].astype(float) * dt_sec
            return starts, ends

        # 觉醒事件：时间步长由总记录时长和标签数组长度推算。
        arousal_signal = np.asarray(algo_data.get('arousal_caisr', np.array([])), dtype=float).reshape(-1)
        arousal_dt = trt_sec / len(arousal_signal) if len(arousal_signal) > 0 else 0.5
        arousal_starts, arousal_ends = _segments_from_binary(arousal_signal, arousal_dt)

        # 呼吸事件：按 OA、CA、MA、HY 和 RERA 五种 CAISR 编码分别分段。
        resp_signal = np.asarray(algo_data.get('resp_caisr', np.array([])), dtype=float).reshape(-1)
        resp_dt = trt_sec / len(resp_signal) if len(resp_signal) > 0 else 1.0
        resp_starts_by_type = {}
        resp_ends_by_type = {}
        for rtype, rcode in [("OA", 1), ("CA", 2), ("MA", 3), ("HY", 4), ("RERA", 5)]:
            s, e = _segments_from_binary(resp_signal == rcode, resp_dt)
            resp_starts_by_type[rtype] = s
            resp_ends_by_type[rtype] = e

        # 肢体运动：分别提取 isolated movement 和 PLM 连续区间。
        limb_signal = np.asarray(algo_data.get('limb_caisr', np.array([])), dtype=float).reshape(-1)
        limb_dt = trt_sec / len(limb_signal) if len(limb_signal) > 0 else 1.0
        l_iso_s, l_iso_e = _segments_from_binary(limb_signal == 1, limb_dt)
        l_plm_s, l_plm_e = _segments_from_binary(limb_signal == 2, limb_dt)

        # ---- 逐 30 秒 epoch 生成 13 维标签特征 ----
        rows = []
        for ep in range(n_epochs):
            t0 = ep * EPOCH_SEC
            t1 = t0 + EPOCH_SEC
            row = []

            # 睡眠分期 one-hot（5 维）。
            for _, code in STAGE_LABELS:
                row.append(1.0 if stages[ep] == code else 0.0)

            # 觉醒事件覆盖比例（1 维）。
            row.append(_event_fraction_in_epoch(arousal_starts, arousal_ends, t0, t1))

            # 五种呼吸事件亚型的覆盖比例（5 维）。
            for rtype in ["OA", "CA", "MA", "HY", "RERA"]:
                row.append(_event_fraction_in_epoch(
                    resp_starts_by_type[rtype], resp_ends_by_type[rtype], t0, t1))

            # isolated movement 与 PLM 的覆盖比例（2 维）。
            row.append(_event_fraction_in_epoch(l_iso_s, l_iso_e, t0, t1))
            row.append(_event_fraction_in_epoch(l_plm_s, l_plm_e, t0, t1))

            rows.append(row)

        rows = np.asarray(rows, dtype=np.float32)  # (N_epochs, 13)
        feat_mean = np.mean(rows, axis=0)
        feat_std = np.std(rows, axis=0)
        return np.concatenate([feat_mean, feat_std]).astype(np.float32)

    @staticmethod
    def event_onehot_feature_names():
        """返回与 26 维 mean/std 输出顺序一致的特征名称列表。"""
        names = []
        for stat in ["mean", "std"]:
            for _, name in STAGE_LABELS:
                names.append(f"onehot_stage_{name}_{stat}")
            names.append(f"onehot_arousal_{stat}")
            for rtype in ["OA", "CA", "MA", "HY", "RERA"]:
                names.append(f"onehot_resp_{rtype}_{stat}")
            names.append(f"onehot_limb_isolated_{stat}")
            names.append(f"onehot_limb_PLM_{stat}")
        return names
