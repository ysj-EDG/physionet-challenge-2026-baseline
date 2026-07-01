#!/usr/bin/env python
"""
30s epoch 标签 One-Hot 特征提取器。

将 CAISR 睡眠分期和事件标注转为每 epoch 的 one-hot 向量，
跨 epoch 取 mean (比例) + std (波动)，共 26 维。
"""

import numpy as np

EPOCH_SEC = 30.0
EVENT_ONEHOT_FEATURE_DIM = 26

# 睡眠分期 one-hot: N1, N2, N3, REM, Wake
STAGE_LABELS = [("N1", 3), ("N2", 2), ("N3", 1), ("REM", 4), ("Wake", 5)]


def _event_fraction_in_epoch(event_starts_sec, event_ends_sec,
                             epoch_start, epoch_end):
    """
    计算 epoch 内有事件覆盖的时间比例。
    event_starts_sec / event_ends_sec: 1D 数组，长度 = 事件数。
    """
    if len(event_starts_sec) == 0:
        return 0.0
    overlap = 0.0
    for s, e in zip(event_starts_sec, event_ends_sec):
        overlap += max(0.0, min(e, epoch_end) - max(s, epoch_start))
    return min(1.0, overlap / EPOCH_SEC)


class EventOneHotMixin:
    """
    从 CAISR 标注提取 per-epoch one-hot 特征，跨 epoch 聚合为 mean+std。
    """

    def extract_event_onehot(self, algo_data):
        """
        Parameters
        ----------
        algo_data : dict
            CAISR 标注信号字典。

        Returns
        -------
        features : (26,) ndarray
            13 类标签 × 2 统计量 (mean, std)。
        """
        if not algo_data or 'stage_caisr' not in algo_data:
            return np.zeros(EVENT_ONEHOT_FEATURE_DIM, dtype=np.float32)

        raw_stages = np.asarray(algo_data['stage_caisr'], dtype=float).reshape(-1)
        valid = np.isin(raw_stages, [1, 2, 3, 4, 5])
        stages = raw_stages[valid].astype(int)
        n_epochs = len(stages)
        if n_epochs == 0:
            return np.zeros(EVENT_ONEHOT_FEATURE_DIM, dtype=np.float32)

        trt_sec = n_epochs * EPOCH_SEC

        # ---- 事件分割 (复用逻辑) ----
        def _segments_from_binary(signal, dt_sec):
            sig = np.asarray(signal, dtype=float).reshape(-1)
            binary = sig > 0
            edges = np.diff(binary.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0].astype(float) * dt_sec
            ends = np.where(edges == -1)[0].astype(float) * dt_sec
            return starts, ends

        # arousal (0.5s)
        arousal_signal = np.asarray(algo_data.get('arousal_caisr', np.array([])), dtype=float).reshape(-1)
        arousal_dt = trt_sec / len(arousal_signal) if len(arousal_signal) > 0 else 0.5
        arousal_starts, arousal_ends = _segments_from_binary(arousal_signal, arousal_dt)

        # respiratory (1s) — 按亚型
        resp_signal = np.asarray(algo_data.get('resp_caisr', np.array([])), dtype=float).reshape(-1)
        resp_dt = trt_sec / len(resp_signal) if len(resp_signal) > 0 else 1.0
        resp_starts_by_type = {}
        resp_ends_by_type = {}
        for rtype, rcode in [("OA", 1), ("CA", 2), ("MA", 3), ("HY", 4), ("RERA", 5)]:
            s, e = _segments_from_binary(resp_signal == rcode, resp_dt)
            resp_starts_by_type[rtype] = s
            resp_ends_by_type[rtype] = e

        # limb (1s) — 按亚型
        limb_signal = np.asarray(algo_data.get('limb_caisr', np.array([])), dtype=float).reshape(-1)
        limb_dt = trt_sec / len(limb_signal) if len(limb_signal) > 0 else 1.0
        l_iso_s, l_iso_e = _segments_from_binary(limb_signal == 1, limb_dt)
        l_plm_s, l_plm_e = _segments_from_binary(limb_signal == 2, limb_dt)

        # ---- 每 epoch 提取 ----
        rows = []
        for ep in range(n_epochs):
            t0 = ep * EPOCH_SEC
            t1 = t0 + EPOCH_SEC
            row = []

            # 睡眠分期 one-hot
            for _, code in STAGE_LABELS:
                row.append(1.0 if stages[ep] == code else 0.0)

            # arousal
            row.append(_event_fraction_in_epoch(arousal_starts, arousal_ends, t0, t1))

            # 呼吸事件 fraction
            for rtype in ["OA", "CA", "MA", "HY", "RERA"]:
                row.append(_event_fraction_in_epoch(
                    resp_starts_by_type[rtype], resp_ends_by_type[rtype], t0, t1))

            # 肢体运动 fraction
            row.append(_event_fraction_in_epoch(l_iso_s, l_iso_e, t0, t1))
            row.append(_event_fraction_in_epoch(l_plm_s, l_plm_e, t0, t1))

            rows.append(row)

        rows = np.asarray(rows, dtype=np.float32)  # (N_epochs, 13)
        feat_mean = np.mean(rows, axis=0)
        feat_std = np.std(rows, axis=0)
        return np.concatenate([feat_mean, feat_std]).astype(np.float32)

    @staticmethod
    def event_onehot_feature_names():
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
