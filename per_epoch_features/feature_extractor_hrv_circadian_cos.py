#!/usr/bin/env python
"""
ECG/HRV 滑动窗口的昼夜节律时间特征提取器。

输出:
    X_circadian: (N_5min_wins, 1) — 每个 ECG/HRV 窗口中点对应的
    circadian cosine 特征，默认窗口长度为 5 分钟、步长为 30 秒。

时间对齐:
    - 第 i 个特征对应窗口起点后的 i * stride_sec + win_sec / 2 时刻
    - 默认设置下，第一个特征对应记录开始后 2.5 分钟

回退策略:
    - 优先通过 pyedflib 读取 EDF 起始时间
    - pyedflib 不可用或读取失败时，直接解析 EDF 固定头字段
    - 起始时间不可用时，为全部窗口返回零值
"""

from datetime import datetime, timedelta
import logging

import numpy as np

logger = logging.getLogger("feature_extractor_hrv_circadian_cos")

# 单个 HRV 窗口仅生成一个昼夜节律余弦特征。
HRV_CIRCADIAN_FEATURE_DIM = 1
HRV_CIRCADIAN_FEATURE_NAMES = ["hrv_circadian_cos"]


# ============================================================================
# 昼夜节律时间编码
# ============================================================================

def circadian_cos(dt):
    """
    将一天中的时刻编码为 24 小时周期的余弦值。

    午夜对应 1，正午对应 -1；日期本身不参与计算。
    """
    frac = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
    return float(np.cos(2.0 * np.pi * frac))


# ============================================================================
# EDF 起始时间读取
# ============================================================================

def read_edf_start_time(edf_path):
    """
    从 PSG 的 EDF 文件头读取记录起始时间。

    首先使用 pyedflib；失败后读取 EDF 固定头中的日期和时间字段。
    两种方式均失败时返回 None，由特征提取函数执行零值回退。
    """
    try:
        import pyedflib
        reader = pyedflib.EdfReader(edf_path)
        try:
            return reader.getStartdatetime()
        finally:
            reader.close()
    except Exception as exc:
        logger.debug("Could not read EDF start time from %s: %s", edf_path, exc)

    try:
        with open(edf_path, "rb") as f:
            header = f.read(184)
        date_raw = header[168:176].decode("latin1").strip()
        time_raw = header[176:184].decode("latin1").strip()
        return datetime.strptime(f"{date_raw} {time_raw}", "%d.%m.%y %H.%M.%S")
    except Exception as exc:
        logger.debug("Could not parse EDF raw start time from %s: %s", edf_path, exc)
        return None


# ============================================================================
# 公有 API
# ============================================================================

def extract_hrv_window_circadian_cos(
    edf_start_time, n_wins, win_sec=300.0, stride_sec=30.0,
):
    """
    为每个 ECG/HRV 滑动窗口提取窗口中点的昼夜节律余弦值。

    Parameters
    ----------
    edf_start_time : datetime or None
        EDF 记录起始时间；为 None 时返回与窗口数匹配的全零特征。
    n_wins : int
        ECG/HRV 滑动窗口数量。
    win_sec : float, default=300.0
        单个窗口时长（秒）。
    stride_sec : float, default=30.0
        相邻窗口起点之间的步长（秒）。

    Returns
    -------
    features : (n_wins, 1) ndarray
        各窗口中点时刻的昼夜节律余弦特征，数据类型为 float32。
    """
    n_wins = int(n_wins)
    if n_wins <= 0:
        return np.zeros((0, HRV_CIRCADIAN_FEATURE_DIM), dtype=np.float32)
    if edf_start_time is None:
        return np.zeros((n_wins, HRV_CIRCADIAN_FEATURE_DIM), dtype=np.float32)

    values = []
    midpoint_offset = float(win_sec) / 2.0
    for i in range(n_wins):
        dt = edf_start_time + timedelta(seconds=i * float(stride_sec) + midpoint_offset)
        values.append(circadian_cos(dt))
    return np.asarray(values, dtype=np.float32).reshape(
        n_wins, HRV_CIRCADIAN_FEATURE_DIM
    )
