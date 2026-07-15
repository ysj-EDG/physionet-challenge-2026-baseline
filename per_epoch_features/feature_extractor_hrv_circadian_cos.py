#!/usr/bin/env python
"""Circadian time features aligned to ECG/HRV sliding windows."""

from datetime import datetime, timedelta
import logging

import numpy as np

logger = logging.getLogger("feature_extractor_hrv_circadian_cos")

HRV_CIRCADIAN_FEATURE_DIM = 1
HRV_CIRCADIAN_FEATURE_NAMES = ["hrv_circadian_cos"]


def circadian_cos(dt):
    frac = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
    return float(np.cos(2.0 * np.pi * frac))


def read_edf_start_time(edf_path):
    """Read PSG recording start time from an EDF header."""
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


def extract_hrv_window_circadian_cos(
    edf_start_time, n_wins, win_sec=300.0, stride_sec=30.0,
):
    """Return one circadian cosine value per ECG/HRV window midpoint."""
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
