#!/usr/bin/env python
"""Portable BSR helpers copied from the validated NeuroCardioKit source.

The original H100 run loaded this exact implementation from
NeuroCardioKit/eeg/_nonlinear.py (SHA-256:
a726a11325ed00f14eaacbb1e0f050c23d2fb9aed9a0387f6533e3f931a2a702).
Keeping the small NumPy-only routine here removes the non-portable absolute
fallback path from the official Docker submission.
"""

import numpy as np

EPOCH_SEC = 30
NCK_BSR_THRESHOLDS = (5, 10, 20)
NCK_BSR_CHANNELS = ("F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1")
NCK_BSR_FEATURE_DIM = len(NCK_BSR_THRESHOLDS) * len(NCK_BSR_CHANNELS)


def eeg_bsr(epochs, thresholds=(5, 10, 20)):
    """Burst Suppression Ratio from NeuroCardioKit's validated implementation."""
    n_epochs, n_chans = epochs.shape[:2]
    thresholds = np.asarray(thresholds)
    ptp = np.ptp(epochs, axis=-1)
    bsr = np.zeros((n_chans, len(thresholds)))

    for i, threshold in enumerate(thresholds):
        bsr[:, i] = np.sum(ptp < threshold, axis=0) / n_epochs * 100.0

    return bsr


def extract_nck_bsr_30s(eeg_data, n_epochs, fs=200.0):
    """Return 18 BSR features aligned to 30-second EEG epochs."""
    eeg_data = np.asarray(eeg_data, dtype=float)
    epoch_samples = int(round(EPOCH_SEC * fs))
    subepoch_samples = int(round(2.0 * fs))

    rows = []
    for epoch_index in range(int(n_epochs)):
        segment = eeg_data[:, epoch_index * epoch_samples:(epoch_index + 1) * epoch_samples]
        n_subepochs = segment.shape[1] // subepoch_samples
        if n_subepochs == 0:
            rows.append(np.zeros(NCK_BSR_FEATURE_DIM, dtype=np.float32))
            continue

        epochs_2s = segment[:, :n_subepochs * subepoch_samples].reshape(
            segment.shape[0], n_subepochs, subepoch_samples,
        ).transpose(1, 0, 2)
        bsr = eeg_bsr(epochs_2s, thresholds=NCK_BSR_THRESHOLDS)
        rows.append(np.asarray(bsr, dtype=np.float32).T.ravel())

    if not rows:
        return np.zeros((0, NCK_BSR_FEATURE_DIM), dtype=np.float32)
    return np.nan_to_num(np.stack(rows, axis=0).astype(np.float32), nan=0.0)


def nck_bsr_feature_names():
    return [
        f"eeg_bsr_{threshold}uv_{channel}"
        for threshold in NCK_BSR_THRESHOLDS
        for channel in NCK_BSR_CHANNELS
    ]
