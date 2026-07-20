#!/usr/bin/env python
"""Portable NumPy EEG burst-suppression-ratio helpers.

The implementation is self-contained so extraction has no external BSR package
or absolute-path dependency.
"""

import numpy as np

EPOCH_SEC = 30
BSR_THRESHOLDS = (5, 10, 20)
BSR_CHANNELS = ("F3-M2", "F4-M1", "C3-M2", "C4-M1", "O1-M2", "O2-M1")
BSR_FEATURE_DIM = len(BSR_THRESHOLDS) * len(BSR_CHANNELS)


def eeg_bsr(epochs, thresholds=(5, 10, 20)):
    """Calculate channel-wise burst suppression ratios from two-second subepochs."""
    n_epochs, n_chans = epochs.shape[:2]
    thresholds = np.asarray(thresholds)
    ptp = np.ptp(epochs, axis=-1)
    bsr = np.zeros((n_chans, len(thresholds)))

    for i, threshold in enumerate(thresholds):
        bsr[:, i] = np.sum(ptp < threshold, axis=0) / n_epochs * 100.0

    return bsr


def extract_bsr_30s(eeg_data, n_epochs, fs=200.0):
    """Return 18 BSR features aligned to 30-second EEG epochs."""
    eeg_data = np.asarray(eeg_data, dtype=float)
    epoch_samples = int(round(EPOCH_SEC * fs))
    subepoch_samples = int(round(2.0 * fs))

    rows = []
    for epoch_index in range(int(n_epochs)):
        segment = eeg_data[:, epoch_index * epoch_samples:(epoch_index + 1) * epoch_samples]
        n_subepochs = segment.shape[1] // subepoch_samples
        if n_subepochs == 0:
            rows.append(np.zeros(BSR_FEATURE_DIM, dtype=np.float32))
            continue

        epochs_2s = segment[:, :n_subepochs * subepoch_samples].reshape(
            segment.shape[0], n_subepochs, subepoch_samples,
        ).transpose(1, 0, 2)
        bsr = eeg_bsr(epochs_2s, thresholds=BSR_THRESHOLDS)
        rows.append(np.asarray(bsr, dtype=np.float32).T.ravel())

    if not rows:
        return np.zeros((0, BSR_FEATURE_DIM), dtype=np.float32)
    return np.nan_to_num(np.stack(rows, axis=0).astype(np.float32), nan=0.0)


def bsr_feature_names():
    return [
        f"eeg_bsr_{threshold}uv_{channel}"
        for threshold in BSR_THRESHOLDS
        for channel in BSR_CHANNELS
    ]
