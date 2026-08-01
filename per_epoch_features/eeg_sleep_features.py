"""
Sleep EEG feature extraction — Python translation of the MATLAB pipeline:
  eeg_segment_psd -> eeg_bsi_psd -> eeg_sleep_PSD_feature

Pure NumPy implementation (no scipy dependency).

Input:  raw EEG (1D/2D ndarray, fs)
Output: (N_segments, 9) feature matrix
"""

import numpy as np

try:
    from scipy.signal import lfilter as _scipy_lfilter
except ImportError:  # Keep the reference implementation usable without SciPy.
    _scipy_lfilter = None

# ---------------------------------------------------------------------------
# Pure-numpy signal utilities (avoid scipy version incompatibility)
# ---------------------------------------------------------------------------


def _detrend(x):
    """Remove linear trend (same as MATLAB detrend / scipy.signal.detrend)."""
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    A = np.column_stack([np.ones(n), t])
    coef = np.linalg.lstsq(A, x, rcond=None)[0]
    return x - A @ coef


def _filtfilt(b, a, x):
    """
    Zero-phase forward-backward digital filter.
    Equivalent to scipy.signal.filtfilt / MATLAB filtfilt.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    # Pad at both ends with reflected copies (3 * max(len(a), len(b)) is typical)
    n_pad = 3 * max(len(a), len(b))
    if n_pad >= n:
        n_pad = n - 1
    if n_pad > 0:
        x_pad = np.concatenate([2 * x[0] - x[n_pad:0:-1], x, 2 * x[-1] - x[-2:-n_pad - 2:-1]])
    else:
        x_pad = x.copy()

    if _scipy_lfilter is not None:
        # scipy.signal.lfilter uses the same zero-initial-state recurrence as
        # the loops below, but executes it in compiled code.  Keep the custom
        # reflection padding so numerical behaviour remains unchanged.
        y_fwd = _scipy_lfilter(b, a, x_pad)
        y_out = _scipy_lfilter(b, a, y_fwd[::-1])[::-1]
    else:
        # Pure NumPy/Python fallback for environments without SciPy.
        y_fwd = np.zeros_like(x_pad)
        for i in range(len(x_pad)):
            y_fwd[i] = b[0] * x_pad[i]
            for j in range(1, len(b)):
                if i - j >= 0:
                    y_fwd[i] += b[j] * x_pad[i - j]
            for j in range(1, len(a)):
                if i - j >= 0:
                    y_fwd[i] -= a[j] * y_fwd[i - j]
            y_fwd[i] /= a[0]

        y_rev = y_fwd[::-1]
        y_out = np.zeros_like(y_rev)
        for i in range(len(y_rev)):
            y_out[i] = b[0] * y_rev[i]
            for j in range(1, len(b)):
                if i - j >= 0:
                    y_out[i] += b[j] * y_rev[i - j]
            for j in range(1, len(a)):
                if i - j >= 0:
                    y_out[i] -= a[j] * y_out[i - j]
            y_out[i] /= a[0]
        y_out = y_out[::-1]

    # Remove padding
    if n_pad > 0:
        y_out = y_out[n_pad:-n_pad]
    return y_out


def _detrend_last_axis(x):
    """Linear detrend many signals at once along the last axis."""
    x = np.asarray(x, dtype=np.float64)
    original_shape = x.shape
    n = original_shape[-1]
    flat = x.reshape(-1, n)
    t = np.arange(n, dtype=np.float64)
    design = np.column_stack([np.ones(n), t])
    coefficients = np.linalg.lstsq(design, flat.T, rcond=None)[0]
    detrended = flat - (design @ coefficients).T
    return detrended.reshape(original_shape)


def _filtfilt_last_axis(b, a, x):
    """Zero-phase filter many signals using the same padding as `_filtfilt`."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    n_pad = min(3 * max(len(a), len(b)), n - 1)
    if n_pad > 0:
        x_pad = np.concatenate([
            2 * x[..., :1] - x[..., n_pad:0:-1],
            x,
            2 * x[..., -1:] - x[..., -2:-n_pad - 2:-1],
        ], axis=-1)
    else:
        x_pad = x.copy()

    if _scipy_lfilter is None:
        flat = x.reshape(-1, x.shape[-1])
        filtered = np.stack([_filtfilt(b, a, row) for row in flat], axis=0)
        return filtered.reshape(x.shape)

    y_fwd = _scipy_lfilter(b, a, x_pad, axis=-1)
    y_out = _scipy_lfilter(b, a, y_fwd[..., ::-1], axis=-1)[..., ::-1]
    if n_pad > 0:
        y_out = y_out[..., n_pad:-n_pad]
    return y_out


def _resample(x, num):
    """Resample x to num samples via FFT (same as MATLAB resample / scipy.signal.resample)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    X = np.fft.fft(x)
    n_half = n // 2 + 1
    num_half = num // 2 + 1
    Y = np.zeros(num, dtype=complex)
    copy_len = min(n_half, num_half)
    Y[:copy_len] = X[:copy_len]
    # Mirror the negative frequencies
    for k in range(1, num - num_half + 1):
        src = n - k
        if src > 0 and src < n:
            Y[num - k] = X[src]
    y = np.fft.ifft(Y).real * (num / n)
    return y


# ---------------------------------------------------------------------------
# 1. Epoch segmentation
# ---------------------------------------------------------------------------

def eeg_epoch(x, seg_len, dlen, do_detrend=True):
    """
    Slice a 1-D signal into overlapping segments.

    Parameters
    ----------
    x : 1D ndarray
    seg_len : segment length in samples
    dlen : step between consecutive segment starts
    do_detrend : remove linear trend and mean from each segment

    Returns
    -------
    XX : (seg_len, n_seg) ndarray
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    n = len(x)
    n_epoch = (n - seg_len) // dlen + 1
    XX = np.zeros((seg_len, n_epoch))
    for i, start in enumerate(range(0, n - seg_len + 1, dlen)):
        seg = x[start:start + seg_len].copy()
        if do_detrend:
            seg = seg - seg.mean()
            seg = _detrend(seg)
        XX[:, i] = seg - seg.mean()
    return XX


# ---------------------------------------------------------------------------
# 2. Artifact detection
# ---------------------------------------------------------------------------

def eeg_is_artifact(x, fs, thr=100.0, dthr=45.0):
    """
    Flag a single-channel segment as artifact.

    Parameters
    ----------
    x : 1D ndarray, EEG samples
    fs : sample rate (Hz)
    thr : amplitude-range threshold (uV)
    dthr : differential threshold

    Returns
    -------
    is_art : bool
    maxminx : peak-to-peak amplitude (after smoothing)
    maxdx : max absolute difference
    """
    dn = max(1, int(round(fs / 50)))  # 20 ms

    dx = x[dn:] - x[:-dn]
    maxdx = np.max(np.abs(dx))

    # moving-average filter (length dn), then detrend
    b = np.ones(dn) / dn
    a = np.array([1.0])
    y = _filtfilt(b, a, x)
    y = _detrend(y)
    maxminx = np.max(y) - np.min(y)

    is_art = (maxdx > dthr) or (maxminx > thr) or (maxminx < 10)
    return is_art, maxminx, maxdx


# ---------------------------------------------------------------------------
# 3. Spectral estimation (Welch-like averaged periodogram)
# ---------------------------------------------------------------------------

def eeg_spectral(xx, fs):
    """
    Averaged periodogram with Hamming window.

    Parameters
    ----------
    xx : (nfft, n_epoch) ndarray — pre-segmented data (no window applied yet)
    fs : sample rate (Hz)

    Returns
    -------
    Pxx : 1D — mean PSD (positive frequencies only), shape (nfft//2,)
    waxis : 1D — frequency axis
    Pxxc : 1D — std of PSD across epochs
    """
    nfft, n_epoch = xx.shape
    w = np.hamming(nfft).reshape(-1, 1)
    U = float(w.T @ w)  # scalar, window energy
    xx_w = xx * w
    Fxx = np.fft.fft(xx_w, axis=0)
    Pxx_all = Fxx * np.conj(Fxx) * 2.0 / (U * fs)
    Pxx = np.mean(Pxx_all, axis=1).real
    Pxxc = np.std(Pxx_all, axis=1, ddof=0)
    n_half = nfft // 2
    Pxx = Pxx[:n_half]
    Pxxc = Pxxc[:n_half]
    fbin = nfft / fs
    waxis = np.arange(n_half, dtype=np.float64) / fbin
    return Pxx, waxis, Pxxc


# ---------------------------------------------------------------------------
# 4. PSD for a single 30-s epoch (sub-segmentation + artifact rejection)
# ---------------------------------------------------------------------------

def eeg_bsi_psd(x, fs=100, epoch_len=200, overlap=100, thr=100.0, dthr=45.0):
    """
    Compute clean PSD for one channel segment.

    Parameters
    ----------
    x : 1D ndarray — one 30-s segment (3000 samples at 100 Hz)
    fs : sample rate
    epoch_len : sub-segment length (samples) for Welch averaging
    overlap : step between sub-segment starts (samples)
    thr, dthr : artifact thresholds

    Returns
    -------
    Pxx : (nfft//2,) ndarray — mean PSD
    waxis : 1D — frequency axis
    pvalue : proportion of artifact-free sub-segments
    maxminx : mean peak-to-peak amplitude across sub-segments
    maxdx : mean max differential across sub-segments
    """
    XX = eeg_epoch(x, epoch_len, overlap, do_detrend=True)
    n_sub = XX.shape[1]

    art_flags = np.zeros(n_sub, dtype=bool)
    maxminx_arr = np.zeros(n_sub)
    maxdx_arr = np.zeros(n_sub)

    for i in range(n_sub):
        art_flags[i], maxminx_arr[i], maxdx_arr[i] = eeg_is_artifact(
            XX[:, i], fs, thr, dthr
        )

    pvalue = 1.0 - art_flags.sum() / n_sub
    XX_clean = XX[:, ~art_flags]

    n_half = epoch_len // 2
    if XX_clean.shape[1] == 0:
        Pxx = np.zeros(n_half)
        fbin = epoch_len / fs
        waxis = np.arange(n_half, dtype=np.float64) / fbin
        return Pxx, waxis, pvalue, np.mean(maxminx_arr), np.mean(maxdx_arr)

    Pxx, waxis, _ = eeg_spectral(XX_clean, fs)
    return Pxx, waxis, pvalue, np.mean(maxminx_arr), np.mean(maxdx_arr)


# ---------------------------------------------------------------------------
# 5. Spectral Edge Frequency 50%
# ---------------------------------------------------------------------------

def eeg_psd_stats_sef50(Pxx, fs):
    """
    50% Spectral Edge Frequency (median frequency).

    The PSD is first weighted by a linear ramp H = [0, 1/N, 2/N, ...)
    to approximate the differentiator [1 -1] frequency response,
    then the cumulative sum in 5–47 Hz is used to find the median.

    Returns
    -------
    SEF : median frequency (Hz), clamped to 47.0
    """
    n = len(Pxx)
    fbin = 2.0 * n / fs

    # Frequency weighting ramp (simplifies freqz([1 -1]) response)
    H = np.arange(n, dtype=np.float64) / n
    Pxx_w = Pxx * H

    f0 = 5.0
    i_start = int(np.floor(fbin * f0 + 1))
    i_end = int(np.floor(fbin * 47 + 1))
    i_start = max(0, min(i_start, n - 1))
    i_end = max(i_start + 1, min(i_end, n))

    Pxx_band = Pxx_w[i_start:i_end]
    Ps = np.sum(Pxx_band)
    if Ps <= 0:
        return 47.0

    cumsum = np.cumsum(Pxx_band)
    idx = int(np.searchsorted(cumsum, 0.50 * Ps))
    idx = min(idx, len(Pxx_band) - 1)

    SEF = f0 + idx / fbin
    return min(SEF, 47.0)


# ---------------------------------------------------------------------------
# 6. Band-power helper
# ---------------------------------------------------------------------------

def _sum_pxx(Pxx, fbin, f_low, f_high):
    """Sum PSD bins from f_low to f_high Hz (inclusive)."""
    i_lo = int(round(f_low * fbin + 1))
    i_hi = int(round(f_high * fbin + 1))
    i_lo = max(0, min(i_lo, len(Pxx) - 1))
    i_hi = max(i_lo, min(i_hi, len(Pxx) - 1))
    return float(np.sum(Pxx[i_lo:i_hi + 1]))


# ---------------------------------------------------------------------------
# 7. Feature extraction from a single PSD
# ---------------------------------------------------------------------------

def eeg_sleep_psd_feature_single(Pxx, fs):
    """
    Extract 9 sleep features from a single PSD vector.

    Returns
    -------
    ft : (9,) ndarray
    """
    if np.any(np.isnan(Pxx)) or np.all(Pxx == 0):
        return np.full(9, np.nan)

    fbin = 2.0 * len(Pxx) / fs

    P0 = _sum_pxx(Pxx, fbin, 0.5, 47)
    Pdelta = _sum_pxx(Pxx, fbin, 0.5, 3)
    Ptheta = _sum_pxx(Pxx, fbin, 4, 7)
    Palpha = _sum_pxx(Pxx, fbin, 8, 12)
    Psigma = _sum_pxx(Pxx, fbin, 11, 16)
    Pbeta = _sum_pxx(Pxx, fbin, 20, 30)
    P1 = _sum_pxx(Pxx, fbin, 30, 47)
    P2 = _sum_pxx(Pxx, fbin, 9, 20)

    SEF = eeg_psd_stats_sef50(Pxx, fs)

    eps = 1e-30
    ft = np.array([
        np.log10(max(P2, eps) / max(P1, eps)),
        np.log10(max(Psigma, eps) / max(Pbeta, eps)),
        np.log10(max(Palpha, eps) / max(Pbeta, eps)),
        np.log10(max(Ptheta, eps) / max(Pbeta, eps)),
        np.log10(max(Pdelta, eps) / max(Pbeta, eps)),
        SEF,
        np.log10(max(P0, eps)),
        np.log10(max(Pbeta, eps)),
        np.log10(max(P1, eps)),
    ])
    return ft


def eeg_sleep_psd_feature(PPP, fs):
    """
    Extract 9 sleep features from a PSD matrix (each row = one epoch's PSD).

    Parameters
    ----------
    PPP : (N, n_bins) ndarray — PSD for each 30-s epoch
    fs : sample rate

    Returns
    -------
    ft : (N, 9) ndarray
    """
    n = PPP.shape[0]
    ft = np.zeros((n, 9))
    for i in range(n):
        ft[i, :] = eeg_sleep_psd_feature_single(PPP[i, :], fs)
    return ft


# ---------------------------------------------------------------------------
# 8. Channel selection
# ---------------------------------------------------------------------------

def _select_best_channel(eeg_data):
    """
    Pick the channel with largest MAD (median absolute deviation).
    This tends to pick a live EEG channel rather than a flat/saturated one.
    """
    if eeg_data.ndim == 1:
        return eeg_data.copy(), 0
    mads = np.array([np.median(np.abs(ch - np.median(ch))) for ch in eeg_data])
    idx = int(np.argmax(mads))
    return eeg_data[idx, :].copy(), idx


# ---------------------------------------------------------------------------
# 9. Top-level pipeline: raw EEG -> sleep features
# ---------------------------------------------------------------------------

def eeg_segment_psd(data, fs, n_seg=None, thr=100.0, dthr=45.0):
    """
    Full pipeline: raw EEG -> PSD matrix -> 9-dim sleep features.

    Parameters
    ----------
    data : 1D or 2D ndarray
        1D = single channel; 2D = (n_channels, n_samples).
    fs : original sample rate (Hz)
    n_seg : number of 30-second segments. None = use all full segments.
    thr, dthr : artifact thresholds (uV).

    Returns
    -------
    features : (N_seg, 9) ndarray — sleep features
    Pxx : (N_seg, 100) ndarray — PSD per segment
    pvalues : (N_seg,) ndarray — fraction of clean sub-segments per segment
    """
    # Channel selection
    x, _ch_idx = _select_best_channel(np.atleast_2d(data))

    # Resample to 100 Hz
    target_fs = 100
    if fs != target_fs:
        n_target = int(round(len(x) * target_fs / fs))
        x = _resample(x, n_target)
    fs = target_fs
    seg_len_samples = 30 * fs  # 3000

    # Split into 30-s epochs
    if n_seg is None:
        n_seg = len(x) // seg_len_samples
    x = x[:seg_len_samples * n_seg]
    XX = x.reshape(n_seg, seg_len_samples)

    # Per-epoch PSD
    n_psd_bins = 200 // 2  # 100
    Pxx_all = np.zeros((n_seg, n_psd_bins))
    pvalues = np.zeros(n_seg)

    for i in range(n_seg):
        Pxx_all[i, :], _, pvalues[i], _, _ = eeg_bsi_psd(
            XX[i, :], fs=fs, epoch_len=200, overlap=100, thr=thr, dthr=dthr
        )

    # Extract features
    features = eeg_sleep_psd_feature(Pxx_all, fs)
    return features, Pxx_all, pvalues


# ---------------------------------------------------------------------------
# 10. Feature name labels
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "log10(P2/P1)",
    "log10(Psigma/Pbeta)",
    "log10(Palpha/Pbeta)",
    "log10(Ptheta/Pbeta)",
    "log10(Pdelta/Pbeta)",
    "SEF50",
    "log10(P0)",
    "log10(Pbeta)",
    "log10(P1)",
]

# ---------------------------------------------------------------------------
# 11. Coherence between EEG channel pairs
# ---------------------------------------------------------------------------

MAX_CHANNELS = 6
N_PAIRS = MAX_CHANNELS * (MAX_CHANNELS - 1) // 2      # 15
N_FFT_BINS = 100                                       # 0–50 Hz at 0.5 Hz resolution

# Total feature vector length: 6*9 + 15*100 = 54 + 1500 = 1554
FEATURE_VECTOR_LEN = MAX_CHANNELS * 9 + N_PAIRS * N_FFT_BINS


def eeg_epoch_coherence(
    eeg_30s,
    fs=100,
    thr=100.0,
    dthr=45.0,
    channel_available=None,
):
    """
    Per-channel PSD features + inter-channel coherence for a single 30-s epoch.

    Parameters
    ----------
    eeg_30s : (n_chan, 3000) ndarray
        30-second EEG at 100 Hz. Supports up to 6 channels.
        Channels are assumed to be in order:
        F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1
        If fewer than 6, missing channels are zero-padded and their
        features + coherence entries will be 0.
    fs : float
        Sample rate (default 100 Hz).
    thr, dthr : float
        Artifact rejection thresholds.

    Returns
    -------
    features : (1554,) ndarray
        [0:54]    — 9 PSD features × 6 channels (channel-major order)
        [54:1554] — 100-bin coherence × 15 channel pairs (pair-major, bin-minor)
    """
    eeg_30s = np.atleast_2d(eeg_30s)
    n_chan_orig = min(eeg_30s.shape[0], MAX_CHANNELS)
    n_chan_orig = max(n_chan_orig, 0)
    if channel_available is None:
        channel_available = np.ones(
            n_chan_orig,
            dtype=bool,
        )
    else:
        channel_available = np.asarray(
            channel_available,
            dtype=bool,
        ).reshape(-1)

        if channel_available.shape != (n_chan_orig,):
            raise ValueError(
                "channel_available shape "
                f"{channel_available.shape}; "
                f"expected ({n_chan_orig},)"
            )

    eeg_30s = eeg_30s[:n_chan_orig]

    epoch_len = 200   # 2s sub-segments
    overlap = 100     # 1s step
    n_half = epoch_len // 2  # 100 bins (0–50 Hz)
    n_sub = (eeg_30s.shape[1] - epoch_len) // overlap + 1  # 29

    # Per-channel: PSD, artifact flags, FFT of clean sub-segments.
    PSD_all = np.zeros((MAX_CHANNELS, n_half))
    art_flags = np.ones((MAX_CHANNELS, n_sub), dtype=bool)  # True = artifact
    FXX_all = np.zeros((MAX_CHANNELS, n_sub, n_half), dtype=np.complex128)

    w = np.hamming(epoch_len)
    U = float(w.T @ w)

    if n_chan_orig > 0 and n_sub > 0:
        # (channel, sub-window, sample).  The view avoids copying the raw
        # windows; subsequent detrending creates the working array.
        XX = np.lib.stride_tricks.sliding_window_view(
            eeg_30s, epoch_len, axis=-1,
        )[..., ::overlap, :]
        XX = np.asarray(XX, dtype=np.float64)
        XX = XX - np.mean(XX, axis=-1, keepdims=True)
        XX = _detrend_last_axis(XX)
        XX = XX - np.mean(XX, axis=-1, keepdims=True)

        dn = max(1, int(round(fs / 50)))
        maxdx = np.max(np.abs(XX[..., dn:] - XX[..., :-dn]), axis=-1)
        smooth_b = np.ones(dn) / dn
        smooth = _filtfilt_last_axis(smooth_b, np.array([1.0]), XX)
        smooth = _detrend_last_axis(smooth)
        maxminx = np.max(smooth, axis=-1) - np.min(smooth, axis=-1)
        valid_art_flags = (
            (maxdx > dthr) | (maxminx > thr) | (maxminx < 10)
        )
        art_flags[:n_chan_orig] = valid_art_flags

        spectra = np.fft.fft(XX * w, axis=-1)[..., :n_half]
        spectra = np.where(~valid_art_flags[..., None], spectra, 0.0)
        FXX_all[:n_chan_orig] = spectra

        power = (spectra * spectra.conj()).real
        clean_counts = np.count_nonzero(~valid_art_flags, axis=1)
        power_sum = np.sum(power, axis=1)
        np.divide(
            power_sum * 2.0,
            clean_counts[:, None] * U * fs,
            out=PSD_all[:n_chan_orig],
            where=clean_counts[:, None] > 0,
        )

    # Sub-segments clean in all available original channels.
    available_indices = np.flatnonzero(
        channel_available
    )

    if available_indices.size > 0:
        clean_mask = ~np.any(
            art_flags[available_indices],
            axis=0,
        )
    else:
        clean_mask = np.zeros(
            n_sub,
            dtype=bool,
        )

    # --- PSD features per channel (9 × 6 = 54) ---
    psd_features = np.zeros((MAX_CHANNELS, 9))
    for ch in range(n_chan_orig):
        if not channel_available[ch]:
            continue

        ft = eeg_sleep_psd_feature_single(PSD_all[ch], fs)
        psd_features[ch] = np.nan_to_num(ft, nan=0.0)

    # --- Coherence per frequency bin per pair (100 × 15 = 1500) ---
    coherence_features = np.zeros(N_PAIRS * N_FFT_BINS)

    if clean_mask.any() and n_chan_orig > 1:
        pair_ch1, pair_ch2 = np.triu_indices(MAX_CHANNELS, k=1)
        valid_pairs = (
            (pair_ch1 < n_chan_orig)
            & (pair_ch2 < n_chan_orig)
            & channel_available[pair_ch1]
            & channel_available[pair_ch2]
        )
        Fxx = FXX_all[pair_ch1[valid_pairs]][:, clean_mask, :]
        Fyy = FXX_all[pair_ch2[valid_pairs]][:, clean_mask, :]
        Pxx = (Fxx * Fxx.conj()).mean(axis=1).real
        Pyy = (Fyy * Fyy.conj()).mean(axis=1).real
        Pxy = (Fxx * Fyy.conj()).mean(axis=1)
        Cxy = np.abs(Pxy) ** 2 / np.maximum(Pxx * Pyy, 1e-30)
        coherence_features.reshape(N_PAIRS, N_FFT_BINS)[valid_pairs] = Cxy

    return np.concatenate([psd_features.ravel(), coherence_features])


def eeg_segment_coherence(
    data,
    fs,
    n_seg=None,
    thr=100.0,
    dthr=45.0,
    channel_available=None,
):
    """
    Batch version: process all 30-s epochs of a recording.

    Parameters
    ----------
    data : 1D or 2D ndarray
        Raw EEG. 1D = single channel; 2D = (n_channels, n_samples).
    fs : original sample rate (Hz)
    n_seg : number of 30-s segments. None = use all full segments.
    thr, dthr : artifact thresholds.

    Returns
    -------
    features : (N_seg, 1554) ndarray
        Each row = eeg_epoch_coherence output for one epoch.
    pvalues : (N_seg,) ndarray
        Mean fraction of clean sub-segments across channels per epoch.
    """
    data_2d = np.atleast_2d(data)
    n_channels = min(data_2d.shape[0], MAX_CHANNELS)

    if channel_available is None:
        channel_available = np.ones(n_channels, dtype=bool)
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

    x, _ = _select_best_channel(data_2d)

    target_fs = 100
    if fs != target_fs:
        n_target = int(round(data_2d.shape[1] * target_fs / fs))
        resampled = np.array([_resample(data_2d[ch], n_target) for ch in range(data_2d.shape[0])])
    else:
        resampled = data_2d.copy()

    seg_len_samples = 30 * target_fs
    if n_seg is None:
        n_seg = resampled.shape[1] // seg_len_samples

    features = np.zeros((n_seg, FEATURE_VECTOR_LEN))
    pvalues = np.zeros(n_seg)

    for i in range(n_seg):
        start = i * seg_len_samples
        epoch_data = resampled[:, start:start + seg_len_samples]
        features[i] = eeg_epoch_coherence(
            epoch_data,
            fs=target_fs,
            thr=thr,
            dthr=dthr,
            channel_available=channel_available,
        )

    return features, pvalues
