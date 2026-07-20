#!/usr/bin/env python
"""
Per-30s-epoch 特征提取器 (为 LSTM 提供时序输入)。

输出:
  X_seq: (N_epochs, 483)  — EEG(432) + EMG(24) + Resp(14) + OneHot(13)
  X_ecg: (N_5min_wins, 12) — 滑动5分钟ECG HRV + circadian_cos, stride=30s
  x_static: (196,)          — demographic(10) + algorithmic(186)

时间对齐:
  - 前 5 分钟 (epoch 0-9) 无 ECG 特征
  - 从 epoch 10 开始每个 epoch 对齐一个 ECG 窗口 [t-5min, t]
"""

import logging
import os
import numpy as np

logger = logging.getLogger("per_epoch_extractor")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from .helper_code import (
    load_demographics, load_diagnoses, load_signal_data,
    DEMOGRAPHICS_FILE, PHYSIOLOGICAL_DATA_SUBFOLDER,
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER, HEADERS,
)

from .feature_extractor_demographic import DemographicMixin
from .feature_extractor_algorithmic import AlgorithmicMixin
from .feature_extractor_eeg_coherence import EEGCoherenceMixin
from .feature_extractor_ecg_neurokit import extract_5min_hrv
from .feature_extractor_eeg_bsr import extract_bsr_30s, bsr_feature_names
from .feature_extractor_hrv_circadian_cos import read_edf_start_time, extract_hrv_window_circadian_cos
from .feature_extractor_emg import _preprocess_emg, _extract_emg_epoch, _fill_nan_linear
from .feature_extractor_resp import (
    _preprocess_resp_channel, _extract_resp_epoch, _extract_thorax_abd_epoch,
    _safe_corr,
)
from .feature_extractor_event_onehot import _event_fractions_by_epoch

EPOCH_SEC = 30
EEG_PER_EPOCH_DIM = 54 + 360 + 18   # spectral + coherence + BSR
EMG_PER_EPOCH_DIM = 8 * 3       # chin + lleg + rleg
RESP_PER_EPOCH_DIM = 14
ONEHOT_PER_EPOCH_DIM = 13
SEQ_FEATURE_DIM = EEG_PER_EPOCH_DIM + EMG_PER_EPOCH_DIM + RESP_PER_EPOCH_DIM + ONEHOT_PER_EPOCH_DIM  # 483
ECG_DIM = 12
STATIC_DIM = 10 + 186  # demographic + algorithmic


class PerEpochExtractor(DemographicMixin, AlgorithmicMixin,
                        EEGCoherenceMixin):
    """
    提取 per-30s-epoch 时序特征 + 滑动5分钟 ECG。
    """

    def __init__(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(_SCRIPT_DIR, "channel_table.csv")
        self.csv_path = os.path.abspath(csv_path)
        self._rename_rules_cache = None

    @property
    def _rename_rules(self):
        if self._rename_rules_cache is None:
            self._rename_rules_cache = self._load_rename_rules(self.csv_path)
        return self._rename_rules_cache

    @staticmethod
    def _load_rename_rules(csv_path):
        import pandas as pd
        try:
            channel_table = pd.read_csv(csv_path)
        except FileNotFoundError:
            return {}
        if 'Channel_Names' not in channel_table.columns:
            return {}
        rename_rules = {}
        for _, row in channel_table.iterrows():
            alias_str_raw = row['Channel_Names']
            if pd.isna(alias_str_raw):
                continue
            try:
                alias_list = [a.strip().replace("'", "").replace('"', "")
                              for a in str(alias_str_raw).split(';')]
                alias_list = [a for a in alias_list if a]
                if alias_list:
                    rename_rules[alias_list[0].lower()] = [str(a) for a in alias_list]
            except (ValueError, SyntaxError, TypeError):
                continue
        return rename_rules

    @staticmethod
    def _get_cleaned_name(channel_name):
        cleaned = channel_name.lower()
        cleaned = cleaned.replace('_pds', '').replace('_eg', '')
        cleaned = cleaned.replace(':', '-')
        return cleaned.strip()

    def _standardize_channels(self, columns_original, rename_rules=None):
        """通道名标准化 + 去重."""
        if rename_rules is None:
            rename_rules = self._rename_rules
        cleaned_to_original = {
            self._get_cleaned_name(col): col for col in columns_original
        }
        channel_map = {}
        for std_name, aliases in rename_rules.items():
            for alias in aliases:
                alias_cleaned = self._get_cleaned_name(alias)
                if alias_cleaned in cleaned_to_original:
                    channel_map[std_name] = cleaned_to_original[alias_cleaned]
                    break
        rename_map = {orig_raw: std_name for std_name, orig_raw in channel_map.items()}
        cols_to_drop = []
        for std_name, matched_raw in channel_map.items():
            aliases_cleaned = {self._get_cleaned_name(a)
                               for a in rename_rules.get(std_name, [])}
            for raw_col in columns_original:
                if (self._get_cleaned_name(raw_col) in aliases_cleaned
                        and raw_col != matched_raw):
                    cols_to_drop.append(raw_col)
        cols_to_drop = sorted(set(cols_to_drop))
        pulse_map = {"pulse": "hr", "pr": "hr"}
        for orig_ch_raw in columns_original:
            orig_ch_cleaned = self._get_cleaned_name(orig_ch_raw)
            for orig_ch_alias, new_ch_standard in pulse_map.items():
                if orig_ch_cleaned == orig_ch_alias:
                    if orig_ch_raw not in rename_map:
                        if new_ch_standard not in rename_map.values():
                            rename_map[orig_ch_raw] = new_ch_standard
        return rename_map, cols_to_drop

    @staticmethod
    def _derive_bipolar_signal(ch_a_signal, ref_signal):
        try:
            if isinstance(ref_signal, tuple):
                sig_b, sig_c = ref_signal
                return ch_a_signal - 0.5 * (sig_b + sig_c)
            else:
                return ch_a_signal - ref_signal
        except Exception:
            return None

    # ========================================================================
    # 公有 API
    # ========================================================================

    def extract_all(self, record, data_folder):
        """
        Returns
        -------
        X_seq : (N_epochs, 483) or None
        X_ecg : (N_5min_wins, 12) or None
        x_static : (196,) or None
        y : int
        mask : (N_epochs,) bool — True = 该 epoch 有完整 ECG 对齐
        """
        patient_id = record.get(HEADERS['bids_folder'], record.get('BidsFolder'))
        site_id = record.get(HEADERS['site_id'], record.get('SiteID'))
        session_id = record.get(HEADERS['session_id'], record.get('SessionID'))
        rec_key = f"{patient_id}_ses-{session_id}"

        logger.debug("Extracting per-epoch features for %s (site=%s)", rec_key, site_id)

        # ---- Static features ----
        demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
        demo_data = load_demographics(demo_file, patient_id, session_id)
        demo_feat = self.extract_demographic_features(demo_data).astype(np.float32)

        algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                                 site_id, f"{rec_key}_caisr_annotations.edf")
        if os.path.exists(algo_file):
            algo_data, _ = load_signal_data(algo_file)
            logger.debug("  algo annotations loaded: %d channels", len(algo_data))
        else:
            algo_data = {}
            logger.debug("  no algo annotations found")
        algo_feat = self.extract_all_algorithmic_features(algo_data).astype(np.float32)

        x_static = np.concatenate([demo_feat, algo_feat])

        # ---- Physiological data ----
        phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                                 site_id, f"{rec_key}.edf")
        if not os.path.exists(phys_file):
            logger.warning("Physiological data not found: %s", phys_file)
            return None, None, x_static, load_diagnoses(demo_file, patient_id), None
        edf_start_time = read_edf_start_time(phys_file)
        phys_data, phys_fs = load_signal_data(phys_file)
        logger.debug("  phys channels loaded: %d (%s)", len(phys_data), list(phys_data.keys())[:8])

        y = load_diagnoses(demo_file, patient_id)

        # ---- 统一标准化 + 双极推导 ----
        std_data, std_fs = self._standardize_and_derive_channels(phys_data, phys_fs)

        # ---- Per-epoch EEG (spectral + coherence) ----
        X_eeg = self._extract_per_epoch_eeg(std_data, std_fs)

        # ---- Per-epoch EMG ----
        X_emg = self._extract_per_epoch_emg(std_data, std_fs)

        # ---- Per-epoch Resp ----
        X_resp = self._extract_per_epoch_resp(std_data, std_fs)

        # ---- Per-epoch OneHot ----
        X_onehot = self._extract_per_epoch_onehot(algo_data)

        if X_resp is None and X_eeg is not None and X_emg is not None:
            n_resp_epochs = min(len(X_eeg), len(X_emg))
            X_resp = np.zeros((n_resp_epochs, RESP_PER_EPOCH_DIM), dtype=np.float32)

        if X_eeg is None or X_emg is None or X_resp is None:
            return None, None, x_static, y, None
        if X_onehot is None:
            fallback_epochs = min(X_eeg.shape[0], X_emg.shape[0], X_resp.shape[0])
            X_onehot = np.zeros((fallback_epochs, ONEHOT_PER_EPOCH_DIM), dtype=np.float32)
            logger.debug(
                "  X_onehot: missing algorithmic annotations/stage_caisr; using zeros %s",
                X_onehot.shape,
            )

        # Align to common N_epochs
        n_epochs = min(X_eeg.shape[0], X_emg.shape[0], X_resp.shape[0], X_onehot.shape[0])
        X_seq = np.concatenate([
            X_eeg[:n_epochs], X_emg[:n_epochs], X_resp[:n_epochs], X_onehot[:n_epochs],
        ], axis=1).astype(np.float32)
        X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
        logger.debug("  X_seq: %s (eeg=%s, emg=%s, resp=%s, onehot=%s)",
                     X_seq.shape, X_eeg.shape, X_emg.shape, X_resp.shape, X_onehot.shape)

        # ---- Sliding ECG (5-min windows, 30s stride) ----
        X_ecg = self._extract_sliding_ecg(
            std_data, std_fs, n_epochs, edf_start_time,
        )
        if X_ecg is not None:
            logger.debug("  X_ecg: %s", X_ecg.shape)
        else:
            logger.debug("  X_ecg: None (no ECG channel found)")

        # Mask: epochs with ECG alignment (kept for the existing five-key NPZ format)
        mask = np.zeros(n_epochs, dtype=bool)
        if X_ecg is not None and X_ecg.shape[0] > 0:
            ecg_start_epoch = 10  # first 5-min window ends at epoch 10
            valid_len = min(max(0, n_epochs - ecg_start_epoch), X_ecg.shape[0])
            if valid_len > 0:
                mask[ecg_start_epoch:ecg_start_epoch + valid_len] = True

        return X_seq, X_ecg, x_static, y, mask

    # ========================================================================
    # 统一通道标准化 + 双极推导
    # ========================================================================

    def _standardize_and_derive_channels(self, phys_data, phys_fs):
        """
        1. 通过 channel_table.csv 标准化通道名
        2. 推导缺失的双极导联 (EEG, EOG, Chin EMG)
        3. 重采样到 200Hz
        返回 (std_data, std_fs): 两个 dict {label: signal}, {label: fs}
        """
        from fractions import Fraction
        from scipy.signal import resample_poly
        TARGET_FS = 200.0

        # ---- Step 1: 标准化通道名 ----
        rename_rules = self._rename_rules
        rename_map, cols_to_drop = self._standardize_channels(
            list(phys_data.keys()), rename_rules,
        )

        std_data = {}
        std_fs = {}
        for old_label, sig in phys_data.items():
            new_label = rename_map.get(old_label, old_label.lower().strip())
            if old_label in cols_to_drop:
                continue
            # Resample
            fs = float(phys_fs.get(old_label, TARGET_FS))
            if not np.isclose(fs, TARGET_FS) and fs > 0:
                ratio = Fraction(float(TARGET_FS) / float(fs)).limit_denominator(1000)
                sig = resample_poly(np.asarray(sig, dtype=float),
                                    up=ratio.numerator, down=ratio.denominator)
            else:
                sig = np.asarray(sig, dtype=float)
            std_data[new_label] = sig
            std_fs[new_label] = TARGET_FS

        # ---- Step 2: 双极 EEG 推导 ----
        # 尝试标准名，再尝试单极+参考组合
        bipolar_configs = [
            ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
            ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
            ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ]
        for target, pos, neg_list in bipolar_configs:
            if target in std_data:
                continue
            if pos not in std_data:
                continue
            ref_signals = []
            for neg in neg_list:
                if neg in std_data:
                    ref_signals.append(std_data[neg])
            if not ref_signals:
                continue
            ref = ref_signals[0] if len(ref_signals) == 1 else tuple(ref_signals)
            derived = self._derive_bipolar_signal(std_data[pos], ref)
            if derived is not None:
                std_data[target] = derived
                std_fs[target] = TARGET_FS

        # ---- Step 3: 双极 EOG 推导 ----
        for target, pos, neg in [('e1-m2', 'e1', 'm2'), ('e2-m1', 'e2', 'm1')]:
            if target in std_data:
                continue
            if pos in std_data and neg in std_data:
                derived = self._derive_bipolar_signal(std_data[pos], std_data[neg])
                if derived is not None:
                    std_data[target] = derived
                    std_fs[target] = TARGET_FS

        # ---- Step 4: 双极 Chin EMG 推导 (I0006: chinl - chinr) ----
        if 'chin1-chin2' not in std_data:
            for pair in [('chinl', 'chinr'), ('chin1', 'chin2'),
                          ('chin 1', 'chin 2')]:
                pos, neg = pair
                if pos in std_data and neg in std_data:
                    derived = self._derive_bipolar_signal(std_data[pos], std_data[neg])
                    if derived is not None:
                        std_data['chin1-chin2'] = derived
                        std_fs['chin1-chin2'] = TARGET_FS
                        break

        return std_data, std_fs

    # ========================================================================
    # Per-epoch sub-extractors
    # ========================================================================

    def _extract_per_epoch_eeg(self, std_data, std_fs):
        """返回 (N_epochs, 432): 54 频谱 + 360 相干 + 18 BSR. 输入已标准化+200Hz."""
        from .eeg_sleep_features import eeg_segment_coherence, N_PAIRS, N_FFT_BINS

        EEG_CH = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1']
        eeg_signals = []
        ref_len = None
        for ch in EEG_CH:
            if ch in std_data and std_data[ch] is not None and len(std_data[ch]) > 1:
                eeg_signals.append(np.asarray(std_data[ch], dtype=float))
                ref_len = len(std_data[ch])
            else:
                eeg_signals.append(np.zeros(ref_len or 6000, dtype=float))

        # Handle rare case where EEG channels have mismatched lengths.
        lengths = set(len(s) for s in eeg_signals)
        if len(lengths) > 1:
            min_len = min(lengths)
            eeg_signals = [s[:min_len] for s in eeg_signals]

        eeg_data = np.stack(eeg_signals, axis=0)
        epoch_features, _ = eeg_segment_coherence(eeg_data, 200.0)  # (N_ep, 1554)
        n_epochs = epoch_features.shape[0]
        if n_epochs == 0:
            return None

        # Coherence: vectorized 24 features x 15 pairs for every epoch.
        coh_spectra = epoch_features[:, 54:].reshape(
            n_epochs, N_PAIRS, N_FFT_BINS,
        )
        coh_all = self._extract_coh_features_batch(coh_spectra)

        spectral = epoch_features[:, :54]  # (N_ep, 54)
        coherence = coh_all.reshape(n_epochs, -1)  # (N_ep, 360)
        bsr_features = extract_bsr_30s(eeg_data, n_epochs, fs=200.0)
        result = np.concatenate([spectral, coherence, bsr_features], axis=1).astype(np.float32)
        return np.nan_to_num(result, nan=0.0)

    def _extract_per_epoch_emg(self, std_data, std_fs):
        """返回 (N_epochs, 24): chin(8) + lleg(8) + rleg(8). 输入已标准化+200Hz."""
        chin_sig = std_data.get('chin1-chin2')
        lleg_sig = std_data.get('lat')
        rleg_sig = std_data.get('rat')
        FS = 200.0
        logger.debug("Per-epoch EMG: chin=%s, lleg=%s, rleg=%s",
                     chin_sig is not None, lleg_sig is not None, rleg_sig is not None)

        n_epochs = None
        per_channel = {}
        for name, sig in [("chin", chin_sig), ("lleg", lleg_sig), ("rleg", rleg_sig)]:
            if sig is None:
                continue
            emg_filt, envelope, baseline = _preprocess_emg(sig, FS)
            ep_samples = int(round(30 * FS))
            n_ep = len(sig) // ep_samples
            if n_epochs is None:
                n_epochs = n_ep
            else:
                n_epochs = min(n_epochs, n_ep)

            mode = "chin" if name == "chin" else "leg"
            feats = []
            for ep in range(n_ep):
                s = ep * ep_samples
                f_ep = emg_filt[s:s + ep_samples]
                e_ep = envelope[s:s + ep_samples]
                try:
                    f8 = _extract_emg_epoch(f_ep, e_ep, FS, baseline, mode=mode)
                except Exception:
                    f8 = np.zeros(8, dtype=np.float32)
                feats.append(f8)
            per_channel[name] = np.stack(feats, axis=0)

        if n_epochs is None:
            return None

        result = []
        for name in ["chin", "lleg", "rleg"]:
            if name in per_channel:
                result.append(per_channel[name][:n_epochs])
            else:
                result.append(np.zeros((n_epochs, 8), dtype=np.float32))
        return np.concatenate(result, axis=1).astype(np.float32)

    def _extract_per_epoch_resp(self, std_data, std_fs):
        """返回 (N_epochs, 14). 输入已标准化+200Hz，内部重采样到25Hz."""
        airflow_sig = std_data.get('airflow')
        if airflow_sig is None:
            airflow_sig = std_data.get('ptaf')
        thorax_sig = std_data.get('chest')
        abdomen_sig = std_data.get('abd')

        target_fs = 25
        IN_FS = 200.0
        processed = {}
        if airflow_sig is not None and len(airflow_sig) > 1:
            processed["airflow"] = _preprocess_resp_channel(airflow_sig, IN_FS)
        if thorax_sig is not None and len(thorax_sig) > 1:
            processed["thorax"] = _preprocess_resp_channel(thorax_sig, IN_FS)
        if abdomen_sig is not None and len(abdomen_sig) > 1:
            processed["abdomen"] = _preprocess_resp_channel(abdomen_sig, IN_FS)

        if not processed:
            return None

        if "thorax" in processed and "abdomen" in processed:
            whole_corr = _safe_corr(processed["thorax"]["clean"], processed["abdomen"]["clean"])
            if np.isfinite(whole_corr) and whole_corr < 0:
                processed["abdomen"]["clean"] *= -1

        ep_samples = int(round(30 * target_fs))
        n_epochs = min(len(p["clean"]) // ep_samples for p in processed.values())

        rows = []
        for ep in range(n_epochs):
            s, e = ep * ep_samples, (ep + 1) * ep_samples
            row = []
            for ch in ["airflow", "thorax", "abdomen"]:
                if ch in processed:
                    p = processed[ch]
                    f8 = _extract_resp_epoch(p, s, e, target_fs, ch, p.get("global_lowflow_runs", []))
                    if ch == "airflow":
                        row.extend(f8[:7])
                    else:
                        row.extend([f8[2], f8[4]])
                else:
                    if ch == "airflow":
                        row.extend([0.0] * 7)
                    else:
                        row.extend([0.0, 0.0])
            if "thorax" in processed and "abdomen" in processed:
                ta = _extract_thorax_abd_epoch(processed["thorax"], processed["abdomen"], s, e, target_fs)
                row.extend(ta[:3])
            else:
                row.extend([0.0, 0.0, 0.0])
            rows.append(row)
        return np.nan_to_num(np.asarray(rows, dtype=np.float32), nan=0.0)

    def _extract_per_epoch_onehot(self, algo_data):
        """返回 (N_epochs, 13)."""
        if not algo_data or 'stage_caisr' not in algo_data:
            return None

        raw_stages = np.asarray(algo_data['stage_caisr'], dtype=float).reshape(-1)
        valid = np.isin(raw_stages, [1, 2, 3, 4, 5])
        stages = raw_stages[valid].astype(int)
        n_epochs = len(stages)
        if n_epochs == 0:
            return None

        trt_sec = n_epochs * EPOCH_SEC

        def _segments(signal, dt_sec):
            sig = np.asarray(signal, dtype=float).reshape(-1)
            binary = sig > 0
            edges = np.diff(binary.astype(int), prepend=0, append=0)
            starts = np.where(edges == 1)[0].astype(float) * dt_sec
            ends = np.where(edges == -1)[0].astype(float) * dt_sec
            return starts, ends

        arousal_signal = np.asarray(algo_data.get('arousal_caisr', []), dtype=float).reshape(-1)
        arousal_dt = trt_sec / len(arousal_signal) if len(arousal_signal) > 0 else 0.5
        a_s, a_e = _segments(arousal_signal, arousal_dt)

        resp_signal = np.asarray(algo_data.get('resp_caisr', []), dtype=float).reshape(-1)
        resp_dt = trt_sec / len(resp_signal) if len(resp_signal) > 0 else 1.0
        resp_ss, resp_es = {}, {}
        for rt, rc in [("OA", 1), ("CA", 2), ("MA", 3), ("HY", 4), ("RERA", 5)]:
            s, e = _segments(resp_signal == rc, resp_dt)
            resp_ss[rt], resp_es[rt] = s, e

        limb_signal = np.asarray(algo_data.get('limb_caisr', []), dtype=float).reshape(-1)
        limb_dt = trt_sec / len(limb_signal) if len(limb_signal) > 0 else 1.0
        l_iso_s, l_iso_e = _segments(limb_signal == 1, limb_dt)
        l_plm_s, l_plm_e = _segments(limb_signal == 2, limb_dt)

        stage_codes = np.array([3, 2, 1, 4, 5], dtype=int)
        columns = [(stages[:, None] == stage_codes).astype(np.float32)]
        columns.append(_event_fractions_by_epoch(a_s, a_e, n_epochs)[:, None])
        for rt in ["OA", "CA", "MA", "HY", "RERA"]:
            columns.append(_event_fractions_by_epoch(
                resp_ss[rt], resp_es[rt], n_epochs,
            )[:, None])
        columns.append(_event_fractions_by_epoch(
            l_iso_s, l_iso_e, n_epochs,
        )[:, None])
        columns.append(_event_fractions_by_epoch(
            l_plm_s, l_plm_e, n_epochs,
        )[:, None])
        return np.concatenate(columns, axis=1)

    def _extract_sliding_ecg(self, std_data, std_fs, n_epochs, edf_start_time=None):
        """
        滑动 5 分钟 ECG HRV + 窗口中点 circadian_cos, stride=30s. 输入已标准化+200Hz.
        返回完整 30s 时间网格；坏窗口在原位置将 11 维 HRV 置零。
        """
        ecg_sig = std_data.get('ecg')
        if ecg_sig is None:
            ecg_sig = std_data.get('ekg')
        if ecg_sig is None or len(ecg_sig) < 2:
            logger.debug("No ECG channel found in std_data")
            return None

        FS = 200.0
        window_samples = int(300 * FS)
        stride_samples = int(30 * FS)
        if len(ecg_sig) < window_samples:
            logger.debug(
                "ECG shorter than one complete 5-min window: %.1fs",
                len(ecg_sig) / FS,
            )
            return None
        n_wins = (len(ecg_sig) - window_samples) // stride_samples + 1

        result = np.zeros((n_wins, 12), dtype=np.float32)
        circadian_all = extract_hrv_window_circadian_cos(
            edf_start_time, n_wins, win_sec=300.0, stride_sec=30.0,
        )
        result[:, 11] = circadian_all[:, 0]
        for i in range(n_wins):
            start = i * stride_samples
            seg = ecg_sig[start:start + window_samples]
            if len(seg) != window_samples:
                continue
            f11, _ = extract_5min_hrv(seg, FS)
            if f11 is None:
                continue
            result[i, :11] = np.nan_to_num(
                np.asarray(f11, dtype=np.float32),
                nan=0.0, posinf=0.0, neginf=0.0,
            )

        return result


# ============================================================================
# 特征名
# ============================================================================

def per_epoch_feature_names():
    """返回 per-epoch 483 维特征的名称列表."""
    from .feature_extractor_eeg_coherence import EEGCoherenceMixin, PAIR_NAMES
    from .feature_extractor_emg import EMG_FEATURE_NAMES
    from .feature_extractor_resp import MODEL_COLS as RESP_COLS

    names = []

    # EEG spectral: 6ch × 9 PSD
    psd_names = ["logP2P1", "logPsigmaPbeta", "logPalphaPbeta", "logPthetaPbeta",
                 "logPdeltaPbeta", "SEF50", "logP0", "logPbeta", "logP1"]
    for ch in ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1']:
        for pn in psd_names:
            names.append(f"eeg_{ch}_{pn}")

    # EEG coherence: 15 pair × 24
    pair_stat_names = [
        "mDelta","mTheta","mAlpha","mSigma","mBeta",
        "aucDelta","aucTheta","aucAlpha","aucSigma","aucBeta",
        "iqrDelta","iqrTheta","iqrAlpha","iqrSigma","iqrBeta",
        "sigDelRat","alpDelRat","betDelRat","highLowRat",
        "centroid","entropy","bandwidth","lowsig","highsig",
    ]
    for pn in PAIR_NAMES:
        for sn in pair_stat_names:
            names.append(f"coh_{pn}_{sn}")

    # EEG BSR: 3 thresholds × 6 channels
    names.extend(bsr_feature_names())

    # EMG: 3ch × 8
    for ch in ['chin','lleg','rleg']:
        for fn in EMG_FEATURE_NAMES:
            names.append(f"emg_{ch}_{fn}")

    # Resp: 14
    for cn in RESP_COLS:
        names.append(f"resp_{cn}")

    # OneHot: 13
    for ln in ["stage_N1","stage_N2","stage_N3","stage_REM","stage_Wake",
               "arousal","resp_OA","resp_CA","resp_MA","resp_HY","resp_RERA",
               "limb_isol","limb_PLM"]:
        names.append(f"oh_{ln}")

    return names
