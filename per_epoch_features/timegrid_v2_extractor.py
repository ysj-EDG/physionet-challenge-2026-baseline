"""Versioned, position-preserving data2 feature extraction."""
from __future__ import annotations

import os
import numpy as np

from .helper_code import (
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER, DEMOGRAPHICS_FILE, HEADERS,
    PHYSIOLOGICAL_DATA_SUBFOLDER, load_demographics, load_diagnoses,
    load_signal_data,
)
from .per_epoch_extractor import (
    ECG_DIM, EEG_PER_EPOCH_DIM, EMG_PER_EPOCH_DIM, EPOCH_SEC,
    ONEHOT_PER_EPOCH_DIM, RESP_PER_EPOCH_DIM, SEQ_FEATURE_DIM, STATIC_DIM,
    PerEpochExtractor,
)
from .feature_extractor_hrv_circadian_cos import read_edf_start_time

EXTRACTION_VERSION = "timegrid_v2.0.0"


class TimegridV2Extractor(PerEpochExtractor):
    """Explicit v2 entry point. The inherited legacy API remains unchanged."""

    extraction_version = EXTRACTION_VERSION

    def extract_all(self, record, data_folder, return_metadata=False):
        patient_id = record.get(HEADERS["bids_folder"], record.get("BidsFolder"))
        site_id = record.get(HEADERS["site_id"], record.get("SiteID"))
        session_id = record.get(HEADERS["session_id"], record.get("SessionID"))
        record_id = f"{patient_id}_ses-{session_id}"
        demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
        demo_data = load_demographics(demo_file, patient_id, session_id)
        demo_feat = self.extract_demographic_features(demo_data).astype(np.float32)
        label = int(load_diagnoses(demo_file, patient_id))

        phys_file = os.path.join(
            data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, str(site_id),
            f"{record_id}.edf",
        )
        if not os.path.exists(phys_file):
            return self._failed(demo_feat, label, record_id, site_id,
                                "physiological_edf_missing", return_metadata)

        edf_start_time = read_edf_start_time(phys_file)
        phys_data, phys_fs, phys_meta = load_signal_data(
            phys_file, return_metadata=True,
        )
        std_data, std_fs = self._standardize_and_derive_channels(phys_data, phys_fs)
        eeg_result = self._extract_per_epoch_eeg(std_data, std_fs, return_metadata=True)
        if eeg_result is None:
            return self._failed(demo_feat, label, record_id, site_id,
                                "eeg_unavailable", return_metadata)
        x_eeg, eeg_meta = eeg_result
        x_emg = self._extract_per_epoch_emg(std_data, std_fs)
        x_resp = self._extract_per_epoch_resp(std_data, std_fs)

        # Missing optional modalities receive full-length fixed-width blocks;
        # placeholders never determine the recording length.
        n_phys = len(x_eeg)
        if x_emg is not None:
            n_phys = min(n_phys, len(x_emg))
        if x_resp is not None:
            n_phys = min(n_phys, len(x_resp))
        if n_phys <= 0:
            return self._failed(demo_feat, label, record_id, site_id,
                                "no_complete_physiological_epoch", return_metadata)
        if x_emg is None:
            x_emg = np.zeros((n_phys, EMG_PER_EPOCH_DIM), dtype=np.float32)
        if x_resp is None:
            x_resp = np.zeros((n_phys, RESP_PER_EPOCH_DIM), dtype=np.float32)

        algo_file = os.path.join(
            data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, str(site_id),
            f"{record_id}_caisr_annotations.edf",
        )
        if os.path.exists(algo_file):
            algo_data, algo_fs, algo_meta = load_signal_data(
                algo_file, return_metadata=True,
            )
            annotation_status = "available"
        else:
            algo_data, algo_fs = {}, {}
            algo_meta = {"duration_sec": np.nan, "startdate": None,
                         "starttime": None, "signal_lengths": {}}
            annotation_status = "missing"

        raw_stage = np.asarray(algo_data.get("stage_caisr", []), dtype=float).reshape(-1)
        aligned_stage = np.full(n_phys, np.nan, dtype=float)
        aligned_stage[:min(n_phys, len(raw_stage))] = raw_stage[:n_phys]
        algo_data["__extraction_mode__"] = "timegrid_v2"
        algo_data["__sampling_frequencies__"] = algo_fs
        algo_data["__aligned_stage_code__"] = aligned_stage

        algo_feat = self.extract_all_algorithmic_features(algo_data).astype(np.float32)
        x_static = np.concatenate([demo_feat, algo_feat]).astype(np.float32)
        onehot, stage_meta = self._extract_per_epoch_onehot(
            algo_data, n_epochs=n_phys, return_metadata=True,
        )
        if onehot is None:
            onehot = np.zeros((n_phys, ONEHOT_PER_EPOCH_DIM), dtype=np.float32)
            stage_meta = {
                "stage_code_raw": raw_stage,
                "stage_code_aligned": aligned_stage,
                "stage_valid": np.zeros(n_phys, dtype=bool),
            }
        x_seq = np.concatenate([
            x_eeg[:n_phys], x_emg[:n_phys], x_resp[:n_phys], onehot,
        ], axis=1).astype(np.float32)
        x_seq = np.nan_to_num(x_seq, nan=0.0, posinf=0.0, neginf=0.0)

        ecg_result = self._extract_sliding_ecg(
            std_data, std_fs, n_phys, edf_start_time, return_metadata=True,
        )
        if ecg_result is None:
            x_ecg = np.zeros((0, ECG_DIM), dtype=np.float32)
            hrv_success = np.zeros(0, dtype=bool)
        else:
            x_ecg, ecg_meta = ecg_result
            hrv_success = np.asarray(ecg_meta["hrv_success"], dtype=bool)
        mask = np.zeros(n_phys, dtype=bool)
        aligned_ecg = min(max(0, n_phys - 10), len(x_ecg))
        mask[10:10 + aligned_ecg] = True

        stage_probability, stage_probability_valid, prob_present = self._stage_probabilities(
            algo_data, n_phys,
        )
        clean = np.asarray(eeg_meta["clean_subsegment_count"], dtype=np.int32)[:n_phys, :6]
        total = np.asarray(eeg_meta["total_subsegment_count"], dtype=np.int32)[:n_phys, :6]
        metadata = {
            "extraction_version": EXTRACTION_VERSION,
            "record_id": record_id,
            "bids_folder": str(patient_id),
            "session_id": str(session_id),
            "site_id": str(site_id),
            "bdsp_patient_id": str(demo_data.get(HEADERS["patient_id"], "")),
            "epoch_start_sec": np.arange(n_phys, dtype=np.float32) * EPOCH_SEC,
            "epoch_duration_sec": np.full(n_phys, EPOCH_SEC, dtype=np.float32),
            "edf_time_offset_sec": np.float32(np.nan),
            "alignment_status": "official_relative_origin;header_offset_unavailable",
            "stage_code_raw": np.asarray(stage_meta["stage_code_raw"], dtype=np.float32),
            "stage_code_aligned": np.asarray(stage_meta["stage_code_aligned"], dtype=np.float32),
            "stage_valid": np.asarray(stage_meta["stage_valid"], dtype=bool),
            "stage_probability": stage_probability,
            "stage_probability_valid": stage_probability_valid,
            "stage_probability_channels_present": prob_present,
            "eeg_channel_available": np.asarray(eeg_meta["channel_available"], dtype=bool)[:6],
            "eeg_clean_subsegment_count": clean,
            "eeg_total_subsegment_count": total,
            "hrv_success": hrv_success,
            "annotation_status": annotation_status,
            "annotation_sampling_frequencies": algo_fs,
            "annotation_signal_lengths": algo_meta.get("signal_lengths", {}),
            "physiological_signal_lengths": phys_meta.get("signal_lengths", {}),
            "physiological_duration_sec": float(phys_meta["duration_sec"]),
            "annotation_duration_sec": float(algo_meta.get("duration_sec", np.nan)),
            "physiological_startdate": phys_meta.get("startdate"),
            "physiological_starttime": phys_meta.get("starttime"),
            "annotation_startdate": algo_meta.get("startdate"),
            "annotation_starttime": algo_meta.get("starttime"),
            "sequence_lengths": np.asarray(
                [len(x_eeg), len(x_emg), len(x_resp), len(onehot)], dtype=np.int32,
            ),
        }
        self._validate_core(x_seq, x_ecg, x_static, label, mask)
        result = (x_seq, x_ecg, x_static, label, mask)
        return (*result, metadata) if return_metadata else result

    @staticmethod
    def _stage_probabilities(algo_data, n_epochs):
        aliases = [
            ("caisr_prob_n1",), ("caisr_prob_n2",), ("caisr_prob_n3",),
            ("caisr_prob_r", "caisr_prob_rem"),
            ("caisr_prob_w", "caisr_prob_wake"),
        ]
        values = np.full((n_epochs, 5), np.nan, dtype=np.float32)
        present = np.zeros(5, dtype=bool)
        for column, keys in enumerate(aliases):
            for key in keys:
                if key in algo_data and len(algo_data[key]):
                    arr = np.asarray(algo_data[key], dtype=float).reshape(-1)
                    values[:min(n_epochs, len(arr)), column] = arr[:n_epochs]
                    present[column] = True
                    break
        return values, np.all(np.isfinite(values), axis=1), present

    @staticmethod
    def _validate_core(x_seq, x_ecg, x_static, label, mask):
        if x_seq.ndim != 2 or x_seq.shape[1] != SEQ_FEATURE_DIM:
            raise ValueError(f"invalid X_seq shape: {x_seq.shape}")
        if x_ecg.ndim != 2 or x_ecg.shape[1] != ECG_DIM:
            raise ValueError(f"invalid X_ecg shape: {x_ecg.shape}")
        if x_static.shape != (STATIC_DIM,):
            raise ValueError(f"invalid x_static shape: {x_static.shape}")
        if label not in (0, 1) or mask.shape != (len(x_seq),):
            raise ValueError("invalid label or mask")
        if not all(np.all(np.isfinite(x)) for x in (x_seq, x_ecg, x_static)):
            raise ValueError("non-finite core feature output")

    @staticmethod
    def _failed(demo_feat, label, record_id, site_id, status, return_metadata):
        result = (None, None, np.concatenate([
            demo_feat, np.zeros(186, dtype=np.float32),
        ]), label, None)
        metadata = {"extraction_version": EXTRACTION_VERSION,
                    "record_id": record_id, "site_id": str(site_id),
                    "physiological_status": status}
        return (*result, metadata) if return_metadata else result
