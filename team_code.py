#!/usr/bin/env python
"""Official Challenge training, model loading, and per-patient inference.

The official scripts call only ``train_model``, ``load_model`` and ``run_model``.
The LSTM architecture and inference implementation live in this module.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from helper_code import DEMOGRAPHICS_FILE, HEADERS, load_label
from feature_scaling import FeatureScaler, FeatureScalingError, SCALER_VERSION
from static_logistic import MODEL_TYPE as STATIC_LOGISTIC_MODEL_TYPE
from static_logistic import StaticLogisticModel
from pooled_logistic_v2 import (
    MODEL_TYPE as POOLED_LOGISTIC_V2_MODEL_TYPE,
    apply_imputer_scaler as apply_pooled_imputer_scaler,
    assemble_features as assemble_pooled_features,
)

# ============================================================================
# Configuration
# ============================================================================
# 固定特征维度和网络超参数；必须同时匹配 NPZ 缓存结构与
# train_lstm.py 生成的 checkpoint。

SEQ_FEATURE_DIM = 483
ECG_DIM = 12
STATIC_DIM = 196
INPUT_DIM = SEQ_FEATURE_DIM + ECG_DIM
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
FC_HIDDEN = 64
ECG_EPOCH_OFFSET = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True

# ============================================================================
# Model Architecture
# ============================================================================
# checkpoint 仅保存 state_dict，因此这里是官方推理加载权重时使用的
# 标准模型结构。

class LSTMModel(nn.Module):
    """与本地及官方训练 checkpoint 完全一致的双层 LSTM。"""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            INPUT_DIM,
            HIDDEN_DIM,
            NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
            bidirectional=False,
        )
        self.fc = nn.Sequential(
            nn.Linear(2 * HIDDEN_DIM + STATIC_DIM, FC_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FC_HIDDEN, 1),
        )

    def forward(self, X_seq, X_ecg, x_static, lengths):
        inputs = torch.cat([X_seq, X_ecg], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=True,
        )
        _, (hidden, _) = self.lstm(packed)
        recurrent_features = torch.cat([hidden[0], hidden[1]], dim=-1)
        combined = torch.cat([recurrent_features, x_static], dim=-1)
        return self.fc(combined).squeeze(-1)

# ============================================================================
# Probability Calibration and Single-Patient Inference
# ============================================================================
# 网络输出原始 logit；Platt 校准器将其转换为概率。_infer_one 复现官方
# 单记录推理路径，并把较短的 ECG 序列从第 10 个 epoch 开始对齐。

def _sigmoid(values):
    """使用数值稳定的 sigmoid 将 logits 转换为概率。"""
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def _apply_calibrator(logits, calibrator):
    """应用保存的 Platt 校准器；无校准器时直接使用 sigmoid。"""
    if not calibrator:
        return _sigmoid(logits)
    coefficient = float(calibrator.get("coef", 1.0))
    intercept = float(calibrator.get("intercept", 0.0))
    return _sigmoid(coefficient * np.asarray(logits, dtype=float) + intercept)


@torch.inference_mode()
def _infer_one(network, sample):
    """构建单患者张量、完成 ECG 对齐并返回一个 logit。"""
    length = int(sample["length"])
    X_seq = torch.zeros(1, length, SEQ_FEATURE_DIM, device=DEVICE)
    X_ecg = torch.zeros(1, length, ECG_DIM, device=DEVICE)
    x_static = torch.zeros(1, STATIC_DIM, device=DEVICE)
    lengths = torch.tensor([length], dtype=torch.long, device=DEVICE)

    X_seq[0, :length] = torch.from_numpy(sample["X_seq"]).to(DEVICE)
    ecg_length = min(sample["X_ecg"].shape[0], length - ECG_EPOCH_OFFSET)
    if ecg_length > 0:
        X_ecg[0, ECG_EPOCH_OFFSET:ECG_EPOCH_OFFSET + ecg_length] = torch.from_numpy(
            sample["X_ecg"][:ecg_length]
        ).to(DEVICE)
    x_static[0] = torch.from_numpy(sample["x_static"]).to(DEVICE)

    logit = network(X_seq, X_ecg, x_static, lengths)
    return float(logit.detach().cpu().item())


def _infer_pooled_logistic_v2(checkpoint, sample, sidecar_path):
    """Apply the checkpoint's shared P3 pooling and frozen linear state."""
    required = {
        "stage_code", "stage_valid", "eeg_channel_available", "record_id"
    }
    with np.load(sidecar_path, allow_pickle=False) as sidecar:
        missing = required - set(sidecar.files)
        if missing:
            raise RuntimeError(
                f"P3 sidecar {sidecar_path} is missing keys: {sorted(missing)}"
            )
        saved_record_id = str(np.asarray(sidecar["record_id"]).item())
        if saved_record_id != sample["rec_key"]:
            raise RuntimeError(
                f"P3 sidecar identity mismatch: {saved_record_id!r} != "
                f"{sample['rec_key']!r}"
            )
        stage_code = np.asarray(sidecar["stage_code"])
        stage_valid = np.asarray(sidecar["stage_valid"])
        channel_available = np.asarray(sidecar["eeg_channel_available"])
    if len(stage_code) < sample["length"] or len(stage_valid) < sample["length"]:
        raise RuntimeError(
            f"P3 sidecar {sidecar_path} has fewer epochs than X_seq: "
            f"stage={len(stage_code)}, valid={len(stage_valid)}, "
            f"X_seq={sample['length']}"
        )
    values = assemble_pooled_features(
        sample["x_static"], stage_code, stage_valid, channel_available,
        sample["X_seq"], checkpoint["arm"],
    )
    med = np.asarray(checkpoint["imputer_median"], dtype=np.float64)
    mean = np.asarray(checkpoint["linear_mean"], dtype=np.float64)
    scale = np.asarray(checkpoint["linear_scale"], dtype=np.float64)
    coefficients = np.asarray(checkpoint["coefficients"], dtype=np.float64)
    if not (values.shape == med.shape == mean.shape == scale.shape == coefficients.shape):
        raise RuntimeError(
            "P3 checkpoint feature-state shapes differ: "
            f"values={values.shape}, median={med.shape}, mean={mean.shape}, "
            f"scale={scale.shape}, coefficients={coefficients.shape}"
        )
    standardized = apply_pooled_imputer_scaler(
        values[None, :], med, mean, scale
    )[0]
    logit = float(standardized @ coefficients + float(checkpoint["intercept"]))
    if not np.isfinite(logit):
        raise RuntimeError(f"Non-finite P3 logit for {sample['rec_key']}")
    return logit

# ============================================================================
# Workspace Paths and Record Identity
# ============================================================================
# 记录键负责关联 demographics 行、split JSON 条目和 NPZ 文件名。
# ROOT 为官方封装中的所有仓库内路径提供统一基准。

ROOT = Path(__file__).resolve().parent
SPLIT_NAMES = ("train", "val", "test", "external")
SUBMISSION_SPLITS_DIR = ROOT / "submission_split"
FIXED_TRAIN_RECORDS = 733
FIXED_VALIDATION_RECORDS = 158
FIXED_SELECTED_RECORDS = FIXED_TRAIN_RECORDS + FIXED_VALIDATION_RECORDS
FIXED_INPUT_RECORDS = 1103
FIXED_EXCLUDED_RECORDS = FIXED_INPUT_RECORDS - FIXED_SELECTED_RECORDS
REQUIRED_NPZ_KEYS = frozenset({"X_seq", "X_ecg", "x_static", "y", "mask"})


def _record_key(record):
    """返回缓存 NPZ 使用的标准 BIDS/session 记录键。"""
    return f"{record[HEADERS['bids_folder']]}_ses-{record[HEADERS['session_id']]}"


def _record_identifiers(record):
    """仅保留生成 split JSON 所需的记录标识。"""
    return {
        HEADERS["bids_folder"]: record[HEADERS["bids_folder"]],
        HEADERS["site_id"]: record[HEADERS["site_id"]],
        HEADERS["session_id"]: record[HEADERS["session_id"]],
    }

# ============================================================================
# Feature Validation and Raw-Data Extraction
# ============================================================================
# 缓存样本和实时提取样本先统一执行 dtype 与 shape 验证；非有限值
# 保留到共享 FeatureScaler 按列处理。

def _normalise_arrays(X_seq, X_ecg, x_static, mask):
    """只验证shape并转换dtype；保留非有限值给共享变换器处理。"""
    if X_seq is None:
        raise ValueError("X_seq is None")
    X_seq = np.asarray(X_seq, dtype=np.float32)
    if X_seq.ndim != 2 or len(X_seq) == 0 or X_seq.shape[1] != SEQ_FEATURE_DIM:
        raise ValueError(f"invalid X_seq shape {X_seq.shape}")

    if X_ecg is None:
        X_ecg = np.zeros((0, ECG_DIM), dtype=np.float32)
    else:
        X_ecg = np.asarray(X_ecg, dtype=np.float32)
    if X_ecg.ndim != 2 or X_ecg.shape[1] != ECG_DIM:
        raise ValueError(f"invalid X_ecg shape {X_ecg.shape}")

    if x_static is None:
        raise ValueError("x_static is None")
    x_static = np.asarray(x_static, dtype=np.float32)
    if x_static.shape != (STATIC_DIM,):
        raise ValueError(f"invalid x_static shape {x_static.shape}")

    if mask is None:
        mask = np.zeros(len(X_seq), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(X_seq),):
        raise ValueError(f"invalid mask shape {mask.shape}")

    return X_seq.copy(), X_ecg.copy(), x_static.copy(), mask


@contextmanager
def _label_optional_extraction():
    """在官方隐藏集不提供标签时，允许特征提取继续运行。"""
    import per_epoch_features.per_epoch_extractor as extractor_module

    original = extractor_module.load_diagnoses
    extractor_module.load_diagnoses = lambda *_args, **_kwargs: -1
    try:
        yield
    finally:
        extractor_module.load_diagnoses = original


def _fallback_arrays(record, x_static, reason):
    """Return a model-safe static-only sample for one failed record."""
    key = _record_key(record)
    static = np.zeros(STATIC_DIM, dtype=np.float32)
    if x_static is not None:
        try:
            candidate = np.asarray(x_static, dtype=np.float32)
            if candidate.shape == (STATIC_DIM,):
                static = candidate.copy()
        except (TypeError, ValueError):
            pass
    warnings.warn(
        f"Feature fallback for {key}: {reason}. "
        "Using one zero sequence epoch and available static features.",
        RuntimeWarning,
        stacklevel=2,
    )
    return (
        np.zeros((1, SEQ_FEATURE_DIM), dtype=np.float32),
        np.zeros((0, ECG_DIM), dtype=np.float32),
        static,
        np.zeros(1, dtype=bool),
    )


def _is_fallback_arrays(arrays):
    """Identify the sentinel sample so transient failures are not cached."""
    X_seq, X_ecg, _x_static, mask = arrays
    return (
        X_seq.shape == (1, SEQ_FEATURE_DIM)
        and not np.any(X_seq)
        and X_ecg.shape == (0, ECG_DIM)
        and mask.shape == (1,)
        and not np.any(mask)
    )


def _extract_features(record, data_folder, allow_fallback=False):
    """Extract one record, optionally degrading failed records to static data."""
    from per_epoch_features.per_epoch_extractor import PerEpochExtractor

    try:
        with _label_optional_extraction():
            result = PerEpochExtractor().extract_all(record, str(data_folder))
    except Exception as exc:
        reason = f"extractor raised {type(exc).__name__}: {exc}"
        if allow_fallback:
            return _fallback_arrays(record, None, reason)
        raise RuntimeError(
            f"Feature extraction failed for {_record_key(record)}: {reason}"
        ) from exc

    try:
        X_seq, X_ecg, x_static, _unused_y, mask = result
    except (TypeError, ValueError) as exc:
        reason = f"extractor returned an invalid result: {exc}"
        if allow_fallback:
            return _fallback_arrays(record, None, reason)
        raise RuntimeError(
            f"Feature extraction failed for {_record_key(record)}: {reason}"
        ) from exc

    try:
        return _normalise_arrays(X_seq, X_ecg, x_static, mask)
    except (TypeError, ValueError) as exc:
        reason = str(exc)
        if allow_fallback:
            return _fallback_arrays(record, x_static, reason)
        raise RuntimeError(
            f"Feature extraction failed for {_record_key(record)}: {reason}"
        ) from exc

# ============================================================================
# Official Training Preparation
# ============================================================================
# 正式提交默认读取仓库内 submission_split 的历史固定成员清单。划分只决定
# 记录成员关系；本地 npz_full 的旧目录名不具有当前划分语义。官方环境没有
# 特征池时，也会先确定成员，再仅为选中的记录现场提取。


def _stable_train_val_split(frame, validation_fraction=0.20):
    """实验兼容模式：对输入数据执行确定性的标签分层划分。"""
    train_indices, val_indices = [], []
    labels = frame.apply(lambda row: load_label(row.to_dict()), axis=1)
    for label in sorted(labels.unique()):
        indices = list(frame.index[labels == label])
        indices.sort(
            key=lambda idx: hashlib.sha256(
                _record_key(frame.loc[idx].to_dict()).encode("utf-8")
            ).hexdigest()
        )
        n_val = max(1, int(round(len(indices) * validation_fraction)))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    return frame.loc[train_indices], frame.loc[val_indices]


def _load_submission_manifest(path, expected_count):
    """读取并严格校验只含记录标识的提交 manifest。"""
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read submission manifest {path}: {exc}") from exc
    if not isinstance(records, list):
        raise ValueError(f"Submission manifest must contain a JSON list: {path}")
    if len(records) != expected_count:
        raise ValueError(
            f"Submission manifest {path} has {len(records)} records; "
            f"expected {expected_count}"
        )

    identifier_fields = {
        HEADERS["bids_folder"], HEADERS["site_id"], HEADERS["session_id"]
    }
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != identifier_fields:
            raise ValueError(
                f"Submission manifest {path} record {index} must contain exactly "
                f"{sorted(identifier_fields)}"
            )
    keys = [_record_key(record) for record in records]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError(
            f"Submission manifest {path} contains duplicate record keys: "
            f"{duplicates[:5]}"
        )
    return records


def _fixed_submission_split(frame):
    """按历史 manifest 精确选择 733/158，并返回被排除的其余记录。"""
    if len(frame) != FIXED_INPUT_RECORDS:
        raise ValueError(
            f"Fixed submission split requires {FIXED_INPUT_RECORDS} input records; "
            f"found {len(frame)}"
        )

    rows = frame.to_dict("records")
    input_keys = [_record_key(row) for row in rows]
    duplicate_input_keys = sorted(
        key for key, count in Counter(input_keys).items() if count > 1
    )
    if duplicate_input_keys:
        raise ValueError(
            "demographics.csv contains duplicate record keys: "
            f"{duplicate_input_keys[:5]}"
        )
    rows_by_key = dict(zip(input_keys, rows))

    train_records = _load_submission_manifest(
        SUBMISSION_SPLITS_DIR / "train_records.json", FIXED_TRAIN_RECORDS
    )
    val_records = _load_submission_manifest(
        SUBMISSION_SPLITS_DIR / "val_records.json", FIXED_VALIDATION_RECORDS
    )
    train_keys = [_record_key(record) for record in train_records]
    val_keys = [_record_key(record) for record in val_records]
    overlap = sorted(set(train_keys) & set(val_keys))
    if overlap:
        raise ValueError(
            f"Submission train/validation manifests overlap: {overlap[:5]}"
        )

    selected_keys = train_keys + val_keys
    if len(selected_keys) != FIXED_SELECTED_RECORDS:
        raise ValueError(
            f"Submission manifests select {len(selected_keys)} records; "
            f"expected {FIXED_SELECTED_RECORDS}"
        )
    missing = sorted(set(selected_keys) - set(rows_by_key))
    if missing:
        raise ValueError(
            "Submission manifest records are missing from demographics.csv: "
            f"{missing[:5]} (total={len(missing)})"
        )

    # SiteID is not part of the filename key, so verify it separately rather than
    # silently accepting a mismatched manifest row.
    for manifest_record in train_records + val_records:
        key = _record_key(manifest_record)
        input_record = rows_by_key[key]
        if str(input_record[HEADERS["site_id"]]) != str(
            manifest_record[HEADERS["site_id"]]
        ):
            raise ValueError(
                f"SiteID mismatch for {key}: manifest="
                f"{manifest_record[HEADERS['site_id']]!r}, demographics="
                f"{input_record[HEADERS['site_id']]!r}"
            )

    selected_key_set = set(selected_keys)
    excluded_keys = [key for key in input_keys if key not in selected_key_set]
    if len(excluded_keys) != FIXED_EXCLUDED_RECORDS:
        raise ValueError(
            f"Fixed submission split excluded {len(excluded_keys)} records; "
            f"expected {FIXED_EXCLUDED_RECORDS}"
        )
    train_frame = pd.DataFrame(
        [rows_by_key[key] for key in train_keys], columns=frame.columns
    )
    val_frame = pd.DataFrame(
        [rows_by_key[key] for key in val_keys], columns=frame.columns
    )
    return train_frame, val_frame, excluded_keys, rows_by_key



def _write_records(frame, path):
    """将官方记录标识写入 split JSON。"""
    records = [_record_identifiers(row) for row in frame.to_dict("records")]
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def _index_feature_pool(pool_root):
    """按文件名索引旧四份 NPZ，并拒绝跨目录重复记录。"""
    index = {}
    if pool_root is None:
        return index
    for old_split in SPLIT_NAMES:
        split_dir = pool_root / old_split
        if not split_dir.is_dir():
            continue
        for path in split_dir.glob("*.npz"):
            previous = index.get(path.name)
            if previous is not None:
                raise ValueError(
                    f"Duplicate training NPZ for {path.name}: {previous} and {path}"
                )
            index[path.name] = path
    return index


def _link_cached_feature(source, target):
    """将已有 NPZ 无复制地放入本次训练的临时划分目录。"""
    try:
        target.symlink_to(source.resolve())
    except OSError:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def _prepare_records(
    records, rows_by_key, split, cache_root, data_folder, feature_pool, verbose=False,
):
    """优先从特征池重组一个新划分，仅为缺失记录提取特征。"""
    split_dir = cache_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    reused = extracted = 0
    total = len(records)
    width = len(str(total))
    for index, record in enumerate(records, start=1):
        key = _record_key(record)
        output_path = split_dir / f"{key}.npz"
        source = feature_pool.get(output_path.name)
        if source is not None:
            if verbose:
                print(
                    f"- {index:>{width}}/{total} [{split}] {key}: "
                    f"reusing {source}",
                    flush=True,
                )
            with np.load(source, allow_pickle=False) as cached:
                missing = REQUIRED_NPZ_KEYS - set(cached.files)
                if missing: raise ValueError(f"NPZ {source} missing keys: {sorted(missing)}")
                _normalise_arrays(cached["X_seq"], cached["X_ecg"], cached["x_static"], cached["mask"])
                if int(np.asarray(cached["y"]).reshape(-1)[0]) != int(load_label(rows_by_key[key])):
                    raise ValueError(f"NPZ label mismatch for {key}")
            _link_cached_feature(source, output_path)
            reused += 1
            continue

        row = rows_by_key[key]
        if verbose:
            print(
                f"- {index:>{width}}/{total} [{split}] {key}: extracting...",
                flush=True,
            )
        X_seq, X_ecg, x_static, mask = _extract_features(
            record, data_folder, allow_fallback=True,
        )
        np.savez_compressed(
            output_path,
            X_seq=X_seq,
            X_ecg=X_ecg,
            x_static=x_static,
            y=np.asarray(load_label(row), dtype=np.int64),
            mask=mask,
        )
        if verbose:
            print(
                f"  saved {output_path} X_seq={X_seq.shape} "
                f"X_ecg={X_ecg.shape}",
                flush=True,
            )
        extracted += 1
    return reused, extracted


def _run_unchanged_trainer(data_folder, model_folder, splits_dir, cache_dir, verbose):
    """使用官方封装传入的路径和随机种子启动 train_lstm.py。"""
    env = os.environ.copy()
    env.update(
        {
            "LSTM_DATA_FOLDER": str(data_folder),
            "LSTM_SPLITS_DIR": str(splits_dir),
            "LSTM_CACHE_DIR": str(cache_dir),
            "LSTM_MODEL_DIR": str(model_folder),
            "LSTM_SEED": env.get("LSTM_SEED", "7"),
            "PYTHONHASHSEED": env.get("LSTM_SEED", "7"),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "LSTM_INPUT_PREPROCESSING": env.get("LSTM_INPUT_PREPROCESSING", SCALER_VERSION),
        }
    )
    command = [sys.executable, str(ROOT / "train_lstm.py")]
    if verbose:
        print("Running unchanged trainer:", " ".join(command))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def train_model(data_folder, model_folder, verbose):
    """通过官方 API 训练，同时复用当前 LSTM 训练后端。"""
    data_folder = Path(data_folder).resolve()
    model_folder = Path(model_folder).resolve()
    model_folder.mkdir(parents=True, exist_ok=True)

    demographics_path = data_folder / DEMOGRAPHICS_FILE
    frame = pd.read_csv(demographics_path)
    split_mode = os.environ.get("LSTM_SPLIT_MODE", "fixed_submission")
    if split_mode == "fixed_submission":
        train_frame, val_frame, excluded_keys, rows_by_key = _fixed_submission_split(frame)
    elif split_mode == "stable_80_20":
        train_frame, val_frame = _stable_train_val_split(frame)
        rows_by_key = {_record_key(row): row for row in frame.to_dict("records")}; excluded_keys = []
    else: raise ValueError(f"Unsupported LSTM_SPLIT_MODE: {split_mode}")
    configured_pool = os.environ.get("LSTM_TRAIN_NPZ_CACHE")
    feature_pool_root = Path(configured_pool).resolve() if configured_pool else None
    feature_pool = _index_feature_pool(feature_pool_root)
    if os.environ.get("LSTM_REQUIRE_TRAIN_NPZ") == "1" and (feature_pool_root is None or not feature_pool):
        raise RuntimeError("LSTM_REQUIRE_TRAIN_NPZ=1 requires a non-empty LSTM_TRAIN_NPZ_CACHE")

    with tempfile.TemporaryDirectory(prefix="challenge2026_train_") as temp_name:
        temp_root = Path(temp_name)
        splits_dir, cache_root = temp_root / "split", temp_root / "npz"
        splits_dir.mkdir(parents=True)
        train_records = _write_records(train_frame, splits_dir / "train_records.json")
        val_records = _write_records(val_frame, splits_dir / "val_records.json")
        train_reused, train_extracted = _prepare_records(
            train_records, rows_by_key, "train", cache_root, data_folder, feature_pool,
            verbose,
        )
        val_reused, val_extracted = _prepare_records(
            val_records, rows_by_key, "val", cache_root, data_folder, feature_pool,
            verbose,
        )
        if verbose:
            train_labels = [load_label(r) for r in train_frame.to_dict("records")]
            val_labels = [load_label(r) for r in val_frame.to_dict("records")]
            print(f"Split mode: {split_mode}\nInput records: {len(frame)}\nTrain records: {len(train_records)}\nValidation records: {len(val_records)}\nExcluded records: {len(excluded_keys)}\nTrain labels: negative={train_labels.count(0)}, positive={train_labels.count(1)}\nValidation labels: negative={val_labels.count(0)}, positive={val_labels.count(1)}\nTrain/validation overlap: 0\nNPZ source pool: {feature_pool_root if feature_pool_root else 'none'}\nTrain NPZ: reused={train_reused}, extracted={train_extracted}\nValidation NPZ: reused={val_reused}, extracted={val_extracted}\nInternal test: not used")
        _run_unchanged_trainer(data_folder, model_folder, splits_dir, cache_root, verbose)

# ============================================================================
# Checkpoint Loading
# ============================================================================
# 官方 run_model.py 仅调用一次：验证 checkpoint、重建 LSTM、恢复校准器
# 与阈值，并准备后续所有患者共用的缓存搜索路径。

def load_model(model_folder, verbose):
    """加载 checkpoint，并返回可跨患者复用的全部推理状态。"""
    model_folder = Path(model_folder).resolve()
    model_path = model_folder / "lstm_model.pt"
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {model_path}")

    saved_dims = checkpoint.get("feature_dims")
    expected_dims = {"X_seq": SEQ_FEATURE_DIM, "X_ecg": ECG_DIM, "x_static": STATIC_DIM}
    if saved_dims is not None and saved_dims != expected_dims:
        raise RuntimeError(f"Checkpoint feature dimensions {saved_dims} do not match {expected_dims}")

    preprocessing_state = checkpoint.get("input_preprocessing")
    if preprocessing_state is None:
        if os.environ.get("LSTM_ALLOW_LEGACY_INPUT") != "1":
            raise RuntimeError(
                "Checkpoint has no input_preprocessing state. Set "
                "LSTM_ALLOW_LEGACY_INPUT=1 only for an explicitly legacy checkpoint."
            )
        warnings.warn(
            "Explicit legacy checkpoint compatibility enabled: using raw nan_to_num + clip[-50,50].",
            RuntimeWarning, stacklevel=2,
        )
        preprocessor = FeatureScaler.legacy_clip()
    else:
        try:
            preprocessor = FeatureScaler.from_state_dict(preprocessing_state)
        except FeatureScalingError as exc:
            raise RuntimeError(f"Invalid checkpoint input preprocessing: {exc}") from exc

    model_type = checkpoint.get("model_type", "lstm")
    if model_type == "lstm":
        if "state_dict" not in checkpoint:
            raise RuntimeError(f"LSTM checkpoint is missing state_dict: {model_path}")
        network = LSTMModel()
        network.load_state_dict(checkpoint["state_dict"])
    elif model_type == STATIC_LOGISTIC_MODEL_TYPE:
        network = StaticLogisticModel.from_checkpoint(checkpoint)
    elif model_type == POOLED_LOGISTIC_V2_MODEL_TYPE:
        network = None
    else:
        raise RuntimeError(f"Unsupported checkpoint model_type: {model_type!r}")
    if network is not None:
        network = network.to(DEVICE)
        network.eval()

    p3_sidecar_root = None
    if model_type == POOLED_LOGISTIC_V2_MODEL_TYPE:
        configured_sidecar = os.environ.get("P3_STAGE_SIDECAR")
        if not configured_sidecar:
            raise RuntimeError(
                "P3_STAGE_SIDECAR must point to the frozen P3 stage sidecar"
            )
        p3_sidecar_root = Path(configured_sidecar).resolve()
        if not p3_sidecar_root.is_dir():
            raise FileNotFoundError(p3_sidecar_root)

    cache_roots = []
    configured_cache = os.environ.get("LSTM_NPZ_CACHE")
    if configured_cache:
        cache_roots.append(Path(configured_cache))
    if (ROOT / "npz_full").is_dir():
        cache_roots.append(ROOT / "npz_full")
    cache_roots.append(model_folder / "inference_cache")
    if verbose:
        print(
            f"Loaded {model_path}; model_type={model_type}; "
            f"cache search roots={cache_roots}"
        )
    decision_output_path = None
    configured_decision_output = os.environ.get("LSTM_DECISION_OUTPUT")
    if configured_decision_output:
        decision_output_path = Path(configured_decision_output).resolve()
        if decision_output_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite decision output: {decision_output_path}"
            )
        decision_output_path.parent.mkdir(parents=True, exist_ok=True)
        with decision_output_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    "SiteID", "BidsFolder", "SessionID", "record_key",
                    "decision_logit", "calibrated_probability", "binary_prediction",
                ]
            )
    return {
        "network": network,
        "model_type": model_type,
        "cache_roots": cache_roots,
        "calibrator": checkpoint.get("calibrator"),
        "threshold": float(checkpoint.get("threshold", 0.5)),
        "preprocessor": preprocessor,
        "decision_output_path": decision_output_path,
        "checkpoint": checkpoint,
        "p3_sidecar_root": p3_sidecar_root,
    }

# ============================================================================
# Inference Sample Loading
# ============================================================================
# 推理优先读取预计算 NPZ；未命中时回退到原始数据特征提取，并尝试写入
# 推理缓存。数值裁剪与训练保持一致，避免极端特征破坏 LSTM 稳定性。

def _cached_sample(record, cache_roots):
    """从首个命中该记录的缓存中返回已验证数组。"""
    name = f"{_record_key(record)}.npz"
    for root in cache_roots:
        candidates = [root / name] + [root / split / name for split in SPLIT_NAMES]
        for path in candidates:
            if not path.is_file():
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    X_seq, X_ecg, x_static, mask = _normalise_arrays(
                        data["X_seq"], data["X_ecg"], data["x_static"], data["mask"]
                    )
                return X_seq, X_ecg, x_static, mask
            except Exception as exc:
                warnings.warn(
                    f"Ignoring invalid inference cache {path}: {type(exc).__name__}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return None


def _sample(record, data_folder, cache_roots, preprocessor):
    """构建 _infer_one 使用的标准化样本字典。"""
    arrays = _cached_sample(record, cache_roots)
    if arrays is None:
        arrays = _extract_features(
            record, data_folder, allow_fallback=True,
        )
        if not _is_fallback_arrays(arrays):
            target = cache_roots[-1] / f"{_record_key(record)}.npz"
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    target, X_seq=arrays[0], X_ecg=arrays[1], x_static=arrays[2],
                    y=np.asarray(-1, dtype=np.int64), mask=arrays[3],
                )
            except OSError:
                pass
    fallback_sequence = _is_fallback_arrays(arrays)
    X_seq, X_ecg, x_static, mask = arrays
    X_seq, X_ecg, x_static = preprocessor.transform_arrays(
        X_seq, X_ecg, x_static, record_id=_record_key(record),
        fallback_sequence=fallback_sequence,
    )
    return {
        "X_seq": X_seq,
        "X_ecg": X_ecg,
        "x_static": x_static,
        "y": -1,
        "mask": mask,
        "length": len(X_seq),
        "rec_key": _record_key(record),
    }

# ============================================================================
# Official Per-Patient Inference Entry Point
# ============================================================================
# 未修改的官方 run_model.py 对每位患者调用一次；返回 Challenge 输出表
# 所需的阈值化二分类结果和校准概率。

def run_model(model, record, data_folder, verbose):
    """执行一条官方记录；异常直接抛出，不以 0.5 隐藏失败。"""
    sample = _sample(record, data_folder, model["cache_roots"], model["preprocessor"])
    if model["model_type"] == POOLED_LOGISTIC_V2_MODEL_TYPE:
        sidecar_path = model["p3_sidecar_root"] / f"{sample['rec_key']}.npz"
        if not sidecar_path.is_file():
            raise FileNotFoundError(sidecar_path)
        logit = _infer_pooled_logistic_v2(
            model["checkpoint"], sample, sidecar_path
        )
    else:
        logit = _infer_one(model["network"], sample)
    probability = float(
        _apply_calibrator(np.asarray([logit]), model["calibrator"])[0]
    )
    if not np.isfinite(probability):
        raise RuntimeError(f"Non-finite prediction for {_record_key(record)}")
    probability = float(np.clip(probability, 0.0, 1.0))
    binary_prediction = bool(probability >= model["threshold"])
    decision_output_path = model.get("decision_output_path")
    if decision_output_path is not None:
        with decision_output_path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    record[HEADERS["site_id"]],
                    record[HEADERS["bids_folder"]],
                    record[HEADERS["session_id"]],
                    _record_key(record),
                    format(logit, ".17g"),
                    format(probability, ".17g"),
                    int(binary_prediction),
                ]
            )
    return binary_prediction, probability
