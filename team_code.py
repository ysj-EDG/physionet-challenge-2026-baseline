#!/usr/bin/env python
"""Official Challenge training, model loading, and per-patient inference.

The official scripts call only ``train_model``, ``load_model`` and ``run_model``.
The LSTM architecture and inference implementation live in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from helper_code import DEMOGRAPHICS_FILE, HEADERS, load_label

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

# ============================================================================
# Workspace Paths and Record Identity
# ============================================================================
# 记录键负责关联 demographics 行、split JSON 条目和 NPZ 文件名。
# ROOT 为官方封装中的所有仓库内路径提供统一基准。

ROOT = Path(__file__).resolve().parent
SPLIT_NAMES = ("train", "val", "test", "external")


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
# 缓存样本和实时提取样本在进入网络前统一执行 dtype、shape、非有限值
# 以及隐藏标签兼容处理。

def _normalise_arrays(X_seq, X_ecg, x_static, mask):
    """验证特征形状、转换 dtype，并替换非有限值。"""
    X_seq = np.asarray(X_seq, dtype=np.float32)
    if X_seq.ndim != 2 or len(X_seq) == 0 or X_seq.shape[1] != SEQ_FEATURE_DIM:
        raise ValueError(f"invalid X_seq shape {X_seq.shape}")

    if X_ecg is None:
        X_ecg = np.zeros((0, ECG_DIM), dtype=np.float32)
    else:
        X_ecg = np.asarray(X_ecg, dtype=np.float32)
    if X_ecg.ndim != 2 or X_ecg.shape[1] != ECG_DIM:
        raise ValueError(f"invalid X_ecg shape {X_ecg.shape}")

    x_static = np.asarray(x_static, dtype=np.float32)
    if x_static.shape != (STATIC_DIM,):
        raise ValueError(f"invalid x_static shape {x_static.shape}")

    if mask is None:
        mask = np.zeros(len(X_seq), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(X_seq),):
        raise ValueError(f"invalid mask shape {mask.shape}")

    return (
        np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(X_ecg, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(x_static, nan=0.0, posinf=0.0, neginf=0.0),
        mask,
    )


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


def _extract_features(record, data_folder):
    """NPZ 缓存不存在时，从 Challenge 原始文件提取单条记录。"""
    from per_epoch_features.per_epoch_extractor import PerEpochExtractor

    with _label_optional_extraction():
        X_seq, X_ecg, x_static, _unused_y, mask = PerEpochExtractor().extract_all(
            record, str(data_folder)
        )
    return _normalise_arrays(X_seq, X_ecg, x_static, mask)

# ============================================================================
# Official Training Preparation
# ============================================================================
# 本地和官方训练始终从 -d 指定的 demographics 走同一套稳定哈希划分。
# 仓库内旧 split JSON 不再参与训练；npz_full 的四个子目录只作为按记录名
# 查询的特征池，命中后链接到本次训练所需的新目录结构。


def _stable_train_val_split(frame, validation_fraction=0.20):
    """对官方训练数据执行确定性的标签分层划分。"""
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


def _write_records(frame, path):
    """将官方记录标识写入 split JSON。"""
    records = [_record_identifiers(row) for row in frame.to_dict("records")]
    path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return records


def _index_feature_pool(pool_root):
    """按文件名索引旧四份 NPZ；原目录名不再具有划分语义。"""
    index = {}
    for old_split in SPLIT_NAMES:
        split_dir = pool_root / old_split
        if not split_dir.is_dir():
            continue
        for path in split_dir.glob("*.npz"):
            index.setdefault(path.name, path)
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


def _prepare_records(records, rows_by_key, split, cache_root, data_folder, feature_pool):
    """优先从特征池重组一个新划分，仅为缺失记录提取特征。"""
    split_dir = cache_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    reused = extracted = 0
    for record in records:
        key = _record_key(record)
        output_path = split_dir / f"{key}.npz"
        source = feature_pool.get(output_path.name)
        if source is not None:
            _link_cached_feature(source, output_path)
            reused += 1
            continue

        row = rows_by_key[key]
        X_seq, X_ecg, x_static, mask = _extract_features(record, data_folder)
        np.savez_compressed(
            output_path,
            X_seq=X_seq,
            X_ecg=X_ecg,
            x_static=x_static,
            y=np.asarray(load_label(row), dtype=np.int64),
            mask=mask,
        )
        extracted += 1
    return reused, extracted


def _link_validation_as_test(cache_root, validation_records):
    """将 validation NPZ 复用为训练器内部诊断 test split。"""
    test_dir = cache_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    for record in validation_records:
        name = f"{_record_key(record)}.npz"
        source, target = cache_root / "val" / name, test_dir / name
        _link_cached_feature(source, target)


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
    train_frame, val_frame = _stable_train_val_split(frame)
    rows_by_key = {_record_key(row): row for row in frame.to_dict("records")}
    feature_pool = _index_feature_pool(ROOT / "npz_full")

    with tempfile.TemporaryDirectory(prefix="challenge2026_train_") as temp_name:
        temp_root = Path(temp_name)
        splits_dir, cache_root = temp_root / "split", temp_root / "npz"
        splits_dir.mkdir(parents=True)
        train_records = _write_records(train_frame, splits_dir / "train_records.json")
        val_records = _write_records(val_frame, splits_dir / "val_records.json")
        (splits_dir / "test_records.json").write_text(
            (splits_dir / "val_records.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        train_reused, train_extracted = _prepare_records(
            train_records, rows_by_key, "train", cache_root, data_folder, feature_pool
        )
        val_reused, val_extracted = _prepare_records(
            val_records, rows_by_key, "val", cache_root, data_folder, feature_pool
        )
        _link_validation_as_test(cache_root, val_records)
        if verbose:
            print(
                "Prepared stable 80/20 split: "
                f"train={len(train_records)}, validation={len(val_records)}; "
                f"reused NPZ={train_reused + val_reused}, "
                f"extracted NPZ={train_extracted + val_extracted}"
            )
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
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise RuntimeError(f"Unsupported checkpoint format: {model_path}")

    saved_dims = checkpoint.get("feature_dims")
    expected_dims = {"X_seq": SEQ_FEATURE_DIM, "X_ecg": ECG_DIM, "x_static": STATIC_DIM}
    if saved_dims is not None and saved_dims != expected_dims:
        raise RuntimeError(f"Checkpoint feature dimensions {saved_dims} do not match {expected_dims}")

    network = LSTMModel().to(DEVICE)
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()

    cache_roots = []
    configured_cache = os.environ.get("LSTM_NPZ_CACHE")
    if configured_cache:
        cache_roots.append(Path(configured_cache))
    if (ROOT / "npz_full").is_dir():
        cache_roots.append(ROOT / "npz_full")
    cache_roots.append(model_folder / "inference_cache")
    if verbose:
        print(f"Loaded {model_path}; cache search roots={cache_roots}")
    return {
        "network": network,
        "cache_roots": cache_roots,
        "calibrator": checkpoint.get("calibrator"),
        "threshold": float(checkpoint.get("threshold", 0.5)),
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
            with np.load(path, allow_pickle=False) as data:
                X_seq, X_ecg, x_static, mask = _normalise_arrays(
                    data["X_seq"], data["X_ecg"], data["x_static"], data["mask"]
                )
            return X_seq, X_ecg, x_static, mask
    return None


def _sample(record, data_folder, cache_roots):
    """构建 _infer_one 使用的标准化样本字典。"""
    arrays = _cached_sample(record, cache_roots)
    if arrays is None:
        arrays = _extract_features(record, data_folder)
        target = cache_roots[-1] / f"{_record_key(record)}.npz"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                target,
                X_seq=arrays[0], X_ecg=arrays[1], x_static=arrays[2],
                y=np.asarray(-1, dtype=np.int64), mask=arrays[3],
            )
        except OSError:
            pass
    X_seq, X_ecg, x_static, mask = arrays
    return {
        "X_seq": np.clip(X_seq, -50.0, 50.0),
        "X_ecg": np.clip(X_ecg, -50.0, 50.0),
        "x_static": np.clip(x_static, -50.0, 50.0),
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
    sample = _sample(record, data_folder, model["cache_roots"])
    logit = _infer_one(model["network"], sample)
    probability = float(
        _apply_calibrator(np.asarray([logit]), model["calibrator"])[0]
    )
    if not np.isfinite(probability):
        raise RuntimeError(f"Non-finite prediction for {_record_key(record)}")
    probability = float(np.clip(probability, 0.0, 1.0))
    return bool(probability >= model["threshold"]), probability
