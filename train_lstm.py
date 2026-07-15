#!/usr/bin/env python
"""Official-training LSTM route using on-demand accelerated PSG features."""

import hashlib
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from helper_code import DEMOGRAPHICS_FILE, HEADERS, find_patients, load_demographics, load_label
from per_epoch_features.per_epoch_extractor import (
    ECG_DIM,
    SEQ_FEATURE_DIM,
    STATIC_DIM,
    PerEpochExtractor,
)

logger = logging.getLogger("train_lstm")

SEQ_DIM = SEQ_FEATURE_DIM
INPUT_DIM = SEQ_DIM + ECG_DIM
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
FC_HIDDEN = 64

BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 80
PATIENCE = 15
VAL_FRACTION = 0.20
SPLIT_VERSION = "official-stratified-sha256-v1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def record_key(record):
    return f"{record[HEADERS['bids_folder']]}_ses-{record[HEADERS['session_id']]}"


def load_training_records(demographics_file):
    """Expand official record identifiers with their labeled demographics rows."""
    records = []
    for identifiers in find_patients(str(demographics_file)):
        record = load_demographics(
            str(demographics_file),
            identifiers[HEADERS["bids_folder"]],
            identifiers[HEADERS["session_id"]],
        )
        if not record:
            logger.warning("Skipping record with no demographics row: %s", record_key(identifiers))
            continue
        records.append(record)
    return records


def _normalise_arrays(X_seq, X_ecg, x_static, mask):
    if X_seq is None:
        raise ValueError("X_seq is None")
    X_seq = np.asarray(X_seq, dtype=np.float32)
    if X_seq.ndim != 2 or X_seq.shape[1] != SEQ_DIM or len(X_seq) == 0:
        raise ValueError(f"invalid X_seq shape {X_seq.shape}; expected (N, {SEQ_DIM})")

    if X_ecg is None:
        X_ecg = np.zeros((0, ECG_DIM), dtype=np.float32)
    else:
        X_ecg = np.asarray(X_ecg, dtype=np.float32)
    if X_ecg.ndim != 2 or X_ecg.shape[1] != ECG_DIM:
        raise ValueError(f"invalid X_ecg shape {X_ecg.shape}; expected (M, {ECG_DIM})")

    x_static = np.asarray(x_static, dtype=np.float32)
    if x_static.shape != (STATIC_DIM,):
        raise ValueError(f"invalid x_static shape {x_static.shape}; expected ({STATIC_DIM},)")

    if mask is None:
        mask = np.zeros(len(X_seq), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(X_seq),):
        raise ValueError(f"invalid mask shape {mask.shape}; expected ({len(X_seq)},)")

    return X_seq, X_ecg, x_static, mask


def cache_records(records, cache_dir, data_folder, extractor):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    retained = []
    for index, record in enumerate(records, start=1):
        key = record_key(record)
        path = cache_dir / f"{key}.npz"
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    _normalise_arrays(data["X_seq"], data["X_ecg"], data["x_static"], data["mask"])
                    int(data["y"])
                retained.append(record)
                continue
            except Exception:
                path.unlink(missing_ok=True)

        try:
            label = int(load_label(record))
            X_seq, X_ecg, x_static, _ignored_label, mask = extractor.extract_all(record, str(data_folder))
            X_seq, X_ecg, x_static, mask = _normalise_arrays(X_seq, X_ecg, x_static, mask)
            np.savez_compressed(
                path,
                X_seq=X_seq,
                X_ecg=X_ecg,
                x_static=x_static,
                y=label,
                mask=mask,
            )
            retained.append(record)
        except Exception as exc:
            logger.warning("Skipping %s during feature extraction: %s", key, exc)

        if index % 25 == 0 or index == len(records):
            logger.info(
                "Feature cache %s: %d/%d complete, valid=%d",
                cache_dir.name, index, len(records), len(retained),
            )
    return retained


def split_training_records(records):
    groups = {0: [], 1: []}
    for record in records:
        try:
            label = int(load_label(record))
        except Exception as exc:
            logger.warning("Skipping unlabeled training record %s: %s", record_key(record), exc)
            continue
        if label in groups:
            groups[label].append(record)

    train_records, val_records = [], []
    for label, group in groups.items():
        ordered = sorted(
            group,
            key=lambda record: hashlib.sha256(
                f"{SPLIT_VERSION}|{record_key(record)}".encode("utf-8")
            ).hexdigest(),
        )
        n_val = 0 if len(ordered) <= 1 else max(1, int(round(VAL_FRACTION * len(ordered))))
        n_val = min(n_val, max(0, len(ordered) - 1))
        val_records.extend(ordered[:n_val])
        train_records.extend(ordered[n_val:])
        logger.info("Internal split class=%d train=%d val=%d", label, len(ordered) - n_val, n_val)

    train_records.sort(key=record_key)
    val_records.sort(key=record_key)
    if not train_records or not val_records:
        raise RuntimeError("Could not construct non-empty train and validation splits from labeled records")
    return train_records, val_records



class PSGDataset(Dataset):
    def __init__(self, records, cache_dir):
        self.records = list(records)
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        key = record_key(self.records[index])
        path = self.cache_dir / f"{key}.npz"
        try:
            with np.load(path, allow_pickle=False) as data:
                X_seq, X_ecg, x_static, mask = _normalise_arrays(
                    data["X_seq"], data["X_ecg"], data["x_static"], data["mask"]
                )
                y = int(data["y"])
        except Exception as exc:
            logger.warning("Skipping unreadable cache %s: %s", key, exc)
            return {"skip": True, "length": 0}

        return {
            "X_seq": torch.as_tensor(np.clip(np.nan_to_num(X_seq), -50.0, 50.0), dtype=torch.float32),
            "X_ecg": torch.as_tensor(np.clip(np.nan_to_num(X_ecg), -50.0, 50.0), dtype=torch.float32),
            "x_static": torch.as_tensor(np.clip(np.nan_to_num(x_static), -50.0, 50.0), dtype=torch.float32),
            "y": torch.tensor(float(y), dtype=torch.float32),
            "mask": torch.as_tensor(mask, dtype=torch.bool),
            "length": len(X_seq),
        }


def collate_fn(batch):
    batch = [item for item in batch if not item.get("skip", False)]
    if not batch:
        return None

    max_len = max(item["length"] for item in batch)
    batch_size = len(batch)
    X_seq = torch.zeros(batch_size, max_len, SEQ_DIM)
    X_ecg = torch.zeros(batch_size, max_len, ECG_DIM)
    x_static = torch.stack([item["x_static"] for item in batch])
    y = torch.stack([item["y"] for item in batch])
    mask_ecg = torch.zeros(batch_size, max_len, dtype=torch.bool)
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    for index, item in enumerate(batch):
        length = item["length"]
        X_seq[index, :length] = item["X_seq"]
        ecg_len = min(item["X_ecg"].shape[0], length - 10)
        if ecg_len > 0:
            X_ecg[index, 10:10 + ecg_len] = item["X_ecg"][:ecg_len]
            mask_ecg[index, 10:10 + ecg_len] = True

    return X_seq, X_ecg, mask_ecg, x_static, y, lengths


class LSTMModel(nn.Module):
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
        self._init_weights()

    def _init_weights(self):
        for name, parameter in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
                size = parameter.size(0)
                parameter.data[size // 4:size // 2].fill_(1.0)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, X_seq, X_ecg, mask_ecg, x_static, lengths):
        del mask_ecg
        packed = nn.utils.rnn.pack_padded_sequence(
            torch.cat([X_seq, X_ecg], dim=-1),
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.fc(torch.cat([h_last, x_static], dim=-1)).squeeze(-1)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        if batch is None:
            continue
        X_seq, X_ecg, mask_ecg, x_static, y, lengths = batch
        X_seq, X_ecg, mask_ecg, x_static, y, lengths = [
            tensor.to(device) for tensor in (X_seq, X_ecg, mask_ecg, x_static, y, lengths)
        ]
        logits = model(X_seq, X_ecg, mask_ecg, x_static, lengths)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(y)
        n_samples += len(y)
    return total_loss / n_samples if n_samples else float("nan")


@torch.no_grad()
def collect_outputs(model, loader):
    model.eval()
    labels, logits = [], []
    for batch in loader:
        if batch is None:
            continue
        X_seq, X_ecg, mask_ecg, x_static, y, lengths = batch
        X_seq, X_ecg, mask_ecg, x_static, lengths = [
            tensor.to(device) for tensor in (X_seq, X_ecg, mask_ecg, x_static, lengths)
        ]
        output = model(X_seq, X_ecg, mask_ecg, x_static, lengths)
        labels.extend(y.numpy().tolist())
        logits.extend(output.cpu().numpy().tolist())
    return np.asarray(labels, dtype=int), np.nan_to_num(np.asarray(logits, dtype=float))


def evaluate(model, loader):
    labels, logits = collect_outputs(model, loader)
    if len(labels) == 0:
        return 0.5, 0.0
    try:
        auroc = float(roc_auc_score(labels, logits))
    except ValueError:
        auroc = 0.5
    capacity = max(1, int(0.05 * len(labels)))
    tpr5 = float(np.mean(labels[np.argsort(logits)[::-1][:capacity]] == 1))
    return auroc, tpr5


def build_balanced_sampler(records, cache_dir):
    labels = []
    for record in records:
        with np.load(Path(cache_dir) / f"{record_key(record)}.npz", allow_pickle=False) as data:
            labels.append(int(data["y"]))

    labels = np.asarray(labels, dtype=int)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        logger.warning("Balanced sampler disabled: only one training class remains")
        return None, {int(classes[0]): int(counts[0])} if len(classes) else {}

    count_by_class = {int(label): int(count) for label, count in zip(classes, counts)}
    weights = np.asarray(
        [len(labels) / (len(classes) * count_by_class[int(label)]) for label in labels],
        dtype=np.float64,
    )
    logger.info("Balanced train sampler enabled: class_counts=%s", count_by_class)
    return WeightedRandomSampler(torch.as_tensor(weights), len(weights), replacement=True), count_by_class


def sigmoid_np(values):
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def fit_platt_calibrator(logits, labels):
    labels = np.asarray(labels, dtype=int).ravel()
    logits = np.asarray(logits, dtype=float).reshape(-1, 1)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return {"method": "identity_sigmoid", "coef": 1.0, "intercept": 0.0}
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, labels)
    return {
        "method": "platt_logistic",
        "coef": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
    }


def apply_calibrator(logits, calibrator):
    return sigmoid_np(
        float(calibrator.get("coef", 1.0)) * np.asarray(logits, dtype=float)
        + float(calibrator.get("intercept", 0.0))
    )


def select_youden_threshold(labels, probabilities):
    labels = np.asarray(labels, dtype=int).ravel()
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5, 0.0, 0.0
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    scores = np.where(np.isfinite(thresholds), tpr - fpr, -np.inf)
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(tpr[index]), float(fpr[index])


def main(data_folder=None, model_dir=None, cache_dir=None, verbose=False):
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    data_folder = Path(data_folder or os.environ.get("LSTM_DATA_FOLDER", ".")).resolve()
    model_dir = Path(model_dir or os.environ.get("LSTM_MODEL_DIR", "lstm_model")).resolve()
    cache_dir = Path(cache_dir or os.environ.get("LSTM_CACHE_DIR", model_dir / "lstm_cache")).resolve()

    demographics = data_folder / DEMOGRAPHICS_FILE
    records = load_training_records(demographics)
    if not records:
        raise RuntimeError(f"No training records found in {demographics}")
    train_records, val_records = split_training_records(records)
    logger.info("Official internal split=%s train=%d val=%d", SPLIT_VERSION, len(train_records), len(val_records))
    extractor = PerEpochExtractor()
    train_records = cache_records(train_records, cache_dir / "train", data_folder, extractor)
    val_records = cache_records(val_records, cache_dir / "val", data_folder, extractor)
    if not train_records or not val_records:
        raise RuntimeError("No valid cached records remain after feature extraction")
    split_strategy = SPLIT_VERSION

    train_dataset = PSGDataset(train_records, cache_dir / "train")
    val_dataset = PSGDataset(val_records, cache_dir / "val")
    sampler, train_counts = build_balanced_sampler(train_records, cache_dir / "train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    positives = int(train_counts.get(1, 0))
    negatives = int(train_counts.get(0, 0))
    pos_weight_value = float(negatives / positives) if positives else 1.0

    model = LSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    best_score, best_auroc, best_tpr5, best_epoch = -float("inf"), 0.0, 0.0, 0
    best_state, patience_counter = None, 0
    logger.info("Checkpoint selection: score = val_auroc")

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        loss = train_epoch(model, train_loader, optimizer, criterion)
        val_auroc, val_tpr5 = evaluate(model, val_loader)
        scheduler.step(val_auroc)
        logger.info(
            "Epoch %3d | loss=%.4f | val_auroc=%.4f | val_tpr5=%.4f | select_score=%.4f | time=%.0fs",
            epoch, loss, val_auroc, val_tpr5, val_auroc, time.time() - start,
        )
        if best_state is None or val_auroc > best_score:
            best_score, best_auroc, best_tpr5, best_epoch = val_auroc, val_auroc, val_tpr5, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d (best epoch=%d)", epoch, best_epoch)
                break

    model.load_state_dict(best_state)
    val_y, val_logits = collect_outputs(model, val_loader)
    calibrator = fit_platt_calibrator(val_logits, val_y)
    val_probabilities = apply_calibrator(val_logits, calibrator)
    threshold, threshold_tpr, threshold_fpr = select_youden_threshold(val_y, val_probabilities)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "lstm_model.pt"
    torch.save(
        {
            "state_dict": best_state,
            "auroc": best_auroc,
            "selection_score": best_score,
            "selection_metric": "val_auroc",
            "best_epoch": best_epoch,
            "best_tpr5": best_tpr5,
            "pos_weight": pos_weight_value,
            "calibrator": calibrator,
            "threshold": threshold,
            "threshold_source": "validation_youden_calibrated_probability",
            "feature_dims": {"X_seq": SEQ_DIM, "X_ecg": ECG_DIM, "x_static": STATIC_DIM},
            "split": {
                "strategy": split_strategy,
                "train_records": len(train_records),
                "val_records": len(val_records),
                "validation_fraction": VAL_FRACTION,
            },
            "train_label_counts": {"positive": positives, "negative": negatives},
            "validation": {
                "raw_auroc": best_auroc,
                "raw_tpr5": best_tpr5,
                "threshold_tpr": threshold_tpr,
                "threshold_fpr": threshold_fpr,
                "calibrated_auroc": float(roc_auc_score(val_y, val_probabilities)) if len(np.unique(val_y)) > 1 else 0.5,
            },
        },
        model_path,
    )
    logger.info("Model saved to %s", model_path)
    return model_path


if __name__ == "__main__":
    main(verbose=True)
