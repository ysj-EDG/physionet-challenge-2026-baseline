#!/usr/bin/env python
"""Official-entry LSTM training and inference code."""

import gc
import json
import logging
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from helper_code import DEMOGRAPHICS_FILE, HEADERS, find_patients, load_diagnoses
from per_epoch_extractor import PerEpochExtractor

logger = logging.getLogger(__name__)

SEQ_DIM = 465
ECG_DIM = 11
STATIC_DIM = 196
INPUT_DIM = SEQ_DIM + ECG_DIM
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
FC_HIDDEN = 64

BATCH_SIZE = int(os.environ.get("LSTM_BATCH_SIZE", "8"))
LR = float(os.environ.get("LSTM_LR", "1e-3"))
EPOCHS = int(os.environ.get("LSTM_EPOCHS", "80"))
PATIENCE = int(os.environ.get("LSTM_PATIENCE", "15"))
VAL_FRACTION = float(os.environ.get("LSTM_VAL_FRACTION", "0.2"))
RANDOM_STATE = int(os.environ.get("LSTM_RANDOM_STATE", "56"))
THRESHOLD = float(os.environ.get("LSTM_THRESHOLD", "0.5"))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x_seq, x_ecg, x_static, lengths):
        x = torch.cat([x_seq, x_ecg], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = torch.cat([h_n[0], h_n[1]], dim=-1)
        combined = torch.cat([h_last, x_static], dim=-1)
        return self.fc(combined).squeeze(-1)


class PSGDataset(Dataset):
    def __init__(self, records, data_folder, cache_dir, extractor):
        self.records = list(records)
        self.data_folder = data_folder
        self.cache_dir = cache_dir
        self.extractor = extractor
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        rec_key = _record_key(rec)
        cache_path = os.path.join(self.cache_dir, f"{rec_key}.npz")
        fallback_cache_path = os.path.join(
            os.path.dirname(self.cache_dir), "all", f"{rec_key}.npz"
        )

        for candidate_cache_path in (cache_path, fallback_cache_path):
            if os.path.exists(candidate_cache_path):
                data = np.load(candidate_cache_path, allow_pickle=True)
                return build_sample(
                    data["X_seq"], data["X_ecg"], data["x_static"], int(data["y"]), data["mask"]
                )

        try:
            x_seq, x_ecg, x_static, y, mask = self.extractor.extract_all(
                rec, self.data_folder
            )
        except Exception as exc:
            logger.warning("Feature extraction failed for %s: %s", rec_key, exc)
            return {"skip": True, "length": 0}

        if x_seq is None or x_static is None or y not in (0, 1):
            logger.warning("Skipping %s: missing features or label", rec_key)
            return {"skip": True, "length": 0}

        if x_ecg is None:
            x_ecg = np.zeros((0, ECG_DIM), dtype=np.float32)
        if mask is None:
            mask = np.zeros(len(x_seq), dtype=bool)

        try:
            np.savez_compressed(cache_path, X_seq=x_seq, X_ecg=x_ecg, x_static=x_static, y=y, mask=mask)
        except Exception:
            pass

        return build_sample(x_seq, x_ecg, x_static, y, mask)


def _record_key(record):
    patient_id = record.get(HEADERS["bids_folder"], record.get("BidsFolder"))
    session_id = record.get(HEADERS["session_id"], record.get("SessionID"))
    return f"{patient_id}_ses-{session_id}"


def _clean_array(x, shape=None):
    arr = np.asarray(x, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, -50.0, 50.0)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}")
    return arr


def build_sample(x_seq, x_ecg, x_static, y=0, mask=None):
    x_seq = _clean_array(x_seq)
    x_static = _clean_array(x_static, (STATIC_DIM,))
    if x_seq.ndim != 2 or x_seq.shape[1] != SEQ_DIM or len(x_seq) == 0:
        raise ValueError(f"Invalid X_seq shape: {x_seq.shape}")

    if x_ecg is None or len(x_ecg) == 0:
        x_ecg = np.zeros((0, ECG_DIM), dtype=np.float32)
    else:
        x_ecg = _clean_array(x_ecg)
        if x_ecg.ndim != 2 or x_ecg.shape[1] != ECG_DIM:
            raise ValueError(f"Invalid X_ecg shape: {x_ecg.shape}")

    if mask is None:
        mask = np.zeros(len(x_seq), dtype=bool)

    return {
        "X_seq": torch.FloatTensor(x_seq),
        "X_ecg": torch.FloatTensor(x_ecg),
        "x_static": torch.FloatTensor(x_static),
        "y": torch.FloatTensor([float(y)]),
        "mask": torch.BoolTensor(mask),
        "length": len(x_seq),
    }


def collate_fn(batch):
    batch = [b for b in batch if not b.get("skip", False)]
    if not batch:
        return None

    max_len = max(b["length"] for b in batch)
    batch_size = len(batch)
    x_seq = torch.zeros(batch_size, max_len, SEQ_DIM)
    x_ecg = torch.zeros(batch_size, max_len, ECG_DIM)
    x_static = torch.stack([b["x_static"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    lengths = torch.LongTensor([b["length"] for b in batch])

    for i, item in enumerate(batch):
        length = item["length"]
        x_seq[i, :length] = item["X_seq"]
        ecg_len = min(item["X_ecg"].shape[0], max(0, length - 10))
        if ecg_len > 0:
            x_ecg[i, 10 : 10 + ecg_len] = item["X_ecg"][:ecg_len]

    return x_seq, x_ecg, x_static, y, lengths


def _labels_for_records(records, data_folder):
    labels = []
    metadata_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    for rec in records:
        patient_id = rec[HEADERS["bids_folder"]]
        try:
            label = load_diagnoses(metadata_file, patient_id)
        except Exception:
            label = -1
        labels.append(label)
    return np.asarray(labels, dtype=int)


def _train_val_split(records, labels):
    valid = [i for i, y in enumerate(labels) if y in (0, 1)]
    if not valid:
        return [], []

    rng = np.random.default_rng(RANDOM_STATE)
    train_idx = []
    val_idx = []
    for cls in (0, 1):
        cls_idx = np.array([i for i in valid if labels[i] == cls], dtype=int)
        rng.shuffle(cls_idx)
        if len(cls_idx) <= 2:
            train_idx.extend(cls_idx.tolist())
            continue
        n_val = max(1, int(round(len(cls_idx) * VAL_FRACTION)))
        n_val = min(n_val, len(cls_idx) - 1)
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    if not val_idx:
        shuffled = np.array(train_idx, dtype=int)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * VAL_FRACTION)))
        val_idx = shuffled[:n_val].tolist()
        train_idx = shuffled[n_val:].tolist()

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return [records[i] for i in train_idx], [records[i] for i in val_idx]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_items = 0
    for batch in loader:
        if batch is None:
            continue
        x_seq, x_ecg, x_static, y, lengths = batch
        pred = model(x_seq.to(device), x_ecg.to(device), x_static.to(device), lengths.to(device))
        loss = criterion(pred, y.to(device).squeeze(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n_items += len(y)
    return total_loss / n_items if n_items else float("nan")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys = []
    probs = []
    for batch in loader:
        if batch is None:
            continue
        x_seq, x_ecg, x_static, y, lengths = batch
        logits = model(x_seq.to(device), x_ecg.to(device), x_static.to(device), lengths.to(device))
        probs.extend(torch.sigmoid(logits).cpu().numpy().reshape(-1).tolist())
        ys.extend(y.numpy().reshape(-1).tolist())

    y_arr = np.asarray(ys, dtype=int)
    p_arr = np.nan_to_num(np.asarray(probs, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    if len(y_arr) == 0:
        return 0.5, 0.0
    try:
        auroc = float(roc_auc_score(y_arr, p_arr))
    except ValueError:
        auroc = 0.5
    cap = max(1, int(0.05 * len(y_arr)))
    idx = np.argsort(p_arr)[::-1]
    tpr5 = float(np.mean(y_arr[idx[:cap]] == 1))
    return auroc, tpr5


def train_lstm_model(data_folder, model_folder, verbose=False):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    os.makedirs(model_folder, exist_ok=True)

    patient_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    records = find_patients(patient_file)
    labels = _labels_for_records(records, data_folder)
    train_records, val_records = _train_val_split(records, labels)
    if not train_records:
        raise RuntimeError("No labeled training records were found.")
    if not val_records:
        val_records = train_records

    device = get_device()
    extractor = PerEpochExtractor()
    cache_root = os.path.join(model_folder, "lstm_feature_cache")
    train_ds = PSGDataset(train_records, data_folder, os.path.join(cache_root, "train"), extractor)
    val_ds = PSGDataset(val_records, data_folder, os.path.join(cache_root, "val"), extractor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = LSTMModel().to(device)
    train_ids = {id(r) for r in train_records}
    train_labels = np.array([labels[i] for i, r in enumerate(records) if id(r) in train_ids], dtype=int)
    n_pos = int(np.sum(train_labels == 1))
    n_neg = int(np.sum(train_labels == 0))
    pos_weight = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

    best_auroc = -1.0
    best_state = None
    stale_epochs = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_auroc, val_tpr5 = evaluate(model, val_loader, device)
        scheduler.step(val_auroc)
        if verbose:
            print(f"Epoch {epoch:03d}: loss={train_loss:.4f} val_auroc={val_auroc:.4f} val_tpr5={val_tpr5:.4f}")
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    torch.save(
        {
            "state_dict": best_state,
            "threshold": THRESHOLD,
            "val_auroc": best_auroc,
            "config": {
                "seq_dim": SEQ_DIM,
                "ecg_dim": ECG_DIM,
                "static_dim": STATIC_DIM,
                "hidden_dim": HIDDEN_DIM,
                "num_layers": NUM_LAYERS,
            },
        },
        os.path.join(model_folder, "model.pt"),
    )
    with open(os.path.join(model_folder, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"val_auroc": best_auroc, "threshold": THRESHOLD}, f, indent=2)

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_lstm_model(model_folder, verbose=False):
    device = get_device()
    model_path = os.path.join(model_folder, "model.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = LSTMModel().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if verbose:
        print(f"Loaded LSTM model from {model_path}")
    return {
        "model": model,
        "extractor": PerEpochExtractor(),
        "device": device,
        "threshold": float(checkpoint.get("threshold", THRESHOLD)),
    }


@torch.no_grad()
def predict_record(bundle: Dict, record: Dict, data_folder: str):
    model = bundle["model"]
    extractor = bundle["extractor"]
    device = bundle["device"]
    threshold = bundle["threshold"]
    try:
        x_seq, x_ecg, x_static, _, mask = extractor.extract_all(record, data_folder)
        sample = build_sample(x_seq, x_ecg, x_static, 0, mask)
    except Exception as exc:
        logger.warning("LSTM inference fallback for %s: %s", _record_key(record), exc)
        return False, 0.5

    batch = collate_fn([sample])
    if batch is None:
        return False, 0.5
    x_seq, x_ecg, x_static, _, lengths = batch
    logits = model(x_seq.to(device), x_ecg.to(device), x_static.to(device), lengths.to(device))
    prob = float(torch.sigmoid(logits)[0].detach().cpu().item())
    prob = float(np.clip(np.nan_to_num(prob, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0))
    return bool(prob >= threshold), prob
