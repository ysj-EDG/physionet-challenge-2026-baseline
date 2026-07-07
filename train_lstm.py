#!/usr/bin/env python
"""
2-Layer LSTM + Static Feature Concatenation.

时序输入: per-30s epoch (EEG 414 + EMG 24 + Resp 14 + OneHot 13) = 465 dims
ECG 输入: 滑动5分钟窗口 11 dims (从第5分钟开始对齐)
静态输入: demographic(10) + algorithmic(186) = 196 dims

架构:
    X_seq (T×465) ──┬── LSTM(2层, 128) ──→ h_last (256)
    X_ecg (T'×11) ──┘   (每步拼接476)
    x_static (196) ──────────┘
     → FC(256+196→64) → ReLU → Dropout → FC(64→1) → sigmoid
"""

import json, logging, os, sys, warnings, time, hashlib
import numpy as np
import joblib
from tqdm import tqdm

warnings.filterwarnings("ignore")

logger = logging.getLogger("train_lstm")

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
DATA_FOLDER = "/database2/physionet2026_kaggle/data"
SPLITS_DIR = os.path.join(DATA_FOLDER, "splits")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lstm_model_kaggle")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lstm_cache_kaggle")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REF_DIR = os.path.join(_SCRIPT_DIR, "ref", "python-example-2026")
sys.path.insert(0, _REF_DIR)
from helper_code import DEMOGRAPHICS_FILE, HEADERS, find_patients

# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

SEQ_DIM = 465
ECG_DIM = 11
STATIC_DIM = 196
INPUT_DIM = SEQ_DIM + ECG_DIM  # 476 per step
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
FC_HIDDEN = 64
BATCH_SIZE = int(os.environ.get("LSTM_BATCH_SIZE", 8))
LR = 1e-3
EPOCHS = int(os.environ.get("LSTM_EPOCHS", 80))
PATIENCE = int(os.environ.get("LSTM_PATIENCE", 15))
RANDOM_SEED = 2026

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Dataset
# ============================================================================

class PSGDataset(Dataset):
    def __init__(self, records, cache_dir, extractor):
        self.records = records
        self.cache_dir = cache_dir
        self.extractor = extractor
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        rec_key = f"{rec['BidsFolder']}_ses-{rec['SessionID']}"
        cache_path = os.path.join(self.cache_dir, f"{rec_key}.npz")
        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            X_seq = data["X_seq"]
            X_ecg = data["X_ecg"]
            x_static = data["x_static"]
            y = int(data["y"])
            mask = data["mask"]
            return self._build_tensors(X_seq, X_ecg, x_static, y, mask)

        logger.debug("Cache miss: %s, extracting...", rec_key)
        X_seq, X_ecg, x_static, y, mask = self.extractor.extract_all(
            rec, DATA_FOLDER)
        if X_seq is None:
            logger.warning("Extraction returned None for %s, skipping record", rec_key)
            # Return sentinel with length=0; collate_fn will filter it out
            return {"length": 0, "skip": True}
        try:
            np.savez_compressed(
                cache_path,
                X_seq=X_seq, X_ecg=X_ecg if X_ecg is not None else np.zeros((0, ECG_DIM)),
                x_static=x_static, y=y,
                mask=mask if mask is not None else np.zeros(len(X_seq), dtype=bool),
            )
        except Exception:
            pass
        return self._build_tensors(X_seq, X_ecg, x_static, y, mask)

    def _build_tensors(self, X_seq, X_ecg, x_static, y, mask):
        if X_ecg is None or len(X_ecg) == 0:
            X_ecg = np.zeros((0, ECG_DIM), dtype=np.float32)
        if mask is None:
            mask = np.zeros(len(X_seq), dtype=bool)
        # Clip extreme values to prevent NaN gradients
        X_seq = np.clip(np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
        X_ecg = np.clip(np.nan_to_num(X_ecg, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
        x_static = np.clip(np.nan_to_num(x_static, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
        return {
            "X_seq": torch.FloatTensor(X_seq),
            "X_ecg": torch.FloatTensor(X_ecg),
            "x_static": torch.FloatTensor(x_static),
            "y": torch.FloatTensor([float(y)]),
            "mask": torch.BoolTensor(mask),
            "length": len(X_seq),
        }


def collate_fn(batch):
    """Pad sequences to batch max length, align ECG. Filters out skipped records."""
    batch = [b for b in batch if not b.get("skip", False)]
    if len(batch) == 0:
        # Return empty batch — caller should skip
        return None

    max_len = max(b["length"] for b in batch)
    bs = len(batch)

    X_seq = torch.zeros(bs, max_len, SEQ_DIM)
    X_ecg = torch.zeros(bs, max_len, ECG_DIM)
    x_static = torch.stack([b["x_static"] for b in batch])
    y = torch.stack([b["y"] for b in batch])
    mask_ecg = torch.zeros(bs, max_len, dtype=torch.bool)
    lengths = torch.LongTensor([b["length"] for b in batch])

    for i, b in enumerate(batch):
        L = b["length"]
        X_seq[i, :L] = b["X_seq"]
        # Align ECG: ecg[t] corresponds to epoch t (t>=10)
        ecg_len = min(b["X_ecg"].shape[0], L - 10)
        if ecg_len > 0:
            X_ecg[i, 10:10 + ecg_len] = b["X_ecg"][:ecg_len]
            mask_ecg[i, 10:10 + ecg_len] = True

    return X_seq, X_ecg, mask_ecg, x_static, y, lengths


# ============================================================================
# Model
# ============================================================================

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            INPUT_DIM, HIDDEN_DIM, NUM_LAYERS,
            batch_first=True, dropout=DROPOUT, bidirectional=False,
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
                # Set forget gate bias to 1 for better long-sequence memory
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, X_seq, X_ecg, mask_ecg, x_static, lengths):
        # Concatenate per-step inputs
        X = torch.cat([X_seq, X_ecg], dim=-1)  # (B, T, 476)

        # Pack for variable-length
        packed = nn.utils.rnn.pack_padded_sequence(
            X, lengths.cpu(), batch_first=True, enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)  # h_n: (2, B, 128)
        h_last = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, 256)

        combined = torch.cat([h_last, x_static], dim=-1)  # (B, 256+196)
        return self.fc(combined).squeeze(-1)


# ============================================================================
# Train / Eval
# ============================================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        if batch is None:
            continue
        X_seq, X_ecg, mask_ecg, x_static, y, lengths = batch
        X_seq, X_ecg, x_static, y = [t.to(device) for t in
            [X_seq, X_ecg, x_static, y]]
        lengths = lengths.to(device)

        pred = model(X_seq, X_ecg, mask_ecg.to(device), x_static, lengths)
        loss = criterion(pred, y.squeeze(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n if n > 0 else float("nan")


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_y, all_pred = [], []
    for batch in loader:
        if batch is None:
            continue
        X_seq, X_ecg, mask_ecg, x_static, y, lengths = batch
        X_seq, X_ecg, x_static = [t.to(device) for t in [X_seq, X_ecg, x_static]]
        pred = model(X_seq, X_ecg, mask_ecg.to(device), x_static, lengths.to(device))
        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
    y, p = np.array(all_y), np.array(all_pred)
    if len(y) == 0:
        return 0.5, 0.0
    if np.any(~np.isfinite(p)):
        n_nan = int(np.sum(~np.isfinite(p)))
        logger.error("Predictions contain %d non-finite values (NaN/Inf). Check input features.", n_nan)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        auroc = roc_auc_score(y, p)
    except ValueError:
        auroc = 0.5
    if not np.isfinite(auroc):
        auroc = 0.5
    n = len(y)
    cap = max(1, int(0.05 * n))
    idx = np.argsort(p)[::-1]
    tpr5 = float(np.mean(y[idx[:cap]] == 1))
    return auroc, tpr5


# ============================================================================
# Main
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from per_epoch_extractor import PerEpochExtractor

    logger.info("Device: %s", device)
    logger.info("Initializing PerEpochExtractor...")
    extractor = PerEpochExtractor()

    # Load records. Official evaluation only provides the Challenge data folder;
    # local split JSON files are optional convenience files, not requirements.
    def load_split_records(split):
        path = os.path.join(SPLITS_DIR, f"{split}_records.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            records = json.load(f)
        logger.info("Loaded %s split: %d records from %s", split, len(records), path)
        return records

    def load_all_records():
        patient_data_file = os.path.join(DATA_FOLDER, DEMOGRAPHICS_FILE)
        records = find_patients(patient_data_file)
        if len(records) == 0:
            raise RuntimeError(f"No records found in {patient_data_file}")
        logger.info("Loaded %d records from %s", len(records), patient_data_file)
        return records

    def make_train_val_split(records):
        records = list(records)
        rng = np.random.default_rng(RANDOM_SEED)
        order = rng.permutation(len(records))
        if len(records) <= 1:
            return records, records
        val_size = max(1, int(round(0.2 * len(records))))
        val_size = min(val_size, len(records) - 1)
        val_idx = set(order[:val_size].tolist())
        train_recs = [rec for i, rec in enumerate(records) if i not in val_idx]
        val_recs = [rec for i, rec in enumerate(records) if i in val_idx]
        return train_recs, val_recs

    train_recs = load_split_records("train")
    val_recs = load_split_records("val")
    test_recs = load_split_records("test")
    if train_recs is None or val_recs is None:
        all_recs = load_all_records()
        train_recs, val_recs = make_train_val_split(all_recs)
    if test_recs is None:
        test_recs = val_recs
    logger.info("Total: Train=%d, Val=%d, Test/Eval=%d", len(train_recs), len(val_recs), len(test_recs))

    # Datasets
    logger.info("Building datasets (cache dir: %s)...", CACHE_DIR)
    train_ds = PSGDataset(train_recs, os.path.join(CACHE_DIR, "train"), extractor)
    val_ds = PSGDataset(val_recs, os.path.join(CACHE_DIR, "val"), extractor)
    test_ds = PSGDataset(test_recs, os.path.join(CACHE_DIR, "test"), extractor)
    logger.info("Dataset sizes: Train=%d, Val=%d, Test=%d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)

    # ---- Feature diagnostic on first batch ----
    logger.info("Running feature diagnostic on first training batch...")
    first_batch = next(iter(train_loader))
    if first_batch is not None:
        X_seq, X_ecg, mask_ecg, x_static, y, lengths = first_batch
        for name, t in [("X_seq", X_seq), ("X_ecg", X_ecg), ("x_static", x_static)]:
            t_flat = t.reshape(-1)
            n_nan = torch.isnan(t_flat).sum().item()
            n_inf = torch.isinf(t_flat).sum().item()
            t_fin = t_flat[torch.isfinite(t_flat)]
            if len(t_fin) > 0:
                logger.info("  %s: shape=%s, nan=%d, inf=%d, min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                            name, tuple(t.shape), n_nan, n_inf,
                            t_fin.min().item(), t_fin.max().item(),
                            t_fin.mean().item(), t_fin.std().item())
            else:
                logger.warning("  %s: shape=%s, ALL NON-FINITE!", name, tuple(t.shape))
        logger.info("  y: %d positives out of %d", int(y.sum().item()), len(y))
    else:
        logger.warning("First batch is empty — all records in train_loader skipped!")

    # Model
    model = LSTMModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("LSTM model created: %s", type(model).__name__)
    logger.info("Trainable params: %d", n_params)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8,
    )

    best_auroc = float("-inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_auroc, val_tpr5 = evaluate(model, val_loader)
        scheduler.step(val_auroc)
        elapsed = time.time() - t0

        logger.info("Epoch %3d | loss=%.4f | val_auroc=%.4f | val_tpr5=%.4f | time=%.0fs",
                    epoch, train_loss, val_auroc, val_tpr5, elapsed)

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d (best AUROC=%.4f)", epoch, best_auroc)
                break

    # Load best & evaluate test
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_auroc = 0.5
    model.load_state_dict(best_state)
    test_auroc, test_tpr5 = evaluate(model, test_loader)
    logger.info("Test: AUROC=%.4f, TPR@5%%=%.4f", test_auroc, test_tpr5)

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lstm_model.pt")
    torch.save({"state_dict": best_state, "auroc": best_auroc}, model_path)
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()
