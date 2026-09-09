#!/usr/bin/env python
"""
2-Layer LSTM + Static Feature Concatenation.

时序输入: per-30s epoch (EEG 432 + EMG 24 + Resp 14 + OneHot 13) = 483 dims
ECG 输入: 滑动5分钟窗口 12 dims (11 HRV + circadian_cos, 从第5分钟开始对齐)
静态输入: demographic(10) + algorithmic(186) = 196 dims

架构:
    X_seq (T×483) ──┬── LSTM(2层, 128) ──→ h_last (256)
    X_ecg (T′×12) ──┘   (每步拼接495)
    x_static (196) ──────────┘
     → FC(256+196→64) → ReLU → Dropout → FC(64→1) → sigmoid
"""

import json, logging, os, random, subprocess, warnings, time
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

logger = logging.getLogger("train_lstm")

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
DATA_FOLDER = os.environ.get("LSTM_DATA_FOLDER", "data")
SPLITS_DIR = os.environ.get("LSTM_SPLITS_DIR", "split")
MODEL_DIR = os.environ.get(
    "LSTM_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "model"),
)
CACHE_DIR = os.environ.get(
    "LSTM_CACHE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "npz_full"),
)


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from evaluate_model import compute_auroc_age
from feature_scaling import (
    DEFAULT_RULES_PATH, LEGACY_MODE, SCALER_VERSION, FeatureScaler, FeatureScalingError,
    file_sha256, normalize_mask_config,
)

SEQ_DIM = 483
ECG_DIM = 12
STATIC_DIM = 196
INPUT_DIM = SEQ_DIM + ECG_DIM
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
FC_HIDDEN = 64
BATCH_SIZE = 8
LR = 1e-3
EPOCHS = 80
PATIENCE = 15
SEED = int(os.environ.get("LSTM_SEED", "42"))
INPUT_PREPROCESSING = os.environ.get("LSTM_INPUT_PREPROCESSING", SCALER_VERSION)
FEATURE_RULES_PATH = os.environ.get("LSTM_FEATURE_RULES", str(DEFAULT_RULES_PATH))
PREPROCESSOR_STATE_CHECKPOINT = os.environ.get("LSTM_PREPROCESSOR_STATE_CHECKPOINT")
INPUT_CLIP_Z = os.environ.get("LSTM_INPUT_CLIP_Z")
INPUT_CLIP_Z = None if INPUT_CLIP_Z in {None, "", "none", "None"} else float(INPUT_CLIP_Z)
try:
    INPUT_MASK_CONFIG = normalize_mask_config(
        json.loads(os.environ.get("LSTM_INPUT_MASK_CONFIG", "{}"))
    )
except (json.JSONDecodeError, TypeError, ValueError) as exc:
    raise FeatureScalingError(f"Invalid LSTM_INPUT_MASK_CONFIG: {exc}") from exc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    """Seed training and request deterministic CUDA kernels where available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


# ============================================================================
# Dataset
# ============================================================================

class PSGDataset(Dataset):
    def __init__(self, records, cache_dir, extractor, preprocessor):
        self.records = records
        self.cache_dir = cache_dir
        self.extractor = extractor
        self.preprocessor = preprocessor
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        rec_key = f"{rec['BidsFolder']}_ses-{rec['SessionID']}"
        cache_path = os.path.join(self.cache_dir, f"{rec_key}.npz")
        if not os.path.exists(cache_path):
            logger.warning("Cache miss: %s, skipping record", rec_key)
            return {"length": 0, "skip": True}

        data = np.load(cache_path, allow_pickle=True)
        X_seq = data["X_seq"]
        X_ecg = data["X_ecg"]
        x_static = data["x_static"]
        y = int(data["y"])
        mask = data["mask"]
        return self._build_tensors(X_seq, X_ecg, x_static, y, mask, rec_key)

    def _build_tensors(self, X_seq, X_ecg, x_static, y, mask, rec_key="unknown"):
        if X_ecg is None or len(X_ecg) == 0:
            X_ecg = np.zeros((0, ECG_DIM), dtype=np.float32)
        if mask is None:
            mask = np.zeros(len(X_seq), dtype=bool)
        age = float(np.asarray(x_static, dtype=float)[0])
        fallback_sequence = (
            np.asarray(X_seq).shape == (1, SEQ_DIM)
            and np.all(np.asarray(X_seq) == 0)
            and len(X_ecg) == 0
            and not np.any(mask)
        )
        X_seq, X_ecg, x_static = self.preprocessor.transform_arrays(
            X_seq, X_ecg, x_static, record_id=rec_key,
            fallback_sequence=fallback_sequence,
        )
        return {
            "X_seq": torch.FloatTensor(X_seq),
            "X_ecg": torch.FloatTensor(X_ecg),
            "x_static": torch.FloatTensor(x_static),
            "age": torch.tensor(age, dtype=torch.float32),
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
    ages = torch.stack([b["age"] for b in batch])
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

    return X_seq, X_ecg, mask_ecg, x_static, ages, y, lengths


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
        X = torch.cat([X_seq, X_ecg], dim=-1)  # (B, T, 495)

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
        X_seq, X_ecg, mask_ecg, x_static, _ages, y, lengths = batch
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
def collect_outputs(model, loader):
    model.eval()
    all_y, all_pred, all_age = [], [], []
    for batch in loader:
        if batch is None:
            continue
        X_seq, X_ecg, mask_ecg, x_static, ages, y, lengths = batch
        all_age.extend(ages.cpu().numpy().tolist())
        X_seq, X_ecg, x_static = [t.to(device) for t in [X_seq, X_ecg, x_static]]
        pred = model(X_seq, X_ecg, mask_ecg.to(device), x_static, lengths.to(device))
        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(pred.cpu().numpy().tolist())
    y = np.asarray(all_y, dtype=float).ravel()
    p = np.asarray(all_pred, dtype=float).ravel()
    if np.any(~np.isfinite(p)):
        n_nan = int(np.sum(~np.isfinite(p)))
        logger.error("Predictions contain %d non-finite values (NaN/Inf). Check input features.", n_nan)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    ages = np.asarray(all_age, dtype=float).ravel()
    return y, p, ages


@torch.no_grad()
def evaluate(model, loader):
    y, p, ages = collect_outputs(model, loader)
    if len(y) == 0:
        return 0.5, 0.5, 0.0
    try:
        age_auroc = float(compute_auroc_age(y, p, ages, gap=2))
    except (ValueError, ZeroDivisionError):
        age_auroc = 0.5
    try:
        auroc = roc_auc_score(y, p)
    except ValueError:
        auroc = 0.5
    n = len(y)
    cap = max(1, int(0.05 * n))
    idx = np.argsort(p)[::-1]
    tpr5 = float(np.mean(y[idx[:cap]] == 1))
    return age_auroc, auroc, tpr5


def count_cached_labels(records, cache_dir):
    labels = []
    missing = 0
    for rec in records:
        rec_key = f"{rec['BidsFolder']}_ses-{rec['SessionID']}"
        cache_path = os.path.join(cache_dir, f"{rec_key}.npz")
        if not os.path.exists(cache_path):
            missing += 1
            continue
        data = np.load(cache_path, allow_pickle=True)
        labels.append(int(data["y"]))
    labels = np.asarray(labels, dtype=int)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    return n_pos, n_neg, missing


def build_balanced_sampler(records, cache_dir, generator=None):
    labels = []
    missing = 0
    for rec in records:
        rec_key = f"{rec['BidsFolder']}_ses-{rec['SessionID']}"
        cache_path = os.path.join(cache_dir, f"{rec_key}.npz")
        if not os.path.exists(cache_path):
            missing += 1
            labels.append(None)
            continue
        data = np.load(cache_path, allow_pickle=True)
        labels.append(int(data["y"]))

    present = np.asarray([y for y in labels if y is not None], dtype=int)
    if len(present) == 0 or len(np.unique(present)) < 2:
        logger.warning(
            "Balanced sampler disabled: present labels=%d unique_classes=%d missing=%d",
            len(present), len(np.unique(present)) if len(present) else 0, missing,
        )
        return None

    classes = np.unique(present)
    counts = {int(cls): int(np.sum(present == cls)) for cls in classes}
    weights = []
    for y in labels:
        if y is None:
            weights.append(0.0)
        else:
            weights.append(float(len(present) / (len(classes) * counts[int(y)])))

    logger.info(
        "Balanced train sampler enabled: class_counts=%s missing=%d num_samples=%d replacement=True",
        counts, missing, len(records),
    )
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )


def sigmoid_np(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def fit_platt_calibrator(logits, labels):
    labels = np.asarray(labels, dtype=int).ravel()
    logits = np.asarray(logits, dtype=float).reshape(-1, 1)
    if len(np.unique(labels)) < 2:
        logger.warning("Validation split has one class only; using identity sigmoid calibration")
        return {"method": "identity_sigmoid", "coef": 1.0, "intercept": 0.0}

    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(logits, labels)
    coef = float(calibrator.coef_[0, 0])
    intercept = float(calibrator.intercept_[0])
    return {"method": "platt_logistic", "coef": coef, "intercept": intercept}


def apply_calibrator(logits, calibrator):
    coef = float(calibrator.get("coef", 1.0))
    intercept = float(calibrator.get("intercept", 0.0))
    return sigmoid_np(coef * np.asarray(logits, dtype=float) + intercept)


def select_youden_threshold(labels, probabilities):
    labels = np.asarray(labels, dtype=int).ravel()
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    if len(np.unique(labels)) < 2:
        return 0.5, 0.0, 0.0
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    finite = np.isfinite(thresholds)
    if not np.any(finite):
        return 0.5, 0.0, 0.0
    j_scores = np.where(finite, tpr - fpr, -np.inf)
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(fpr[best_idx])


# ============================================================================
# Main
# ============================================================================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    seed_everything(SEED)
    logger.info("Device: %s", device)
    logger.info("Random seed: %d (deterministic algorithms enabled)", SEED)
    logger.info("Data folder: %s", DATA_FOLDER)
    logger.info("Splits dir: %s", SPLITS_DIR)
    logger.info("Cache dir: %s", CACHE_DIR)
    logger.info("Model dir: %s", MODEL_DIR)
    extractor = None

    # Load splits
    def load_records(split):
        path = os.path.join(SPLITS_DIR, f"{split}_records.json")
        records = json.load(open(path))
        logger.info("Loaded %s split: %d records from %s", split, len(records), path)
        return records

    train_recs = load_records("train")
    val_recs = load_records("val")
    external_path = os.path.join(SPLITS_DIR, "external_records.json")
    external_recs = load_records("external") if os.path.exists(external_path) else None
    logger.info(
        "Total: Train=%d, Val=%d, External=%s (internal test is not used)",
        len(train_recs), len(val_recs),
        len(external_recs) if external_recs is not None else "not provided",
    )

    # Fit once on unique raw training caches before weighted sampling, or reuse
    # only a frozen preprocessing state for controlled input ablations.
    train_cache_dir = os.path.join(CACHE_DIR, "train")
    if INPUT_PREPROCESSING == SCALER_VERSION:
        if PREPROCESSOR_STATE_CHECKPOINT:
            source_path = os.path.abspath(PREPROCESSOR_STATE_CHECKPOINT)
            source_checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
            source_state = source_checkpoint.get("input_preprocessing")
            preprocessor = FeatureScaler.from_state_dict(source_state)
            if preprocessor.mode != SCALER_VERSION:
                raise FeatureScalingError(
                    f"Preprocessing source must be {SCALER_VERSION}, got {preprocessor.mode}"
                )
            logger.info(
                "Loaded frozen %s preprocessing state only from %s (sha256=%s); "
                "network/calibrator/threshold were not loaded",
                SCALER_VERSION, source_path, file_sha256(source_path),
            )
        else:
            train_manifest_path = os.path.join(SPLITS_DIR, "train_records.json")
            try:
                code_head = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=os.path.dirname(os.path.abspath(__file__)), text=True,
                ).strip()
            except (OSError, subprocess.SubprocessError):
                code_head = None
            logger.info("Fitting %s from %d train records; rules=%s", SCALER_VERSION, len(train_recs), FEATURE_RULES_PATH)
            preprocessor = FeatureScaler.fit_from_cache(
                train_recs, train_cache_dir, rules_path=FEATURE_RULES_PATH,
                sample_cap=128, seed=SEED, manifest_path=train_manifest_path,
                code_head=code_head,
            )
        preprocessor = preprocessor.with_runtime_config(
            clip_z=INPUT_CLIP_Z,
            mask_config=INPUT_MASK_CONFIG,
        )
        logger.info(
            "Input runtime config: clip_z=%s mask_config=%s",
            preprocessor.clip_z, json.dumps(preprocessor.mask_config, sort_keys=True),
        )
    elif INPUT_PREPROCESSING == LEGACY_MODE:
        logger.warning("Using explicit legacy_clip input preprocessing")
        preprocessor = FeatureScaler.legacy_clip()
    else:
        raise FeatureScalingError(f"Unsupported LSTM_INPUT_PREPROCESSING={INPUT_PREPROCESSING!r}")

    # Datasets
    logger.info("Building datasets (cache dir: %s)...", CACHE_DIR)
    train_ds = PSGDataset(train_recs, train_cache_dir, extractor, preprocessor)
    val_ds = PSGDataset(val_recs, os.path.join(CACHE_DIR, "val"), extractor, preprocessor)
    external_ds = (
        PSGDataset(external_recs, os.path.join(CACHE_DIR, "external"), extractor, preprocessor)
        if external_recs is not None else None
    )
    logger.info(
        "Dataset sizes: Train=%d, Val=%d, External=%s",
        len(train_ds), len(val_ds),
        len(external_ds) if external_ds is not None else "not provided",
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)
    train_sampler = build_balanced_sampler(
        train_recs,
        os.path.join(CACHE_DIR, "train"),
        generator=loader_generator,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=(train_sampler is None), sampler=train_sampler,
                              collate_fn=collate_fn, num_workers=0,
                              generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)
    external_loader = (
        DataLoader(external_ds, batch_size=BATCH_SIZE, shuffle=False,
                   collate_fn=collate_fn, num_workers=0)
        if external_ds is not None else None
    )

    # ---- Feature diagnostic on first batch ----
    logger.info("Running feature diagnostic on first training batch...")
    first_batch = next(iter(train_loader))
    if first_batch is not None:
        X_seq, X_ecg, mask_ecg, x_static, ages, y, lengths = first_batch
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
        logger.info(
            "  age (unclipped metric input): min=%.1f, max=%.1f",
            ages.min().item(), ages.max().item(),
        )
    else:
        logger.warning("First batch is empty — all records in train_loader skipped!")

    # Model
    model = LSTMModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("LSTM model created: %s", type(model).__name__)
    logger.info("Trainable params: %d", n_params)

    train_pos, train_neg, train_missing = count_cached_labels(train_recs, os.path.join(CACHE_DIR, "train"))
    if train_pos <= 0:
        logger.warning("No positive labels found in cached train split; using pos_weight=1.0")
        pos_weight_value = 1.0
    else:
        pos_weight_value = float(train_neg / train_pos)
    logger.info(
        "Train cached labels: pos=%d neg=%d missing=%d pos_weight=%.6f",
        train_pos, train_neg, train_missing, pos_weight_value,
    )

    optimizer = optim.Adam(model.parameters(), lr=LR)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8,
    )

    best_score = -float("inf")
    best_age_auroc = 0.0
    best_auroc = 0.0
    best_tpr5 = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    logger.info("Checkpoint selection: score = val_age_conditioned_auroc (gap=2)")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_age_auroc, val_auroc, val_tpr5 = evaluate(model, val_loader)
        selection_score = float(val_age_auroc)
        scheduler.step(selection_score)
        elapsed = time.time() - t0

        logger.info("Epoch %3d | loss=%.4f | val_age_auroc=%.4f | val_auroc=%.4f | val_tpr5=%.4f | select_score=%.4f | time=%.0fs",
                    epoch, train_loss, val_age_auroc, val_auroc, val_tpr5,
                    selection_score, elapsed)

        if best_state is None or selection_score > best_score:
            best_score = selection_score
            best_age_auroc = val_age_auroc
            best_auroc = val_auroc
            best_tpr5 = val_tpr5
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                logger.info("Early stopping at epoch %d (best epoch=%d age-AUROC=%.4f AUROC=%.4f TPR@5%%=%.4f)",
                            epoch, best_epoch, best_age_auroc, best_auroc, best_tpr5)
                break

    # Load best, calibrate on validation logits, and optionally evaluate external ranking.
    model.load_state_dict(best_state)
    val_y, val_logits, _val_ages = collect_outputs(model, val_loader)
    calibrator = fit_platt_calibrator(val_logits, val_y)
    val_prob_calibrated = apply_calibrator(val_logits, calibrator)
    val_threshold, val_threshold_tpr, val_threshold_fpr = select_youden_threshold(val_y, val_prob_calibrated)
    val_raw_prob = sigmoid_np(val_logits)

    try:
        val_calibrated_auroc = float(roc_auc_score(val_y, val_prob_calibrated))
    except ValueError:
        val_calibrated_auroc = 0.5

    logger.info(
        "Validation calibration: method=%s coef=%.6f intercept=%.6f threshold=%.6f raw_prob_mean=%.6f calibrated_prob_mean=%.6f calibrated_auroc=%.4f",
        calibrator["method"], calibrator["coef"], calibrator["intercept"],
        val_threshold, float(np.mean(val_raw_prob)), float(np.mean(val_prob_calibrated)),
        val_calibrated_auroc,
    )
    logger.info(
        "Validation threshold (Youden): threshold=%.6f tpr=%.4f fpr=%.4f",
        val_threshold, val_threshold_tpr, val_threshold_fpr,
    )

    if external_loader is not None:
        logger.info(
            "External metrics skipped during training; "
            "evaluate saved checkpoint separately with true labels."
        )

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "lstm_model.pt")
    torch.save({
        "state_dict": best_state,
        "seed": SEED,
        "auroc": best_auroc,
        "age_conditioned_auroc": best_age_auroc,
        "selection_score": best_score,
        "selection_metric": "val_age_conditioned_auroc_gap2",
        "best_epoch": best_epoch,
        "best_tpr5": best_tpr5,
        "pos_weight": pos_weight_value,
        "calibrator": calibrator,
        "threshold": val_threshold,
        "threshold_source": "validation_youden_calibrated_probability",
        "validation": {
            "raw_auroc": best_auroc,
            "age_conditioned_auroc": best_age_auroc,
            "raw_tpr5": best_tpr5,
            "selection_score": best_score,
            "selection_metric": "val_age_conditioned_auroc_gap2",
            "best_epoch": best_epoch,
            "calibrated_auroc": val_calibrated_auroc,
            "threshold_tpr": val_threshold_tpr,
            "threshold_fpr": val_threshold_fpr,
            "raw_probability_mean": float(np.mean(val_raw_prob)),
            "calibrated_probability_mean": float(np.mean(val_prob_calibrated)),
        },
        "train_label_counts": {
            "positive": train_pos,
            "negative": train_neg,
            "missing_cache": train_missing,
        },
        "feature_dims": {"X_seq": SEQ_DIM, "X_ecg": ECG_DIM, "x_static": STATIC_DIM},
        "input_preprocessing": preprocessor.state_dict(),
        "input_preprocessing_source": {
            "checkpoint": os.path.abspath(PREPROCESSOR_STATE_CHECKPOINT),
            "sha256": file_sha256(PREPROCESSOR_STATE_CHECKPOINT),
        } if PREPROCESSOR_STATE_CHECKPOINT else None,
    }, model_path)
    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()
