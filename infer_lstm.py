#!/usr/bin/env python
"""
LSTM 推理脚本 — 预加载所有特征到内存，按长度排序批量推理。

输出文件:
  - lstm_test_predictions.csv  按 Challenge 格式的预测结果
  - lstm_test_metrics.txt      所有测试指标

用法:
    python infer_lstm.py
    python infer_lstm.py --data_folder /path/to/training_set
    python infer_lstm.py --batch_size 256 --output_dir ./results
"""

import argparse, json, logging, os, sys, time, warnings, gc
import numpy as np
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm

warnings.filterwarnings("ignore")

logger = logging.getLogger("infer_lstm")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REF_DIR = os.path.join(_SCRIPT_DIR, "ref", "python-example-2026")
sys.path.insert(0, _REF_DIR)

DEFAULT_DATA_FOLDER = "/database2/physionet2026_kaggle/data"
DEFAULT_MODEL_PATH = os.path.join(_SCRIPT_DIR, "lstm_model_kaggle", "lstm_model.pt")
DEFAULT_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "lstm_results_kaggle")

SEQ_DIM = 465
ECG_DIM = 11
STATIC_DIM = 196
INPUT_DIM = SEQ_DIM + ECG_DIM
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
FC_HIDDEN = 64

# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True


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

    def forward(self, X_seq, X_ecg, x_static, lengths):
        X = torch.cat([X_seq, X_ecg], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            X, lengths.cpu(), batch_first=True, enforce_sorted=True,
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = torch.cat([h_n[0], h_n[1]], dim=-1)
        combined = torch.cat([h_last, x_static], dim=-1)
        return self.fc(combined).squeeze(-1)


# ============================================================================
# Preload + Sorted Batch Inference
# ============================================================================

def preload_all(cache_dir, test_records):
    """
    一次性加载所有测试记录的特征到内存。
    返回按序列长度降序排列的样本列表（减少 padding 浪费）。
    """
    samples = []
    miss_keys = []

    for rec in tqdm(test_records, desc="Loading features"):
        rec_key = f"{rec['BidsFolder']}_ses-{rec['SessionID']}"
        cache_path = os.path.join(cache_dir, f"{rec_key}.npz")

        if os.path.exists(cache_path):
            data = np.load(cache_path, allow_pickle=True)
            X_seq = data["X_seq"]
            X_ecg = data["X_ecg"]
            x_static = data["x_static"]
            y = int(data["y"])
            mask = data["mask"]
        else:
            miss_keys.append((rec, rec_key))
            continue

        # 清洗 + 转 tensor
        X_seq = np.clip(np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
        if X_ecg is None or len(X_ecg) == 0:
            X_ecg_arr = np.zeros((0, ECG_DIM), dtype=np.float32)
        else:
            X_ecg_arr = np.clip(np.nan_to_num(X_ecg, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
        x_static = np.clip(np.nan_to_num(x_static, nan=0.0, posinf=0.0, neginf=0.0), -50.0, 50.0)
        if mask is None:
            mask = np.zeros(len(X_seq), dtype=bool)

        L = len(X_seq)
        samples.append({
            "X_seq": X_seq,
            "X_ecg": X_ecg_arr,
            "x_static": x_static,
            "y": y,
            "mask": mask,
            "length": L,
            "rec_key": rec_key,
        })

    # cache miss 直接跳过
    if miss_keys:
        logger.warning("Cache miss: %d records, skipped", len(miss_keys))
        for _, rec_key in miss_keys:
            logger.debug("  skipped: %s", rec_key)

    # 按长度降序排列 —— pack_padded_sequence 要求
    samples.sort(key=lambda s: s["length"], reverse=True)
    logger.info("Preloaded %d samples (cache hit=%d, miss=%d)",
                len(samples), len(samples) - len(miss_keys), len(miss_keys))
    return samples


@torch.inference_mode()
def infer_batched(model, samples, batch_size=256):
    """
    对预加载的样本做批量推理。
    samples 已按长度降序排列，同一 batch 内长度相近 → padding 浪费最小。
    """
    model.eval()
    all_y, all_prob, all_keys = [], [], []
    n = len(samples)

    for start in tqdm(range(0, n, batch_size), desc="Inference"):
        end = min(start + batch_size, n)
        batch = samples[start:end]
        bs = len(batch)
        max_len = batch[0]["length"]  # 降序排列，第一个最长

        # 构建 batch tensors
        X_seq_b = torch.zeros(bs, max_len, SEQ_DIM, device=device)
        X_ecg_b = torch.zeros(bs, max_len, ECG_DIM, device=device)
        x_static_b = torch.zeros(bs, STATIC_DIM, device=device)
        lengths_b = torch.zeros(bs, dtype=torch.long, device=device)

        for i, s in enumerate(batch):
            L = s["length"]
            X_seq_b[i, :L] = torch.from_numpy(s["X_seq"]).to(device)
            ecg_len = min(s["X_ecg"].shape[0], L - 10)
            if ecg_len > 0:
                X_ecg_b[i, 10:10 + ecg_len] = torch.from_numpy(s["X_ecg"][:ecg_len]).to(device)
            x_static_b[i] = torch.from_numpy(s["x_static"]).to(device)
            lengths_b[i] = L

        # 前向
        logits = model(X_seq_b, X_ecg_b, x_static_b, lengths_b)
        prob = torch.sigmoid(logits)

        all_y.extend([s["y"] for s in batch])
        all_prob.extend(prob.cpu().numpy().tolist())
        all_keys.extend([s["rec_key"] for s in batch])

    return np.array(all_y), np.array(all_prob), all_keys


# ============================================================================
# Metrics
# ============================================================================

from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve,
)

def compute_all_metrics(y_true, y_prob):
    # 确保 1D 数组
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_prob = np.clip(np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    n = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    y_pred = (y_prob >= 0.5).astype(int)

    m = OrderedDict()
    m["n_total"] = n
    m["n_positive"] = n_pos
    m["n_negative"] = n_neg
    m["pos_ratio"] = n_pos / n if n > 0 else 0.0
    m["prob_mean"] = float(np.mean(y_prob))
    m["prob_std"] = float(np.std(y_prob))
    m["prob_median"] = float(np.median(y_prob))
    m["prob_min"] = float(np.min(y_prob))
    m["prob_max"] = float(np.max(y_prob))

    try:
        m["AUROC"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        m["AUROC"] = float("nan")
    try:
        m["AUPRC"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        m["AUPRC"] = float("nan")

    m["Accuracy"] = float(accuracy_score(y_true, y_pred))
    m["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    m["Precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    m["Recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    m["Specificity"] = float(recall_score(1 - y_true, 1 - y_pred, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m["TP"], m["TN"], m["FP"], m["FN"] = int(tp), int(tn), int(fp), int(fn)

    idx_sorted = np.argsort(y_prob)[::-1]
    for cap_pct in [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]:
        capacity = max(1, int(cap_pct * n))
        m[f"TPR@{int(cap_pct*100)}%"] = float(np.mean(y_true[idx_sorted[:capacity]] == 1))
        m[f"capacity@{int(cap_pct*100)}%"] = capacity

    pos_prob = y_prob[y_true == 1]
    neg_prob = y_prob[y_true == 0]
    m["pos_class_prob_mean"] = float(np.mean(pos_prob)) if len(pos_prob) > 0 else float("nan")
    m["neg_class_prob_mean"] = float(np.mean(neg_prob)) if len(neg_prob) > 0 else float("nan")

    if n_pos > 0 and n_neg > 0:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_j_idx = np.argmax(j_scores)
        m["best_threshold_youden"] = float(thresholds[best_j_idx])
        m["best_tpr_youden"] = float(tpr[best_j_idx])
        m["best_fpr_youden"] = float(fpr[best_j_idx])
    else:
        m["best_threshold_youden"] = float("nan")
        m["best_tpr_youden"] = float("nan")
        m["best_fpr_youden"] = float("nan")

    return m


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LSTM Inference on a Split")
    parser.add_argument("--data_folder", type=str, default=DEFAULT_DATA_FOLDER)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="LSTM cache dir (default: lstm_cache_kaggle/<split>)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test", "external"],
                        help="Split to evaluate (default: test)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for inference (default: 256)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    SPLITS_DIR = os.path.join(args.data_folder, "splits")
    CACHE_DIR = args.cache_dir or os.path.join(_SCRIPT_DIR, "lstm_cache_kaggle", args.split)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LSTM Inference — %s Split Evaluation", args.split)
    logger.info("=" * 60)
    logger.info("Data folder:   %s", args.data_folder)
    logger.info("Model path:    %s", args.model)
    logger.info("Cache dir:     %s", CACHE_DIR)
    logger.info("Output dir:    %s", args.output_dir)
    logger.info("Split:         %s", args.split)
    logger.info("Device:        %s", device)
    logger.info("Batch size:    %d", args.batch_size)

    # ---- 加载 split 记录 ----
    test_path = os.path.join(SPLITS_DIR, f"{args.split}_records.json")
    if not os.path.exists(test_path):
        logger.error("Split records not found: %s", test_path)
        sys.exit(1)
    with open(test_path) as f:
        test_records = json.load(f)
    logger.info("Test records: %d", len(test_records))

    # ---- 加载模型 ----
    logger.info("Loading model...")
    model = LSTMModel().to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        logger.info("Loaded state_dict (val_auroc=%.4f)", checkpoint.get("auroc", float("nan")))
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded raw state_dict")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable params: %d", n_params)

    # ---- 预加载所有特征到内存 ----
    t0 = time.time()

    samples = preload_all(CACHE_DIR, test_records)
    load_time = time.time() - t0
    logger.info("Preload complete in %.1fs (%.0f rec/s)", load_time, len(samples) / load_time if load_time > 0 else 0)

    # ---- 批量推理 ----
    logger.info("Running inference (batch_size=%d)...", args.batch_size)
    t1 = time.time()
    y_true, y_prob, rec_keys = infer_batched(model, samples, args.batch_size)
    infer_time = time.time() - t1
    logger.info("Inference complete in %.1fs (%.0f rec/s)", infer_time, len(y_true) / infer_time if infer_time > 0 else 0)

    # 清理 GPU 内存
    del samples, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 去除非有限值
    n_bad = int(np.sum(~np.isfinite(y_prob)))
    if n_bad > 0:
        logger.warning("Found %d non-finite predictions — clipped", n_bad)
        y_prob = np.clip(np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    total_time = time.time() - t0
    logger.info("Predictions: %d records, %d positive", len(y_true), int(np.sum(y_true == 1)))
    logger.info("Total wall time: %.1fs", total_time)

    # ---- 计算指标 ----
    logger.info("Computing metrics...")
    metrics = compute_all_metrics(y_true, y_prob)

    # ---- 写入指标文件 ----
    metrics_path = os.path.join(args.output_dir, "lstm_test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("LSTM Model — Test Set Metrics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model:            {args.model}\n")
        f.write(f"Data folder:      {args.data_folder}\n")
        f.write(f"Device:           {device}\n")
        f.write(f"Preload time:     {load_time:.1f}s\n")
        f.write(f"Inference time:   {infer_time:.1f}s\n")
        f.write(f"Total wall time:  {total_time:.1f}s\n")
        f.write(f"Batch size:       {args.batch_size}\n\n")

        sections = [
            ("Basic Info", ["n_total", "n_positive", "n_negative", "pos_ratio"]),
            ("Probability Distribution", ["prob_mean", "prob_std", "prob_median", "prob_min", "prob_max"]),
            ("Core Metrics", ["AUROC", "AUPRC", "Accuracy", "F1", "Precision", "Recall", "Specificity"]),
            ("Confusion Matrix", ["TP", "TN", "FP", "FN"]),
            ("TPR at Capacity", [
                "TPR@1%", "capacity@1%", "TPR@5%", "capacity@5%",
                "TPR@10%", "capacity@10%", "TPR@20%", "capacity@20%",
                "TPR@30%", "capacity@30%", "TPR@50%", "capacity@50%",
            ]),
            ("Class-conditional Probabilities", ["pos_class_prob_mean", "neg_class_prob_mean"]),
            ("Optimal Threshold (Youden)", ["best_threshold_youden", "best_tpr_youden", "best_fpr_youden"]),
        ]

        for section_name, keys in sections:
            f.write(f"--- {section_name} ---\n")
            for k in keys:
                if k in metrics:
                    v = metrics[k]
                    fmt = f"{v:.6f}" if isinstance(v, float) else str(v)
                    f.write(f"  {k:30s} = {fmt}\n")
            f.write("\n")

        f.write("--- All Metrics (key=value) ---\n")
        for k, v in metrics.items():
            f.write(f"  {k} = {v}\n")

    logger.info("Metrics saved to %s", metrics_path)

    # ---- 写入预测 CSV ----
    predictions_path = os.path.join(args.output_dir, "lstm_test_predictions.csv")

    patient_ids = []
    for key in rec_keys:
        parts = key.rsplit("_ses-", 1)
        patient_ids.append(parts[0] if len(parts) == 2 else key)

    y_pred_binary = (y_prob >= 0.5).astype(int)
    df_pred = pd.DataFrame({
        "BDSPPatientID": patient_ids,
        "Cognitive_Impairment": y_pred_binary,
        "Cognitive_Impairment_Probability": y_prob,
    })
    df_pred.to_csv(predictions_path, index=False)
    logger.info("Predictions saved to %s (%d rows)", predictions_path, len(df_pred))

    # ---- 打印摘要 ----
    print("\n" + "=" * 60)
    print("LSTM Test Results Summary")
    print("=" * 60)
    print(f"  Load:    {load_time:.1f}s  |  Infer: {infer_time:.1f}s  |  Total: {total_time:.1f}s")
    print(f"  N        = {metrics['n_total']}  (pos={metrics['n_positive']}, neg={metrics['n_negative']})")
    print(f"  AUROC    = {metrics['AUROC']:.4f}")
    print(f"  AUPRC    = {metrics['AUPRC']:.4f}")
    print(f"  Accuracy = {metrics['Accuracy']:.4f}")
    print(f"  F1       = {metrics['F1']:.4f}")
    print(f"  TPR@5%   = {metrics['TPR@5%']:.4f}  (cap={metrics['capacity@5%']})")
    print(f"  TPR@10%  = {metrics['TPR@10%']:.4f}  (cap={metrics['capacity@10%']})")
    print(f"  TPR@20%  = {metrics['TPR@20%']:.4f}  (cap={metrics['capacity@20%']})")
    print(f"\n  Metrics     → {metrics_path}")
    print(f"  Predictions → {predictions_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
