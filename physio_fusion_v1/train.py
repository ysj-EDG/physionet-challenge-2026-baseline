"""CI-blind shared coherence-pair autoencoder training and selection."""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from model import SharedPairAutoencoder


FAMILY_SLICES = {
    "means": slice(0, 5),
    "auc": slice(5, 10),
    "iqr": slice(10, 15),
    "ratios": slice(15, 19),
    "spectrum": slice(19, 22),
    "sigma": slice(22, 24),
}


@dataclass(frozen=True)
class AEConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 4096
    max_epochs: int = 50
    patience: int = 5


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict(model: nn.Module, values: np.ndarray, device: torch.device,
             batch_size: int = 16_384) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start:start + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def _r2_by_feature(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    residual = np.square(target - prediction).sum(axis=0, dtype=np.float64)
    centered = target - target.mean(axis=0, dtype=np.float64)
    total = np.square(centered).sum(axis=0, dtype=np.float64)
    return 1.0 - residual / total


def _pearson_by_feature(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    target_centered = target - target.mean(axis=0, dtype=np.float64)
    pred_centered = prediction - prediction.mean(axis=0, dtype=np.float64)
    numerator = (target_centered * pred_centered).sum(axis=0, dtype=np.float64)
    denominator = np.sqrt(
        np.square(target_centered).sum(axis=0, dtype=np.float64)
        * np.square(pred_centered).sum(axis=0, dtype=np.float64)
    )
    return numerator / denominator


def evaluate_coherence_ae(
    model: nn.Module,
    validation: dict[str, list[tuple[str, np.ndarray]]],
    device: torch.device,
) -> dict[str, object]:
    per_site_mse: dict[str, float] = {}
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    record_counts: dict[str, int] = {}
    for site, records in sorted(validation.items()):
        record_mse: list[float] = []
        for _, values in records:
            prediction = _predict(model, values, device)
            record_mse.append(float(np.square(values - prediction).mean(dtype=np.float64)))
            targets.append(values)
            predictions.append(prediction)
        per_site_mse[site] = float(np.mean(record_mse))
        record_counts[site] = len(record_mse)
    site_balanced_mse = float(np.mean(list(per_site_mse.values())))
    target = np.concatenate(targets)
    prediction = np.concatenate(predictions)
    feature_mse = np.square(target - prediction).mean(axis=0, dtype=np.float64)
    feature_r2 = _r2_by_feature(target, prediction)
    feature_pearson = _pearson_by_feature(target, prediction)
    overall_residual = float(np.square(target - prediction).sum(dtype=np.float64))
    overall_total = float(np.square(target - target.mean(dtype=np.float64)).sum(dtype=np.float64))
    overall_r2 = 1.0 - overall_residual / overall_total
    family_metrics = {
        name: {
            "mse": float(feature_mse[slc].mean()),
            "r2": float(feature_r2[slc].mean()),
            "pearson": float(feature_pearson[slc].mean()),
        }
        for name, slc in FAMILY_SLICES.items()
    }
    finite_values = np.concatenate([
        np.array([site_balanced_mse, overall_r2], dtype=np.float64),
        np.array(list(per_site_mse.values()), dtype=np.float64),
        feature_mse, feature_r2, feature_pearson,
        np.array([metric for family in family_metrics.values() for metric in family.values()]),
    ])
    if not np.isfinite(finite_values).all():
        raise FloatingPointError("non-finite coherence reconstruction metric")
    return {
        "site_balanced_mse": site_balanced_mse,
        "per_site_record_balanced_mse": per_site_mse,
        "validation_record_counts": record_counts,
        "overall_r2": float(overall_r2),
        "feature_mse": feature_mse.tolist(),
        "feature_r2": feature_r2.tolist(),
        "feature_pearson": feature_pearson.tolist(),
        "family_metrics": family_metrics,
    }


def train_coherence_ae(
    latent_dim: int,
    seed: int,
    train_values: np.ndarray,
    validation: dict[str, list[tuple[str, np.ndarray]]],
    device: torch.device,
    config: AEConfig = AEConfig(),
    return_model: bool = False,
) -> dict[str, object]:
    if train_values.ndim != 2 or train_values.shape[1] != 24 or not len(train_values):
        raise ValueError(f"invalid training matrix: {train_values.shape}")
    if not np.isfinite(train_values).all():
        raise ValueError("non-finite training matrix")
    set_seed(seed)
    model = SharedPairAutoencoder(latent_dim).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss(reduction="mean")
    train_tensor = torch.from_numpy(train_values).to(device)
    best_metric = math.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    started = time.time()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_tensor), device=device)
        loss_sum = 0.0
        sample_count = 0
        for start in range(0, len(train_tensor), config.batch_size):
            index = permutation[start:start + config.batch_size]
            batch = train_tensor[index]
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss: d={latent_dim}, seed={seed}, epoch={epoch}"
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            sample_count += len(batch)
        metrics = evaluate_coherence_ae(model, validation, device)
        train_mse = loss_sum / sample_count
        selection = float(metrics["site_balanced_mse"])
        history.append({"epoch": epoch, "train_mse": train_mse, "site_balanced_val_mse": selection})
        print(
            f"d={latent_dim} seed={seed} epoch={epoch} train_mse={train_mse:.8g} "
            f"val_mse={selection:.8g} best={best_metric:.8g}",
            flush=True,
        )
        if selection < best_metric:
            best_metric = selection
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("autoencoder produced no checkpoint")
    model.load_state_dict(best_state)
    best_metrics = evaluate_coherence_ae(model, validation, device)
    result = {
        "latent_dim": latent_dim,
        "seed": seed,
        "best_epoch": best_epoch,
        "elapsed_sec": time.time() - started,
        "config": asdict(config),
        "history": history,
        **best_metrics,
    }
    if return_model:
        result["model"] = model
    return result


def summarize_and_select(
    seed_results: Sequence[dict[str, object]], candidates: Sequence[int] = (4, 8, 12, 16),
) -> tuple[list[dict[str, object]], int, int, float]:
    summaries: list[dict[str, object]] = []
    for latent_dim in candidates:
        rows = [row for row in seed_results if int(row["latent_dim"]) == latent_dim]
        if len(rows) != 3:
            raise RuntimeError(f"d={latent_dim} has {len(rows)} successful seeds, expected 3")
        values = np.asarray([row["site_balanced_mse"] for row in rows], dtype=np.float64)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(len(values)))
        sites = sorted(rows[0]["per_site_record_balanced_mse"])
        summary: dict[str, object] = {
            "latent_dim": latent_dim,
            "mean_site_balanced_val_mse": mean,
            "se_site_balanced_val_mse": se,
        }
        for site in sites:
            summary[f"mean_{site}_record_balanced_mse"] = float(np.mean([
                row["per_site_record_balanced_mse"][site] for row in rows
            ]))
        summary["mean_overall_r2"] = float(np.mean([row["overall_r2"] for row in rows]))
        summaries.append(summary)
    best = min(summaries, key=lambda row: (row["mean_site_balanced_val_mse"], row["latent_dim"]))
    best_d = int(best["latent_dim"])
    threshold = float(best["mean_site_balanced_val_mse"] + best["se_site_balanced_val_mse"])
    selected_d = min(
        int(row["latent_dim"])
        for row in summaries
        if float(row["mean_site_balanced_val_mse"]) <= threshold
    )
    for row in summaries:
        row["best_d"] = best_d
        row["one_se_threshold"] = threshold
        row["selected"] = int(row["latent_dim"]) == selected_d
    return summaries, selected_d, best_d, threshold
