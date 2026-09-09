"""Shared low-capacity static classifier used by P2 training and inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn


STATIC_DIM = 196
MODEL_TYPE = "static_logistic"


class StaticLogisticModel(nn.Module):
    """Apply a frozen full-static StandardScaler and selected-column logit."""

    def __init__(
        self,
        selected_indices: Sequence[int],
        scaler_mean: Sequence[float],
        scaler_scale: Sequence[float],
        coefficients: Sequence[float],
        intercept: float,
    ):
        super().__init__()
        indices = np.asarray(selected_indices, dtype=np.int64)
        mean = np.asarray(scaler_mean, dtype=np.float64)
        scale = np.asarray(scaler_scale, dtype=np.float64)
        coef = np.asarray(coefficients, dtype=np.float64)
        intercept_value = np.asarray(float(intercept), dtype=np.float64)

        if mean.shape != (STATIC_DIM,) or scale.shape != (STATIC_DIM,):
            raise ValueError(
                f"Static scaler must have {STATIC_DIM} entries, got "
                f"mean={mean.shape} scale={scale.shape}"
            )
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError("selected_indices must be a non-empty 1D sequence")
        if len(set(indices.tolist())) != len(indices):
            raise ValueError("selected_indices must be unique")
        if np.any(indices < 0) or np.any(indices >= STATIC_DIM):
            raise ValueError("selected_indices are outside the static input")
        if coef.shape != (len(indices),):
            raise ValueError(f"Expected {len(indices)} coefficients, got {coef.shape}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("Static scaler contains non-finite values")
        if np.any(scale <= 0):
            raise ValueError("Static scaler scales must be positive")
        if not np.all(np.isfinite(coef)) or not np.isfinite(intercept_value):
            raise ValueError("Static logistic parameters must be finite")

        self.register_buffer("selected_indices", torch.from_numpy(indices))
        self.register_buffer("scaler_mean", torch.from_numpy(mean))
        self.register_buffer("scaler_scale", torch.from_numpy(scale))
        self.register_buffer("coefficients", torch.from_numpy(coef))
        self.register_buffer("intercept", torch.from_numpy(intercept_value))

    def forward(self, X_seq, X_ecg, x_static, lengths):
        del X_seq, X_ecg, lengths
        values = x_static.to(dtype=self.scaler_mean.dtype)
        standardized = (values - self.scaler_mean) / self.scaler_scale
        selected = standardized.index_select(1, self.selected_indices)
        return selected.matmul(self.coefficients) + self.intercept

    @classmethod
    def from_checkpoint(cls, checkpoint: Mapping) -> "StaticLogisticModel":
        if checkpoint.get("model_type") != MODEL_TYPE:
            raise ValueError(
                f"Expected model_type={MODEL_TYPE!r}, "
                f"got {checkpoint.get('model_type')!r}"
            )
        scaler = checkpoint.get("linear_scaler")
        linear = checkpoint.get("linear_model")
        if not isinstance(scaler, Mapping) or not isinstance(linear, Mapping):
            raise ValueError("Static logistic checkpoint is missing scaler/model state")
        model = cls(
            selected_indices=checkpoint.get("selected_indices", []),
            scaler_mean=scaler.get("mean", []),
            scaler_scale=scaler.get("scale", []),
            coefficients=linear.get("coefficients", []),
            intercept=linear.get("intercept", float("nan")),
        )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("Static logistic checkpoint is missing state_dict")
        model.load_state_dict(state_dict)
        return model
