"""Minimal local temporal model shared by Stage/Event13 and EEG72."""
from __future__ import annotations

import torch
from torch import nn


class ResidualTemporalBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, dilation=dilation, padding=dilation),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(32, 32, kernel_size=3, dilation=dilation, padding=dilation),
            nn.Dropout(0.1),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


def masked_mean_std(tokens: torch.Tensor, block_padding_mask: torch.Tensor) -> torch.Tensor:
    """Return concatenated mean/std over real blocks; True means padding."""
    valid = (~block_padding_mask).to(tokens.dtype).unsqueeze(-1)
    count = valid.sum(1).clamp_min(1.0)
    mean = (tokens * valid).sum(1) / count
    variance = ((tokens - mean[:, None, :]).square() * valid).sum(1) / count
    return torch.cat([mean, variance.clamp_min(0).sqrt()], dim=-1)


class LocalTCNClassifier(nn.Module):
    receptive_field_epochs = 1 + 2 * (1 + 1 + 2 + 2)

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.projection = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU())
        self.temporal = nn.Sequential(ResidualTemporalBlock(1), ResidualTemporalBlock(2))
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, blocks: torch.Tensor, block_padding_mask: torch.Tensor) -> torch.Tensor:
        if blocks.ndim != 4 or blocks.shape[2] != 10 or blocks.shape[3] != self.input_dim:
            raise ValueError(f"Expected [B,M,10,{self.input_dim}], got {tuple(blocks.shape)}")
        if block_padding_mask.shape != blocks.shape[:2]:
            raise ValueError("block_padding_mask shape mismatch")
        if torch.any((~block_padding_mask).sum(1) == 0):
            raise ValueError("A patient cannot contain only padded blocks")
        batch, n_blocks = blocks.shape[:2]
        x = self.projection(blocks.reshape(batch * n_blocks, 10, self.input_dim))
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        token = torch.cat([x.mean(1), x.std(1, unbiased=False)], dim=-1)
        patient = masked_mean_std(token.reshape(batch, n_blocks, 64), block_padding_mask)
        return self.head(patient).squeeze(-1)

