"""Models required by Physio Fusion V1 Gate 2."""
from __future__ import annotations

import torch
from torch import nn


class SharedPairAutoencoder(nn.Module):
    """One shared 24-D coherence-pair autoencoder for all 15 EEG pairs."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        if latent_dim not in (4, 8, 12, 16):
            raise ValueError(f"unsupported latent_dim={latent_dim}")
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(24, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 24),
        )

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(values))


class PhysioFusion192(nn.Module):
    """Frozen Gate4 epoch fusion and patient logit head."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_fusion = nn.Sequential(
            nn.Linear(361, 256),
            nn.ReLU(),
            nn.Linear(256, 192),
        )
        self.patient_head = nn.Linear(384, 1)

    def encode_epoch(self, values: torch.Tensor) -> torch.Tensor:
        return self.epoch_fusion(values)

    def forward(self, values: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        emb = self.encode_epoch(values)
        mask = present.to(dtype=emb.dtype).unsqueeze(-1)
        count = mask.sum(dim=1).clamp_min(1.0)
        mean = (emb * mask).sum(dim=1) / count
        centered = (emb - mean.unsqueeze(1)) * mask
        variance = centered.square().sum(dim=1) / count
        # Preserve population standard deviation while avoiding 0/0 gradients
        # for constant night embeddings. The dtype tiny is numerically negligible.
        std = torch.sqrt(variance.clamp_min(torch.finfo(variance.dtype).tiny))
        return self.patient_head(torch.cat([mean, std], dim=1)).squeeze(1)
