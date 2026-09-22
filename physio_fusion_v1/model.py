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
