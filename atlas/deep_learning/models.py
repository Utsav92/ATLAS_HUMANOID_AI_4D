"""Deep Learning Architectures for ATLAS HUMANOID AI 4D.
Compact, CPU-friendly PyTorch networks: MLP, CNN, GRU, TCN, Autoencoder, and Transformer.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactMLP(nn.Module):
    """Multi-Layer Perceptron for robot action prediction and state mapping."""
    def __init__(self, in_dim: int = 50, hidden_dim: int = 64, out_dim: int = 43):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompactCNN(nn.Module):
    """Convolutional Neural Network for tactile image processing and visual contact recognition."""
    def __init__(self, in_channels: int = 3, out_classes: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv(x).flatten(1)
        return self.fc(feat)


class CompactGRU(nn.Module):
    """Gated Recurrent Unit for temporal robot trajectory and motion prediction."""
    def __init__(self, in_dim: int = 43, hidden_dim: int = 64, num_layers: int = 2, out_dim: int = 43):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_dim)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TemporalConvNet(nn.Module):
    """1D Dilated Temporal Convolutional Network for sequential sensor features."""
    def __init__(self, in_channels: int = 43, out_channels: int = 43):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        return self.net(x)


class StateAutoencoder(nn.Module):
    """Autoencoder for state compression, latent representation, and reconstruction error anomaly detection."""
    def __init__(self, state_dim: int = 50, latent_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self.forward(x)
        return torch.mean((x - recon) ** 2, dim=-1)


class CompactTransformerEncoder(nn.Module):
    """Compact Transformer Encoder for inter-joint attention and multi-sensor coordination."""
    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2, out_dim: int = 43):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.input_proj = nn.Linear(43, d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 43)
        h = self.input_proj(x)
        out = self.transformer(h)
        return self.head(out[:, -1, :])
