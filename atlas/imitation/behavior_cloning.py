"""Behavior Cloning and Imitation Learning for Humanoid Manipulation.
Implements standard BC, Temporal BC (Action Chunking), and offline policy evaluation.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from atlas.deep_learning.models import CompactMLP, CompactGRU


class BehaviorCloningPolicy(nn.Module):
    """Feedforward Behavior Cloning Policy."""
    def __init__(self, state_dim: int = 50, action_dim: int = 43, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TemporalBCPolicy(nn.Module):
    """Temporal Behavior Cloning with historical sequence conditioning."""
    def __init__(self, state_dim: int = 50, action_dim: int = 43, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        # state_seq: (batch, history_len, state_dim)
        out, _ = self.gru(state_seq)
        return self.head(out[:, -1, :])


class ImitationTrainer:
    def __init__(self, policy: nn.Module, lr: float = 1e-3, device: str = "cpu"):
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.criterion = nn.MSELoss()

    def train_epoch(self, loader: DataLoader) -> float:
        self.policy.train()
        total_loss = 0.0
        count = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.policy(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += float(loss.item()) * len(x)
            count += len(x)
        return total_loss / max(count, 1)

    def evaluate(self, loader: DataLoader) -> float:
        self.policy.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.policy(x)
                loss = self.criterion(pred, y)
                total_loss += float(loss.item()) * len(x)
                count += len(x)
        return total_loss / max(count, 1)
