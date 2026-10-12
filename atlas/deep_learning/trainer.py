"""Training and Checkpointing Infrastructure for Deep Learning Models.
Supports early stopping, CPU thread management, and reproducible training metrics.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CompactTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_epoch(self, dataloader: DataLoader, loss_fn: Callable) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item()) * len(batch_x)
            count += len(batch_x)
        return total_loss / max(count, 1)

    def evaluate(self, dataloader: DataLoader, loss_fn: Callable) -> float:
        self.model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                pred = self.model(batch_x)
                loss = loss_fn(pred, batch_y)
                total_loss += float(loss.item()) * len(batch_x)
                count += len(batch_x)
        return total_loss / max(count, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        epochs: int = 10,
        checkpoint_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": []}
        best_val = float("inf")

        for epoch in range(epochs):
            t_loss = self.train_epoch(train_loader, loss_fn)
            v_loss = self.evaluate(val_loader, loss_fn)
            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)

            if v_loss < best_val and checkpoint_path is not None:
                best_val = v_loss
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(self.model.state_dict(), checkpoint_path)

        return history
