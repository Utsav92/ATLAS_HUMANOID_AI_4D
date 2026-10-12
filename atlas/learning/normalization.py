"""Observation and Action Normalizer for Humanoid Imitation & Reinforcement Learning.
Provides standardized Z-score observation normalization and bounded [-1, 1] action scaling.
Guarantees strictly identical normalization statistics across training and physical inference.
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch


class ObservationActionNormalizer:
    """Normalizes high-dimensional state observations and actuator action targets."""

    def __init__(self,
                 obs_dim: int = 70,
                 action_dim: int = 29,
                 action_ranges: Optional[np.ndarray] = None,  # (action_dim, 2) [min, max]
                 eps: float = 1e-6):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.eps = eps

        # Observation statistics (running mean and variance)
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32)
        self.obs_std = np.ones(obs_dim, dtype=np.float32)
        self.is_fit = False

        # Action limits: default Unitree G1 actuator limits [-1.57, 1.57]
        if action_ranges is not None:
            self.action_min = action_ranges[:, 0].astype(np.float32)
            self.action_max = action_ranges[:, 1].astype(np.float32)
        else:
            self.action_min = -np.ones(action_dim, dtype=np.float32) * 1.57
            self.action_max = np.ones(action_dim, dtype=np.float32) * 1.57

        self.action_scale = (self.action_max - self.action_min) / 2.0
        self.action_offset = (self.action_max + self.action_min) / 2.0

    def fit_observations(self, obs_dataset: np.ndarray):
        """Compute mean and standard deviation from training observations."""
        self.obs_mean = np.mean(obs_dataset, axis=0).astype(np.float32)
        self.obs_std = np.std(obs_dataset, axis=0).astype(np.float32)
        # Prevent division by zero for inactive DOFs
        self.obs_std = np.maximum(self.obs_std, self.eps)
        self.is_fit = True

    def normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Apply Z-score standardization: (obs - mean) / std."""
        if not self.is_fit:
            return obs
        return (obs - self.obs_mean) / self.obs_std

    def denormalize_observation(self, norm_obs: np.ndarray) -> np.ndarray:
        """Inverse standardization: norm_obs * std + mean."""
        if not self.is_fit:
            return norm_obs
        return norm_obs * self.obs_std + self.obs_mean

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Scale continuous action from physical limits [a_min, a_max] to normalized [-1, 1]."""
        n_act = min(len(action), self.action_dim)
        scaled = (action[:n_act] - self.action_offset[:n_act]) / (self.action_scale[:n_act] + self.eps)
        return np.clip(scaled, -1.0, 1.0).astype(np.float32)

    def denormalize_action(self, norm_action: np.ndarray) -> np.ndarray:
        """Map normalized policy action [-1, 1] back to valid physical actuator limits."""
        n_act = min(len(norm_action), self.action_dim)
        clipped_norm = np.clip(norm_action[:n_act], -1.0, 1.0)
        action_phys = clipped_norm * self.action_scale[:n_act] + self.action_offset[:n_act]
        return action_phys.astype(np.float32)

    def save(self, filepath: str):
        """Save normalization parameters to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
            "action_min": self.action_min.tolist(),
            "action_max": self.action_max.tolist(),
            "is_fit": self.is_fit
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> ObservationActionNormalizer:
        """Load normalization parameters from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        ranges = np.stack([np.array(data["action_min"]), np.array(data["action_max"])], axis=1)
        norm = cls(obs_dim=data["obs_dim"], action_dim=data["action_dim"], action_ranges=ranges)
        norm.obs_mean = np.array(data["obs_mean"], dtype=np.float32)
        norm.obs_std = np.array(data["obs_std"], dtype=np.float32)
        norm.is_fit = data["is_fit"]
        return norm
