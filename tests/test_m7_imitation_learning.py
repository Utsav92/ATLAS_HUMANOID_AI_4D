"""Unit tests for Milestone M7: Imitation Learning and Manipulation."""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from atlas.imitation.behavior_cloning import BehaviorCloningPolicy, TemporalBCPolicy, ImitationTrainer


def test_m7_bc_policy_forward():
    policy = BehaviorCloningPolicy(state_dim=50, action_dim=43)
    x = torch.randn(8, 50)
    out = policy(x)
    assert out.shape == (8, 43)


def test_m7_temporal_bc_policy():
    policy = TemporalBCPolicy(state_dim=50, action_dim=43)
    x_seq = torch.randn(8, 5, 50)  # 5 history steps
    out = policy(x_seq)
    assert out.shape == (8, 43)


def test_m7_imitation_training():
    torch.manual_seed(42)
    # Synthetic demonstrations: 64 samples
    states = torch.randn(64, 50)
    # Target actions with structured relationship
    actions = torch.tanh(states[:, :43] * 0.5)

    train_dl = DataLoader(TensorDataset(states[:48], actions[:48]), batch_size=16)
    val_dl = DataLoader(TensorDataset(states[48:], actions[48:]), batch_size=16)

    policy = BehaviorCloningPolicy(state_dim=50, action_dim=43)
    trainer = ImitationTrainer(policy, lr=2e-3)

    initial_loss = trainer.evaluate(val_dl)
    for _ in range(5):
        trainer.train_epoch(train_dl)
    final_loss = trainer.evaluate(val_dl)

    assert final_loss < initial_loss
