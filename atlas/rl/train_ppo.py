"""PPO Training and Evaluation Pipeline for Unitree G1 Humanoid.
Integrates Stable-Baselines3 with custom checkpointing and reward logging.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from atlas.rl.environments import G1StandingEnv, G1WalkingEnv


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        return True


def train_standing_ppo(
    total_timesteps: int = 2048,
    save_path: Optional[str] = None
) -> PPO:
    """Train a CPU-friendly PPO policy for G1 standing."""
    env = G1StandingEnv(max_episode_steps=200)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        gamma=0.98,
        learning_rate=3e-4,
        verbose=1,
        device="cpu"
    )

    model.learn(total_timesteps=total_timesteps)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)

    return model


def evaluate_ppo_policy(model: PPO, n_eval_episodes: int = 5) -> Dict[str, float]:
    """Evaluate trained policy on G1StandingEnv."""
    env = G1StandingEnv(max_episode_steps=200)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes)
    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward)
    }
