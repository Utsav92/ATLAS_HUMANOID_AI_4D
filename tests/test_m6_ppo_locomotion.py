"""Unit tests for Milestone M6: PPO Balancing and Locomotion."""

import pytest
import numpy as np
import os
from atlas.rl.environments import G1StandingEnv, G1WalkingEnv
from atlas.rl.train_ppo import train_standing_ppo, evaluate_ppo_policy


def test_m6_environment_gym_interface():
    env = G1StandingEnv(max_episode_steps=50)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(obs, np.ndarray)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_obs, reward, term, trunc, step_info = env.step(action)

    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(term, bool)
    assert isinstance(trunc, bool)
    assert "r_height" in step_info
    assert "r_upright" in step_info


def test_m6_ppo_short_training(tmp_path):
    ckpt_path = str(tmp_path / "test_g1_ppo.zip")
    # Smoke training test: 256 steps
    model = train_standing_ppo(total_timesteps=256, save_path=ckpt_path)
    assert model is not None
    assert os.path.exists(ckpt_path)

    metrics = evaluate_ppo_policy(model, n_eval_episodes=2)
    assert "mean_reward" in metrics
    assert isinstance(metrics["mean_reward"], float)
