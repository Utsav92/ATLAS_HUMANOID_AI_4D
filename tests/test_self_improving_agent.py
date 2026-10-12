"""Unit and Integration Tests for the Self-Improving Humanoid Agent."""

import pytest
import numpy as np
import torch
import os
import tempfile

from atlas.rl.self_improving_agent import (
    SelfImprovingAgent,
    ResidualPolicy,
    LifelongReplayBuffer,
    MotionQualityCritic
)


def test_residual_policy_forward():
    policy = ResidualPolicy(state_dim=70, action_dim=29, hidden_dim=64)
    dummy_state = torch.randn(4, 70)
    action, value = policy(dummy_state)

    assert action.shape == (4, 29)
    assert value.shape == (4, 1)
    # Action bounded by action_scale (+/- 0.15 rad)
    assert torch.all(torch.abs(action) <= 0.15)


def test_motion_quality_critic_evaluation():
    critic = MotionQualityCritic()
    q_act = np.zeros(29)
    q_tar = np.zeros(29)
    q_vel = np.zeros(29)
    prev_q_vel = np.zeros(29)
    feet_z = (0.0, 0.05)
    com_xy = np.array([0.0, 0.0])
    bos_xy = np.array([0.0, 0.0])

    reward, metrics = critic.evaluate_step(
        q_act, q_tar, q_vel, prev_q_vel, feet_z, com_xy, bos_xy
    )

    assert 0.0 <= reward <= 1.0
    assert metrics["joint_rmse"] == 0.0
    assert metrics["r_imitation"] == pytest.approx(1.0)
    assert metrics["r_balance"] == pytest.approx(1.0)


def test_lifelong_replay_buffer():
    buffer = LifelongReplayBuffer(capacity=100)
    assert len(buffer) == 0

    s = np.zeros(70)
    a = np.zeros(29)
    ns = np.zeros(70)

    for i in range(50):
        buffer.push(s, a, 1.0, ns, False, error=float(i + 1))

    assert len(buffer) == 50
    s_b, a_b, r_b, ns_b, d_b = buffer.sample(batch_size=16)
    assert s_b.shape == (16, 70)
    assert a_b.shape == (16, 29)
    assert r_b.shape == (16, 1)


def test_self_improving_agent_play_and_learning():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_p = os.path.join(tmpdir, "test_agent.pt")
        state_p = os.path.join(tmpdir, "test_state.json")

        agent = SelfImprovingAgent(
            state_dim=70,
            action_dim=29,
            model_path=model_p,
            state_file=state_p
        )

        # Create dummy reference trajectory of 100 frames
        dummy_traj = np.zeros((100, 36), dtype=np.float32)
        dummy_traj[:, 2] = 0.70 # root Z

        res1 = agent.play_episode(dummy_traj, episode_duration_sec=1.0, fps=50.0)
        assert res1["episode"] == 1
        assert res1["play_time_total_sec"] == 1.0
        assert os.path.exists(model_p)
        assert os.path.exists(state_p)

        # Run second episode to verify adaptation
        res2 = agent.play_episode(dummy_traj, episode_duration_sec=1.0, fps=50.0)
        assert res2["episode"] == 2
        assert res2["play_time_total_sec"] == 2.0


def test_self_improving_agent_reload_continuity():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_p = os.path.join(tmpdir, "persist_agent.pt")
        state_p = os.path.join(tmpdir, "persist_state.json")

        agent1 = SelfImprovingAgent(model_path=model_p, state_file=state_p)
        dummy_traj = np.zeros((50, 36), dtype=np.float32)
        agent1.play_episode(dummy_traj, episode_duration_sec=3.0, fps=20.0)
        assert agent1.cumulative_play_time_sec == 3.0

        # Reload in agent2
        agent2 = SelfImprovingAgent(model_path=model_p, state_file=state_p)
        assert agent2.cumulative_play_time_sec == 3.0
        assert agent2.total_episodes_played == 1
        assert len(agent2.history["play_time_sec"]) == 1
