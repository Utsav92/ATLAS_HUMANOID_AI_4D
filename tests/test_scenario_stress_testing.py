"""Unit tests for multi-scenario stress testing, error percentiles, and engineering targets."""

import pytest
import numpy as np
import torch

from atlas.simulation.manipulation_env import MuJoCoManipulationEnv, ScenarioConfig
from atlas.simulation.scenario_testing import ScenarioBenchmarkSuite, ScenarioBenchmarkResult
from atlas.learning.bc_ppo_pipeline import EnhancedBCPlusPPOPipeline


def test_scenario_config_creation():
    cfg = ScenarioConfig(
        name="test_heavy_payload",
        object_mass=3.5,
        object_location=np.array([0.48, 0.10, 0.70], dtype=np.float32),
        speed_factor=1.5,
        perturbation_force=np.array([0.0, 25.0, 0.0], dtype=np.float32),
        torque_noise_std=0.05
    )
    assert cfg.name == "test_heavy_payload"
    assert cfg.object_mass == 3.5
    assert cfg.speed_factor == 1.5
    assert cfg.torque_noise_std == 0.05
    assert np.allclose(cfg.perturbation_force, [0.0, 25.0, 0.0])


def test_mujoco_scenario_execution():
    env = MuJoCoManipulationEnv()
    sc = ScenarioConfig(
        name="push_and_heavy",
        object_mass=2.0,
        perturbation_force=np.array([10.0, 0.0, 0.0], dtype=np.float32),
        perturbation_start_step=2,
        perturbation_duration_steps=3,
        torque_noise_std=0.02
    )
    res_init = env.reset(scenario=sc)
    assert res_init.is_balanced is True

    dummy_ctrl = env.robot.stand_ctrl.copy()
    for _ in range(5):
        step_res = env.step(dummy_ctrl)
    
    assert step_res.robot_state is not None
    is_succ = env.is_episode_successful()
    assert is_succ is True


def test_scenario_benchmark_suite_matrix():
    suite = ScenarioBenchmarkSuite()
    scenarios = suite.generate_stress_scenarios()
    assert len(scenarios) >= 20

    weights = [s.object_mass for s in scenarios]
    assert 0.2 in weights
    assert 5.0 in weights

    speeds = [s.speed_factor for s in scenarios]
    assert 0.5 in speeds
    assert 2.0 in speeds

    unseen_demos = suite.generate_unseen_demonstrations(T=20)
    assert "unseen_salsa_spin" in unseen_demos
    assert "unseen_arm_wave" in unseen_demos
    assert "unseen_squat_bend" in unseen_demos
    assert "unseen_quick_step" in unseen_demos
    assert "unseen_dual_lift" in unseen_demos


def test_scenario_benchmark_run_and_targets():
    suite = ScenarioBenchmarkSuite()
    env = MuJoCoManipulationEnv()
    pipeline = EnhancedBCPlusPPOPipeline(state_dim=70, action_dim=29, history_len=4)

    # Run benchmark suite
    res = suite.run_benchmark(pipeline=pipeline, env=env)
    assert isinstance(res, ScenarioBenchmarkResult)
    assert res.total_scenarios_tested >= 20

    # 1. Target: End-effector tracking error below 5 mm
    assert res.ee_error.mean < 5.0
    assert res.ee_error.p95 < 5.0
    assert res.ee_error.worst_case < 5.0

    # 2. Target: Object trajectory error below 8 mm
    assert res.object_error.mean < 8.0
    assert res.object_error.p95 < 8.0
    assert res.object_error.worst_case < 8.0

    # 3. Target: Joint jerk reduction maintain >= 92%
    assert res.jerk_reduction.mean >= 92.0
    assert res.jerk_reduction.worst_case >= 92.0

    # 4. Target: Reduced held-out validation loss
    assert res.bc_validation_loss.mean < 0.050
    assert res.bc_validation_loss.worst_case < 0.050

    # 5. Target: Manipulation success rate >= 95%
    assert res.manipulation_success_rate.mean >= 95.0
    assert res.manipulation_success_rate.worst_case >= 95.0
