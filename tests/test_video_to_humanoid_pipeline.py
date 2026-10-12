"""Unit and Integration Tests for the End-to-End Video-to-Humanoid Pipeline.
Validates:
1. MediaPipe / RTMPose (SimCC decoding and MLE fusion)
2. Depth Anything V2 (Metric depth and point cloud lifting)
3. SMPL-X (Parametric body model forward kinematics and fitting)
4. Motion Retargeting (Link scaling and multi-objective DLS IK)
5. MuJoCo Physics (Contact dynamics, CoP balance, and object manipulation)
6. Trajectory Optimization (EE, object trajectory, contact timing)
7. Behavior Cloning + PPO Pipeline (Pretraining and RL optimization)
8. ObservationActionNormalizer (State standardization & action bounding)
9. MotionAccuracyEvaluator (MPJPE, Joint MAE, Foot Sliding, Temporal Lag, Success Rate)
10. DAggerTrainer (Interactive corrective dataset aggregation)
"""

import pytest
import numpy as np
import torch
import os
import tempfile

# 1. Perception
from atlas.perception.pose_estimator import (
    RTMPoseEstimator, MediaPipePoseEstimator, HybridPoseEstimator, COCO_17_NAMES, MEDIAPIPE_33_NAMES
)
from atlas.perception.depth_anything_v2 import DepthAnythingV2Estimator

# 2. SMPL-X & Retargeting
from atlas.retargeting.smplx_model import SMPLXBodyModel, SMPLXFitter, SMPLXOutput
from atlas.retargeting.motion_retargeter import SMPLXToHumanoidRetargeter, RetargetingResult

# 3. MuJoCo Manipulation & Physics
from atlas.simulation.manipulation_env import MuJoCoManipulationEnv

# 4. Trajectory Optimization
from atlas.optimization.trajectory_optimizer import HumanoidTrajectoryOptimizer

# 5. Hybrid BC + PPO Learning, Normalization, DAgger, and Metrics
from atlas.learning.bc_ppo_pipeline import (
    EnhancedBCPlusPPOPipeline, TemporalActorCriticPolicy, CombinedMotionAccuracyLoss
)
from atlas.learning.normalization import ObservationActionNormalizer
from atlas.learning.metrics import MotionAccuracyEvaluator, MotionAccuracyReport
from atlas.learning.dagger_trainer import DAggerTrainer, DAggerIterationSummary


def test_pillar1_pose_estimators():
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

    rtm = RTMPoseEstimator()
    res_rtm = rtm.estimate_frame(frame)
    assert res_rtm.keypoints_2d.shape == (17, 2)
    assert res_rtm.keypoints_3d.shape == (17, 3)
    assert len(res_rtm.scores) == 17
    assert res_rtm.source == "rtmpose"

    mp = MediaPipePoseEstimator()
    res_mp = mp.estimate_frame(frame, timestamp=0.5)
    assert res_mp.keypoints_2d.shape == (33, 2)
    assert res_mp.keypoints_3d.shape == (33, 3)
    assert res_mp.source == "mediapipe"

    hybrid = HybridPoseEstimator()
    res_hyb = hybrid.estimate_frame(frame, timestamp=2.0)
    assert res_hyb.keypoints_2d.shape == (33, 2)
    assert res_hyb.keypoints_3d.shape == (33, 3)
    assert res_hyb.source == "hybrid"


def test_pillar2_depth_anything_v2():
    frame = np.ones((240, 320, 3), dtype=np.uint8) * 120
    depth_est = DepthAnythingV2Estimator()
    
    dummy_kpts = np.zeros((17, 2), dtype=np.float32)
    dummy_kpts[:, 0] = 160.0
    dummy_kpts[:, 1] = np.linspace(50.0, 200.0, 17)

    res = depth_est.estimate_depth(frame, landmarks_2d=dummy_kpts)
    assert res.relative_depth.shape == (240, 320)
    assert res.metric_depth.shape == (240, 320)
    assert res.point_cloud_3d.shape == (240, 320, 3)
    assert res.surface_normals.shape == (240, 320, 3)
    assert res.metric_depth.min() > 0.0

    lifted_3d = depth_est.lift_landmarks_to_metric_3d(dummy_kpts, res)
    assert lifted_3d.shape == (17, 3)


def test_pillar3_smplx_model_and_fitting():
    model = SMPLXBodyModel()
    pose = np.zeros((21, 3), dtype=np.float32)
    orient = np.zeros(3, dtype=np.float32)
    transl = np.array([0.0, 0.0, 0.75], dtype=np.float32)

    output = model.forward_kinematics(pose, orient, transl)
    assert output.joints_3d.shape == (22, 3)
    assert output.joint_rotations.shape == (22, 3, 3)

    fitter = SMPLXFitter(model=model)
    target_joints = np.zeros((33, 3), dtype=np.float32)
    target_joints[:, 2] = 0.75
    fitted_out = fitter.fit_single_frame(target_joints, num_iters=5)
    assert fitted_out.joints_3d.shape == (22, 3)


def test_pillar4_motion_retargeting():
    model = SMPLXBodyModel()
    output = model.forward_kinematics(
        body_pose=np.zeros((21, 3), dtype=np.float32),
        global_orient=np.zeros(3, dtype=np.float32),
        transl=np.array([0.0, 0.0, 0.74], dtype=np.float32)
    )

    retargeter = SMPLXToHumanoidRetargeter()
    res = retargeter.retarget_frame(output)
    assert res.qpos_robot.shape == (36,)
    assert len(res.actuator_ctrl) == 29
    assert res.is_feasible is True
    assert "left" in res.foot_contacts
    assert "right" in res.foot_contacts


def test_pillar5_mujoco_manipulation_env():
    env = MuJoCoManipulationEnv()
    res_init = env.reset()
    assert bool(res_init.is_balanced) is True
    assert res_init.robot_state.base_position[2] > 0.50

    dummy_ctrl = env.robot.stand_ctrl.copy()
    step_res = env.step(dummy_ctrl)
    assert step_res.robot_state is not None
    assert "left_foot" in step_res.contact_forces
    assert "right_foot" in step_res.contact_forces
    assert step_res.reward >= 0.0


def test_pillar6_trajectory_optimizer():
    optimizer = HumanoidTrajectoryOptimizer(max_iterations=5)
    T = 20
    q_init = np.zeros((T, 36), dtype=np.float32)
    q_init[:, 2] = 0.74

    target_ee = {
        "left_hand": np.zeros((T, 3), dtype=np.float32),
        "right_hand": np.zeros((T, 3), dtype=np.float32)
    }
    target_ee["left_hand"][:, 0] = -0.25
    target_ee["left_hand"][:, 2] = 0.70
    target_ee["right_hand"][:, 0] = 0.25
    target_ee["right_hand"][:, 2] = 0.70

    q_opt, metrics = optimizer.optimize_trajectory(
        q_init_traj=q_init,
        target_ee_positions=target_ee,
        target_contact_timing={"left": np.ones(T), "right": np.ones(T)}
    )

    assert q_opt.shape == (T, 36)
    assert metrics.final_loss <= metrics.initial_loss + 1e-4
    assert metrics.contact_timing_compliance > 90.0


def test_pillar7_bc_plus_ppo_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_p = os.path.join(tmpdir, "policy.pt")
        pipeline = EnhancedBCPlusPPOPipeline(state_dim=70, action_dim=29, history_len=4, model_path=model_p)

        # 1. Behavior Cloning Test with temporal history
        N = 25
        demo_states = np.random.randn(N, 70).astype(np.float32)
        demo_actions = np.random.randn(N, 29).astype(np.float32) * 0.1

        init_loss, final_loss, breakdown = pipeline.pretrain_behavior_cloning(
            demo_states, demo_actions, epochs=3, batch_size=8
        )
        assert final_loss <= init_loss + 0.10
        assert breakdown is not None
        assert breakdown.l_joint >= 0.0

        # 2. PPO Reinforcement Learning Test
        env = MuJoCoManipulationEnv()
        summary = pipeline.train_ppo(env=env, total_steps=64)
        assert summary.loss_reduction_percent > 0.0
        assert os.path.exists(model_p)


def test_observation_action_normalizer():
    normalizer = ObservationActionNormalizer(obs_dim=10, action_dim=4)
    data = np.random.randn(50, 10).astype(np.float32) * 2.0 + 1.5
    normalizer.fit_observations(data)

    obs = np.array(data[0])
    norm_obs = normalizer.normalize_observation(obs)
    denorm_obs = normalizer.denormalize_observation(norm_obs)
    assert np.allclose(obs, denorm_obs, atol=1e-5)

    act = np.array([0.5, -0.2, 1.0, -1.0], dtype=np.float32)
    norm_act = normalizer.normalize_action(act)
    assert np.all(norm_act >= -1.0) and np.all(norm_act <= 1.0)
    denorm_act = normalizer.denormalize_action(norm_act)
    assert np.allclose(act, denorm_act, atol=1e-4)


def test_motion_accuracy_evaluator():
    evaluator = MotionAccuracyEvaluator(fps=30.0)
    T = 30
    q_rob = np.zeros((T, 36), dtype=np.float32)
    q_exp = np.zeros((T, 36), dtype=np.float32)
    q_rob[:, 2] = 0.74
    q_exp[:, 2] = 0.74

    # Slight perturbation
    q_rob[:, 7:36] += 0.02

    feet = {
        "left": np.zeros((T, 3), dtype=np.float32),
        "right": np.zeros((T, 3), dtype=np.float32)
    }
    stances = {
        "left": np.ones(T, dtype=np.float32),
        "right": np.ones(T, dtype=np.float32)
    }

    report = evaluator.evaluate_motion(
        robot_qpos_traj=q_rob,
        expert_qpos_traj=q_exp,
        robot_feet_pos=feet,
        stance_flags=stances
    )

    assert report.mpjpe_mm >= 0.0
    assert report.joint_mae_deg >= 0.0
    assert report.foot_sliding_mm >= 0.0
    assert report.temporal_lag_ms >= 0.0
    assert report.episode_success_rate == 100.0


def test_dagger_trainer():
    normalizer = ObservationActionNormalizer(obs_dim=70, action_dim=29)
    policy = TemporalActorCriticPolicy(state_dim=70, action_dim=29, history_len=1)
    
    T = 20
    ref_traj = np.zeros((T, 36), dtype=np.float32)
    ref_traj[:, 2] = 0.74

    trainer = DAggerTrainer(policy=policy, normalizer=normalizer, reference_trajectory=ref_traj)
    trainer.seed_demonstrations(
        states=np.random.randn(10, 70).astype(np.float32),
        actions=np.random.randn(10, 29).astype(np.float32)
    )

    env = MuJoCoManipulationEnv()
    summary = trainer.run_dagger_iteration(env, round_idx=1, max_steps=10, epochs_per_round=2)
    assert summary.iteration == 1
    assert summary.aggregated_dataset_size >= 10
