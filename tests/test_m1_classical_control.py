"""Unit tests for Milestone M1: Standing and classical robot control."""

import pytest
import numpy as np
from atlas.simulation.world import AtlasWorld
from atlas.control.pd import JointPDController
from atlas.control.stabilizer import PostureStabilizer
from atlas.control.kinematics import G1Kinematics
from atlas.control.supervisor import SafetySupervisor


def test_m1_pd_controller():
    nu = 43
    pd = JointPDController(nu=nu)
    q_des = np.zeros(nu, dtype=np.float32)
    q_curr = np.ones(nu, dtype=np.float32) * 0.1
    qvel = np.zeros(nu, dtype=np.float32)

    tau = pd.compute_torque(q_des, q_curr, qvel)
    assert tau.shape == (nu,)
    # Tau should be negative because q_curr > q_des
    assert np.all(tau < 0.0)


def test_m1_posture_stabilizer():
    world = AtlasWorld()
    state = world.reset()
    stabilizer = PostureStabilizer()

    # Intentionally shift CoM offset in state
    state.com_position[0] += 0.05
    offsets = stabilizer.compute_stabilization_offset(state)
    assert len(offsets) == len(state.ctrl)
    # Ankle pitch should be non-zero to compensate
    assert abs(offsets[stabilizer.left_ankle_pitch_idx]) > 0.0


def test_m1_kinematics_and_ik():
    world = AtlasWorld()
    world.reset()
    kin = G1Kinematics(world.model, world.data)

    init_pos, _ = kin.get_forward_kinematics(kin.right_hand_id)
    assert init_pos.shape == (3,)

    # Target slightly forward (e.g. +5cm along X)
    target = init_pos + np.array([0.05, 0.0, 0.0], dtype=np.float32)

    right_arm_dofs = kin.get_arm_dofs("right")
    assert len(right_arm_dofs) >= 6
    new_qpos, success, err = kin.solve_dls_ik(kin.right_hand_id, target, right_arm_dofs, max_iters=30, tolerance=0.02)
    assert err < 0.05  # Error should converge closely


def test_m1_safety_supervisor():
    world = AtlasWorld()
    state = world.reset()
    supervisor = SafetySupervisor(world.model)

    raw_ctrl = world.robot.stand_ctrl.copy()
    # Apply huge out-of-bounds torque
    raw_ctrl[0] = 999.0

    output = supervisor.validate_action(raw_ctrl, state, world.robot.stand_ctrl)
    assert output.action[0] < 999.0  # Must be clamped
    assert "Actuator saturation clamped" in output.details["reasons"][0]

    # Test tilt violation
    state.base_orientation = np.array([0.5, 0.8, 0.0, 0.0], dtype=np.float32)  # heavily tilted
    output_tilted = supervisor.validate_action(raw_ctrl, state, world.robot.stand_ctrl)
    assert output_tilted.safety_override is True
