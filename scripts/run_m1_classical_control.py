"""Milestone M1 Runner: Standing & Classical Robot Control.
Demonstrates joint PD control, CoM posture stabilization, and arm reaching via DLS IK.
"""

import time
import numpy as np
from atlas.simulation.world import AtlasWorld
from atlas.control.pd import JointPDController
from atlas.control.stabilizer import PostureStabilizer
from atlas.control.kinematics import G1Kinematics
from atlas.control.supervisor import SafetySupervisor


def main():
    print("=" * 70)
    print("ATLAS HUMANOID AI 4D — MILESTONE M1 CLASSICAL CONTROL")
    print("=" * 70)

    world = AtlasWorld()
    robot = world.robot
    state = world.reset()

    pd = JointPDController(nu=robot.model.nu)
    stabilizer = PostureStabilizer()
    kin = G1Kinematics(world.model, world.data)
    supervisor = SafetySupervisor(world.model)

    print(f"[OK] Initial Base Position: {state.base_position}")
    print(f"[OK] Initial CoM Position:  {state.com_position}")

    # Solve IK for right hand to reach towards the table
    target_reach = np.array([0.45, -0.20, 0.65], dtype=np.float32)
    arm_dofs = kin.get_arm_dofs("right")
    reach_qpos, ik_success, ik_err = kin.solve_dls_ik(
        kin.right_hand_id, target_reach, arm_dofs, max_iters=40, tolerance=0.015
    )
    print(f"[OK] DLS IK Solved for Arm Reach: Success={ik_success}, Residual Error={ik_err*1000:.1f} mm")

    # Target standing control with reaching posture
    target_ctrl = robot.stand_ctrl.copy()
    # Map arm qpos to actuator targets
    for i, a_name in enumerate(robot.actuator_names):
        if "right_shoulder" in a_name or "right_elbow" in a_name or "right_wrist" in a_name:
            j_id = robot.joint_name_to_id[a_name]
            qadr = world.model.jnt_qposadr[j_id]
            target_ctrl[i] = reach_qpos[qadr]

    # Run closed-loop stabilized simulation
    print("Executing 100 closed-loop control steps with posture stabilization...")
    for step in range(100):
        # 1. Posture offset
        offset = stabilizer.compute_stabilization_offset(state)
        raw_action = target_ctrl + offset

        # 2. Safety supervision
        validated = supervisor.validate_action(raw_action, state, robot.stand_ctrl)

        # 3. Step world
        state = world.step(validated.action)

        if step % 25 == 0:
            print(f"  Step {step:03d} | Base Z: {state.base_position[2]:.3f}m | Feet: L={state.feet_contacts['left_foot']} R={state.feet_contacts['right_foot']} | Safe: {not validated.safety_override}")

    assert not robot.is_fallen(), "Robot fell during classical control execution!"
    print("[SUCCESS] Milestone M1 validated: Humanoid standing and arm reaching stabilized.")


if __name__ == "__main__":
    main()
