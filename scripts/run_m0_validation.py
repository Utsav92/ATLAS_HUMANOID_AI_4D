"""Milestone M0 Runner: Windows Setup & MuJoCo Robot Validation.
Performs simulation sanity check, posture verification, and camera acquisition.
"""

import time
import numpy as np
from atlas.simulation.world import AtlasWorld


def main():
    print("=" * 70)
    print("ATLAS HUMANOID AI 4D — MILESTONE M0 VALIDATION")
    print("=" * 70)

    start_time = time.time()
    world = AtlasWorld()
    robot = world.robot

    print(f"[OK] Model Loaded: {robot.model.nq} generalized coords, {robot.model.nu} actuators")
    print(f"[OK] Robot Actuators: {len(robot.actuator_names)}")
    print(f"[OK] Stand Keyframe Pitch/Roll nominal height: {world.data.qpos[2]:.3f} m")

    # Simulate 100 control steps (2 seconds)
    print("Running 100 standing simulation steps...")
    for step in range(100):
        state = world.step(robot.stand_ctrl)
        if step % 25 == 0:
            print(f"  Step {step:03d} | Base Z: {state.base_position[2]:.3f}m | CoM Z: {state.com_position[2]:.3f}m | Fallen: {robot.is_fallen()}")

    assert not robot.is_fallen(), "Robot fell during nominal standing!"
    print("[OK] Standing stability verified.")

    # Vision acquisition test
    rgb = world.render_rgb()
    depth = world.render_depth()
    print(f"[OK] Vision Acquisition: RGB shape {rgb.shape}, Depth shape {depth.shape}")

    # Object query
    cube_pose = world.get_object_pose("red_cube")
    print(f"[OK] Red Cube Pose: {cube_pose}")

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] Milestone M0 completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
