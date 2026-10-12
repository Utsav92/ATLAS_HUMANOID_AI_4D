import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/unitree_g1/scene.xml')
d = mujoco.MjData(m)
l_w = m.body('left_wrist_yaw_link').id
l_sh = m.body('left_shoulder_pitch_link').id
r_w = m.body('right_wrist_yaw_link').id
r_sh = m.body('right_shoulder_pitch_link').id

print("=== Testing Human to G1 Elbow Mapping ===")
for human_bend in [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]:
    # In human: 0.0 is straight, pi is bent.
    # In G1: positive is extended straight, negative is bent tight.
    # Range of G1 elbow is [-1.0472, 2.0944]
    g1_elbow = 1.55 - human_bend * 1.15
    d.qpos[:] = 0
    d.qpos[2] = 0.75
    d.qpos[23] = 1.2 # shoulder roll 1.2 (lifted out)
    d.qpos[25] = g1_elbow
    mujoco.mj_kinematics(m, d)
    rel = d.xpos[l_w] - d.xpos[l_sh]
    reach = np.linalg.norm(rel)
    print(f"Human bend={human_bend:.2f} rad -> G1 elbow={g1_elbow:+.2f} rad -> arm reach={reach:.3f} m, y(lat)={rel[1]:+.3f} m")

print("\n=== Testing Wide Spread & Close ===")
# Wide open:
d.qpos[:] = 0
d.qpos[2] = 0.75
d.qpos[22] = 0.0   # pitch
d.qpos[23] = 1.45  # roll (lifted out wide)
d.qpos[24] = 0.85  # yaw (opened wide)
d.qpos[25] = 1.35  # elbow extended straight
d.qpos[29] = 0.0
d.qpos[30] = -1.45
d.qpos[31] = -0.85
d.qpos[32] = 1.35
mujoco.mj_kinematics(m, d)
spread_wide = np.linalg.norm(d.xpos[l_w] - d.xpos[r_w])
print(f"Wide open pose: wrist spread = {spread_wide:.3f} m")

# Closed in front of chest:
d.qpos[:] = 0
d.qpos[2] = 0.75
d.qpos[22] = -0.4  # pitch (forward)
d.qpos[23] = 0.35  # roll (low)
d.qpos[24] = -0.75 # yaw (adducted across chest)
d.qpos[25] = -0.65 # elbow bent tight
d.qpos[29] = -0.4
d.qpos[30] = -0.35
d.qpos[31] = 0.75
d.qpos[32] = -0.65
mujoco.mj_kinematics(m, d)
spread_closed = np.linalg.norm(d.xpos[l_w] - d.xpos[r_w])
print(f"Closed pose: wrist spread = {spread_closed:.3f} m")
