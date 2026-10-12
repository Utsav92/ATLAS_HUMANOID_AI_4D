import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path('assets/unitree_g1/scene.xml')
d = mujoco.MjData(m)
l_w = m.body('left_wrist_yaw_link').id
l_sh = m.body('left_shoulder_pitch_link').id

print("=== Testing Left Arm Joint Effects ===")
tests = [
    ('left_shoulder_pitch_joint', [-1.5, -0.75, 0.0, 0.75, 1.5]),
    ('left_shoulder_roll_joint', [-0.5, 0.0, 0.75, 1.5, 2.0]),
    ('left_shoulder_yaw_joint', [-1.5, -0.75, 0.0, 0.75, 1.5]),
    ('left_elbow_joint', [-0.8, -0.4, 0.0, 0.6, 1.2, 1.8])
]

for jname, vals in tests:
    print(f"\n--- {jname} ---")
    jid = m.joint(jname).id
    adr = m.jnt_qposadr[jid]
    for v in vals:
        d.qpos[:] = 0
        d.qpos[2] = 0.75
        d.qpos[adr] = v
        mujoco.mj_kinematics(m, d)
        rel = d.xpos[l_w] - d.xpos[l_sh]
        print(f"  val={v:+.2f} rad -> wrist rel shoulder: x(fwd)={rel[0]:+.3f}, y(lat)={rel[1]:+.3f}, z(up)={rel[2]:+.3f}, arm_len={np.linalg.norm(rel):.3f}")
