import numpy as np
import mujoco

# Load mocap
d_mocap = np.load('data/salsa_dl_mocap_9000_frames.npz')
lms = d_mocap['landmarks_3d_meters']
N = len(lms)

# Load existing trajectory
d_old = np.load('data/robot_salsa_trajectory_9000_frames.npz')
q_old = d_old['q_traj']

m = mujoco.MjModel.from_xml_path('assets/unitree_g1/scene.xml')
d_sim = mujoco.MjData(m)
lw_id = m.body('left_wrist_yaw_link').id
rw_id = m.body('right_wrist_yaw_link').id

# Measure old spread
old_dists = []
for q in q_old[::10]:
    d_sim.qpos[:] = q
    mujoco.mj_kinematics(m, d_sim)
    old_dists.append(np.linalg.norm(d_sim.xpos[lw_id] - d_sim.xpos[rw_id]))
old_dists = np.array(old_dists)

# Dancer spread
dancer_dists = np.linalg.norm(lms[::10, 15] - lms[::10, 16], axis=-1)

print(f"DANCER wrist spread: min={dancer_dists.min():.3f} m, max={dancer_dists.max():.3f} m, mean={dancer_dists.mean():.3f} m, p90={np.percentile(dancer_dists, 90):.3f} m")
print(f"OLD ROBOT wrist spread: min={old_dists.min():.3f} m, max={old_dists.max():.3f} m, mean={old_dists.mean():.3f} m, p90={np.percentile(old_dists, 90):.3f} m")
