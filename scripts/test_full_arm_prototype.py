import numpy as np
import mujoco

d_mocap = np.load('data/salsa_dl_mocap_9000_frames.npz')
lms = d_mocap['landmarks_3d_meters']
N = len(lms)

m = mujoco.MjModel.from_xml_path('assets/unitree_g1/scene.xml')
d_sim = mujoco.MjData(m)
lw_id = m.body('left_wrist_yaw_link').id
rw_id = m.body('right_wrist_yaw_link').id

# Let's test on 100 sample frames first
indices = np.linspace(0, N-1, 100).astype(int)
sim_dists = []
dancer_dists = []

for idx in indices:
    lm = lms[idx]
    ls, rs = lm[11], lm[12]
    le, re = lm[13], lm[14]
    lw, rw = lm[15], lm[16]
    lh, rh = lm[23], lm[24]

    d_dance = np.linalg.norm(lw - rw)
    dancer_dists.append(d_dance)

    # Torso frame
    mid_sh = 0.5 * (ls + rs)
    mid_hip = 0.5 * (lh + rh)
    v_spine = mid_sh - mid_hip
    u_up = v_spine / (np.linalg.norm(v_spine) + 1e-6)

    v_sh = ls - rs
    # Gram-Schmidt orthonormalization
    u_lat = v_sh - np.dot(v_sh, u_up) * u_up
    u_lat = u_lat / (np.linalg.norm(u_lat) + 1e-6)
    u_fwd = np.cross(u_lat, u_up)
    u_fwd = u_fwd / (np.linalg.norm(u_fwd) + 1e-6)

    # Left arm vectors
    v_l_up = le - ls
    v_l_fore = lw - le
    v_l_arm = lw - ls

    x_arm_l = np.dot(v_l_arm, u_fwd)
    y_arm_l = np.dot(v_l_arm, u_lat)
    z_arm_l = np.dot(v_l_arm, u_up)

    # Left elbow
    cos_el_l = np.dot(v_l_up, v_l_fore) / (np.linalg.norm(v_l_up) * np.linalg.norm(v_l_fore) + 1e-6)
    bend_l = np.arccos(np.clip(cos_el_l, -1.0, 1.0))
    q_el_l = np.clip(1.50 - bend_l * 1.15, -0.95, 1.60)

    # Left shoulder roll (lateral elevation)
    elev_l = np.arctan2(y_arm_l, -z_arm_l)
    q_sr_l = np.clip(elev_l * 1.15 + 0.10, -0.45, 2.05)

    # Left shoulder yaw (horizontal sweep)
    phi_yaw_l = np.arctan2(y_arm_l, x_arm_l)
    q_sy_l = np.clip(phi_yaw_l - 0.70, -0.95, 1.45)

    # Left shoulder pitch (forward/backward)
    q_sp_l = np.clip(-x_arm_l * 1.65, -1.25, 1.10)

    # Right arm vectors
    v_r_up = re - rs
    v_r_fore = rw - re
    v_r_arm = rw - rs

    x_arm_r = np.dot(v_r_arm, u_fwd)
    y_arm_r = -np.dot(v_r_arm, u_lat) # positive outward
    z_arm_r = np.dot(v_r_arm, u_up)

    cos_el_r = np.dot(v_r_up, v_r_fore) / (np.linalg.norm(v_r_up) * np.linalg.norm(v_r_fore) + 1e-6)
    bend_r = np.arccos(np.clip(cos_el_r, -1.0, 1.0))
    q_el_r = np.clip(1.50 - bend_r * 1.15, -0.95, 1.60)

    elev_r = np.arctan2(y_arm_r, -z_arm_r)
    q_sr_r = -np.clip(elev_r * 1.15 + 0.10, -0.45, 2.05)

    phi_yaw_r = np.arctan2(y_arm_r, x_arm_r)
    q_sy_r = -np.clip(phi_yaw_r - 0.70, -0.95, 1.45)

    q_sp_r = np.clip(-x_arm_r * 1.65, -1.25, 1.10)

    d_sim.qpos[:] = 0
    d_sim.qpos[2] = 0.75
    d_sim.qpos[22] = q_sp_l
    d_sim.qpos[23] = q_sr_l
    d_sim.qpos[24] = q_sy_l
    d_sim.qpos[25] = q_el_l

    d_sim.qpos[29] = q_sp_r
    d_sim.qpos[30] = q_sr_r
    d_sim.qpos[31] = q_sy_r
    d_sim.qpos[32] = q_el_r

    mujoco.mj_kinematics(m, d_sim)
    d_sim_val = np.linalg.norm(d_sim.xpos[lw_id] - d_sim.xpos[rw_id])
    sim_dists.append(d_sim_val)

sim_dists = np.array(sim_dists)
dancer_dists = np.array(dancer_dists)

print(f"NEW Sim Wrist Spread: min={sim_dists.min():.3f} m, max={sim_dists.max():.3f} m, mean={sim_dists.mean():.3f} m")
print(f"Dancer Wrist Spread:  min={dancer_dists.min():.3f} m, max={dancer_dists.max():.3f} m, mean={dancer_dists.mean():.3f} m")
