import numpy as np

d_mocap = np.load('data/salsa_dl_mocap_9000_frames.npz')
lms = d_mocap['landmarks_3d_meters']
N = len(lms)

# Pre-smooth landmarks with a 5-frame moving average to remove camera jitter
kernel = np.ones(5) / 5.0
lms_smooth = np.zeros_like(lms)
for j in range(33):
    for c in range(3):
        lms_smooth[:, j, c] = np.convolve(lms[:, j, c], kernel, mode='same')

q_arm = np.zeros((N, 8)) # [sp_l, sr_l, sy_l, el_l, sp_r, sr_r, sy_r, el_r]

for idx in range(N):
    lm = lms_smooth[idx]
    ls, rs = lm[11], lm[12]
    le, re = lm[13], lm[14]
    lw, rw = lm[15], lm[16]
    lh, rh = lm[23], lm[24]

    # Torso frame
    mid_sh = 0.5 * (ls + rs)
    mid_hip = 0.5 * (lh + rh)
    v_spine = mid_sh - mid_hip
    u_up = v_spine / (np.linalg.norm(v_spine) + 1e-6)

    v_sh = ls - rs
    u_lat = v_sh - np.dot(v_sh, u_up) * u_up
    u_lat = u_lat / (np.linalg.norm(u_lat) + 1e-6)
    u_fwd = np.cross(u_lat, u_up)
    u_fwd = u_fwd / (np.linalg.norm(u_fwd) + 1e-6)

    # Left arm
    v_l_up = le - ls
    v_l_fore = lw - le
    v_l_arm = lw - ls

    x_arm_l = np.dot(v_l_arm, u_fwd)
    y_arm_l = np.dot(v_l_arm, u_lat)
    z_arm_l = np.dot(v_l_arm, u_up)

    cos_el_l = np.dot(v_l_up, v_l_fore) / (np.linalg.norm(v_l_up) * np.linalg.norm(v_l_fore) + 1e-6)
    bend_l = np.arccos(np.clip(cos_el_l, -1.0, 1.0))
    q_el_l = np.clip(1.50 - bend_l * 1.15, -0.95, 1.60)

    elev_l = np.arctan2(y_arm_l, -z_arm_l)
    q_sr_l = np.clip(elev_l * 1.15 + 0.10, -0.45, 2.05)

    phi_yaw_l = np.arctan2(y_arm_l, x_arm_l)
    q_sy_l = np.clip(phi_yaw_l - 0.70, -0.95, 1.45)

    q_sp_l = np.clip(-x_arm_l * 1.65, -1.25, 1.10)

    # Right arm
    v_r_up = re - rs
    v_r_fore = rw - re
    v_r_arm = rw - rs

    x_arm_r = np.dot(v_r_arm, u_fwd)
    y_arm_r = -np.dot(v_r_arm, u_lat)
    z_arm_r = np.dot(v_r_arm, u_up)

    cos_el_r = np.dot(v_r_up, v_r_fore) / (np.linalg.norm(v_r_up) * np.linalg.norm(v_r_fore) + 1e-6)
    bend_r = np.arccos(np.clip(cos_el_r, -1.0, 1.0))
    q_el_r = np.clip(1.50 - bend_r * 1.15, -0.95, 1.60)

    elev_r = np.arctan2(y_arm_r, -z_arm_r)
    q_sr_r = -np.clip(elev_r * 1.15 + 0.10, -0.45, 2.05)

    phi_yaw_r = np.arctan2(y_arm_r, x_arm_r)
    q_sy_r = -np.clip(phi_yaw_r - 0.70, -0.95, 1.45)

    q_sp_r = np.clip(-x_arm_r * 1.65, -1.25, 1.10)

    q_arm[idx] = [q_sp_l, q_sr_l, q_sy_l, q_el_l, q_sp_r, q_sr_r, q_sy_r, q_el_r]

# Check max frame-to-frame delta (continuity check at 300 fps)
max_dq = np.max(np.abs(np.diff(q_arm, axis=0)), axis=0)
print(f"Max frame-to-frame dq across 9000 frames: {max_dq}")
for i, name in enumerate(['L_pitch', 'L_roll', 'L_yaw', 'L_el', 'R_pitch', 'R_roll', 'R_yaw', 'R_el']):
    print(f"  {name:8s}: max dq = {max_dq[i]:.4f} rad/frame (at 300fps, velocity = {max_dq[i]*300:.2f} rad/s)")
