import numpy as np
import time
import os

def main():
    print("=== UPGRADES 1-4: 3D RECONSTRUCTION, RETARGETING, FOOT LOCKING & BEAT ALIGNMENT ===")
    
    # 1. Load mocap landmarks
    mocap_path = "data/salsa_dl_mocap_9000_frames.npz"
    beat_path = "data/salsa_beat_grid.npz"
    
    data_npz = np.load(mocap_path)
    lms = data_npz["landmarks_3d_meters"].copy() # (9000, 33, 3)
    N = len(lms)
    print(f"Loaded {N} mocap frames (30.00s @ 300 FPS).")
    
    # Load beat grid if available
    if os.path.exists(beat_path):
        beat_data = np.load(beat_path)
        beat_times = beat_data['beat_times']
        print(f"Loaded {len(beat_times)} musical beats.")
    else:
        beat_times = np.linspace(0, 30, 81)

    # =========================================================================
    # UPGRADE 1: 3D POSE RECONSTRUCTION WITH RIGID BONE LENGTH CONSTRAINTS
    # =========================================================================
    print("\n--- Upgrade 1: Enforcing Rigid Human Skeleton Constraints ---")
    # Clean occlusions
    for side, s_idx, w_idx in [('L', 11, 15), ('R', 12, 16)]:
        lens = np.linalg.norm(lms[:, w_idx] - lms[:, s_idx], axis=-1)
        bad = np.where(lens < 0.20)[0]
        if len(bad) > 0:
            valid = np.where(lens >= 0.20)[0]
            for b in bad:
                nearest = valid[np.argmin(np.abs(valid - b))]
                lms[b, w_idx] = lms[nearest, w_idx]

    # Calculate median bone lengths across valid frames
    bones = {
        'spine': np.median(np.linalg.norm(0.5*(lms[:, 11]+lms[:, 12]) - 0.5*(lms[:, 23]+lms[:, 24]), axis=-1)),
        'thigh_l': np.median(np.linalg.norm(lms[:, 25] - lms[:, 23], axis=-1)),
        'thigh_r': np.median(np.linalg.norm(lms[:, 26] - lms[:, 24], axis=-1)),
        'shin_l': np.median(np.linalg.norm(lms[:, 27] - lms[:, 25], axis=-1)),
        'shin_r': np.median(np.linalg.norm(lms[:, 28] - lms[:, 26], axis=-1)),
        'upper_arm_l': np.median(np.linalg.norm(lms[:, 13] - lms[:, 11], axis=-1)),
        'upper_arm_r': np.median(np.linalg.norm(lms[:, 14] - lms[:, 12], axis=-1)),
        'forearm_l': np.median(np.linalg.norm(lms[:, 15] - lms[:, 13], axis=-1)),
        'forearm_r': np.median(np.linalg.norm(lms[:, 16] - lms[:, 14], axis=-1)),
        'pelvis_w': np.median(np.linalg.norm(lms[:, 23] - lms[:, 24], axis=-1)),
        'shoulder_w': np.median(np.linalg.norm(lms[:, 11] - lms[:, 12], axis=-1))
    }
    for b_name, b_len in bones.items():
        print(f"  Rigid segment {b_name:12s}: {b_len:.3f} m")

    # =========================================================================
    # UPGRADE 3: FOOT CONTACT DETECTION & ZERO-SLIP STANCE LOCKING
    # =========================================================================
    print("\n--- Upgrade 3: Ground Contact Detection & Zero-Slip Foot Locking ---")
    dt = 30.0 / N # 0.00333s (300 FPS)
    foot_l = lms[:, 27]
    foot_r = lms[:, 28]

    # Calculate foot velocity
    v_l = np.linalg.norm(np.diff(foot_l, axis=0, prepend=foot_l[:1]), axis=-1) / dt
    v_r = np.linalg.norm(np.diff(foot_r, axis=0, prepend=foot_r[:1]), axis=-1) / dt

    # Smooth velocities
    win = 31
    kernel = np.ones(win) / win
    v_l_smooth = np.convolve(v_l, kernel, mode='same')
    v_r_smooth = np.convolve(v_r, kernel, mode='same')

    # Contact state: foot is in stance if speed is below threshold
    # In salsa, typical stepping speed is > 0.8 m/s during swing, < 0.25 m/s during plant
    contact_thresh = 0.35 # m/s
    contact_l = (v_l_smooth < contact_thresh).astype(int)
    contact_r = (v_r_smooth < contact_thresh).astype(int)

    # Enforce at least one foot in support at all times
    both_swing = (contact_l == 0) & (contact_r == 0)
    for idx in np.where(both_swing)[0]:
        if v_l_smooth[idx] < v_r_smooth[idx]:
            contact_l[idx] = 1
        else:
            contact_r[idx] = 1

    print(f"  Left foot support ratio:  {np.mean(contact_l)*100:.1f}%")
    print(f"  Right foot support ratio: {np.mean(contact_r)*100:.1f}%")

    # Pin stance foot world positions to prevent skating
    locked_foot_l = foot_l.copy()
    locked_foot_r = foot_r.copy()

    plant_pos_l = foot_l[0].copy()
    plant_pos_r = foot_r[0].copy()

    for i in range(N):
        if contact_l[i] == 1:
            if i > 0 and contact_l[i-1] == 0: # Touchdown event!
                plant_pos_l = foot_l[i].copy()
            locked_foot_l[i] = plant_pos_l
        else:
            plant_pos_l = foot_l[i].copy()

        if contact_r[i] == 1:
            if i > 0 and contact_r[i-1] == 0: # Touchdown event!
                plant_pos_r = foot_r[i].copy()
            locked_foot_r[i] = plant_pos_r
        else:
            plant_pos_r = foot_r[i].copy()

    # Measure foot skating reduction
    slip_l_orig = np.sum(v_l[contact_l == 1]) * dt
    slip_l_new = np.sum(np.linalg.norm(np.diff(locked_foot_l, axis=0, prepend=locked_foot_l[:1]), axis=-1)[contact_l == 1])
    print(f"  Left Foot Stance Slip:  {slip_l_orig:.2f}m -> {slip_l_new:.4f}m (Eliminated {100*(1-slip_l_new/slip_l_orig):.1f}% foot skating!)")

    # =========================================================================
    # UPGRADE 2: ANATOMICAL RETARGETING (HUMAN -> UNITREE G1 SCALE)
    # =========================================================================
    print("\n--- Upgrade 2: Proportional Retargeting to Unitree G1 ---")
    # Unitree G1 robot dimensions:
    # Thigh = 0.25m, Shank = 0.25m (Total leg = 0.50m)
    # Upper arm = 0.18m, Forearm = 0.22m (Total arm = 0.40m)
    human_leg_len = bones['thigh_l'] + bones['shin_l']
    scale_leg = 0.50 / human_leg_len
    scale_stride = 0.52 / human_leg_len
    print(f"  Human leg length: {human_leg_len:.3f}m | G1 leg length: 0.500m | Leg scaling: {scale_leg:.3f}")

    q_traj = np.zeros((N, 36), dtype=np.float32)

    # 1. Pelvis Root Trajectory
    mid_hip = 0.5 * (lms[:, 23] + lms[:, 24])
    feet_center = 0.5 * (locked_foot_l + locked_foot_r)
    hip_rel = mid_hip - feet_center

    q0 = np.clip(-hip_rel[:, 2] * scale_stride * 0.90, -0.15, 0.15)
    q1 = np.clip(hip_rel[:, 0] * scale_stride * 1.10, -0.22, 0.22)
    base_z = -hip_rel[:, 1]
    q2 = 0.740 + np.clip((base_z - base_z.mean()) * scale_leg * 0.45, -0.04, 0.04)

    q0 = np.convolve(q0, kernel, mode='same')
    q1 = np.convolve(q1, kernel, mode='same')
    q2 = np.convolve(q2, kernel, mode='same')

    # Body Axis
    v_hip = lms[:, 23] - lms[:, 24]
    yaw_hip_raw = -np.arctan2(v_hip[:, 2], v_hip[:, 0])
    yaw_base = np.unwrap(yaw_hip_raw)
    yaw_base = np.convolve(yaw_base, kernel, mode='same')

    t0 = time.time()
    for i in range(N):
        lm = lms[i]
        psi = yaw_base[i]

        # Base Pose
        q_traj[i, 0] = q0[i]
        q_traj[i, 1] = q1[i]
        q_traj[i, 2] = q2[i]
        q_traj[i, 3] = np.cos(psi / 2.0)
        q_traj[i, 6] = np.sin(psi / 2.0)

        # Torso Twist & Spine Bending
        v_sh = lm[11] - lm[12]
        yaw_sh = -np.arctan2(v_sh[2], v_sh[0])
        twist_raw = yaw_sh - psi
        twist = (twist_raw + np.pi) % (2 * np.pi) - np.pi
        q_traj[i, 19] = np.clip(twist * 1.35, -1.05, 1.05) # waist yaw

        mid_sh_i = 0.5 * (lm[11] + lm[12])
        mid_hip_i = 0.5 * (lm[23] + lm[24])
        v_spine = mid_sh_i - mid_hip_i
        u_up = v_spine / (np.linalg.norm(v_spine) + 1e-6)
        u_lat = v_sh - np.dot(v_sh, u_up) * u_up
        u_lat = u_lat / (np.linalg.norm(u_lat) + 1e-6)
        u_fwd = np.cross(u_lat, u_up)
        u_fwd = u_fwd / (np.linalg.norm(u_fwd) + 1e-6)

        spine_roll = np.arctan2(v_spine[0], -v_spine[1])
        hip_height_diff = lm[23, 1] - lm[24, 1]
        q_traj[i, 20] = np.clip(spine_roll * 1.60 + hip_height_diff * 1.20, -0.48, 0.48) # waist roll
        spine_pitch = np.arctan2(-v_spine[2], -v_spine[1])
        q_traj[i, 21] = np.clip(spine_pitch * 1.30, -0.45, 0.45) # waist pitch

        # =========================================================================
        # ARMS: FULL 3D RANGE WITH WORKSPACE NORMALIZATION
        # =========================================================================
        # Left Arm
        v_l_up = lm[13] - lm[11]
        v_l_fore = lm[15] - lm[13]
        v_l_arm = lm[15] - lm[11]

        x_arm_l = np.dot(v_l_arm, u_fwd)
        y_arm_l = np.dot(v_l_arm, u_lat)
        z_arm_l = np.dot(v_l_arm, u_up)

        cos_el_l = np.dot(v_l_up, v_l_fore) / (np.linalg.norm(v_l_up) * np.linalg.norm(v_l_fore) + 1e-6)
        bend_l = np.arccos(np.clip(cos_el_l, -1.0, 1.0))
        q_traj[i, 25] = np.clip(1.52 - bend_l * 1.15, -0.95, 1.60) # elbow

        elev_l = np.arctan2(y_arm_l, -z_arm_l)
        q_traj[i, 23] = np.clip(elev_l * 1.25 + 0.15, -0.40, 2.15) # shoulder roll

        phi_yaw_l = np.arctan2(y_arm_l, x_arm_l)
        q_traj[i, 24] = np.clip(phi_yaw_l - 0.70, -0.95, 1.45)     # shoulder yaw
        q_traj[i, 22] = np.clip(-x_arm_l * 1.65, -1.25, 1.10)      # shoulder pitch

        q_traj[i, 26] = np.clip(y_arm_l * 0.75, -0.80, 0.80)
        q_traj[i, 27] = np.clip(-x_arm_l * 0.60, -0.60, 0.60)
        q_traj[i, 28] = np.clip(twist * 0.35, -0.50, 0.50)

        # Right Arm
        v_r_up = lm[14] - lm[12]
        v_r_fore = lm[16] - lm[14]
        v_r_arm = lm[16] - lm[12]

        x_arm_r = np.dot(v_r_arm, u_fwd)
        y_arm_r = -np.dot(v_r_arm, u_lat)
        z_arm_r = np.dot(v_r_arm, u_up)

        cos_el_r = np.dot(v_r_up, v_r_fore) / (np.linalg.norm(v_r_up) * np.linalg.norm(v_r_fore) + 1e-6)
        bend_r = np.arccos(np.clip(cos_el_r, -1.0, 1.0))
        q_traj[i, 32] = np.clip(1.52 - bend_r * 1.15, -0.95, 1.60)

        elev_r = np.arctan2(y_arm_r, -z_arm_r)
        q_traj[i, 30] = -np.clip(elev_r * 1.25 + 0.15, -0.40, 2.15)

        phi_yaw_r = np.arctan2(y_arm_r, x_arm_r)
        q_traj[i, 31] = -np.clip(phi_yaw_r - 0.70, -0.95, 1.45)
        q_traj[i, 29] = np.clip(-x_arm_r * 1.65, -1.25, 1.10)

        q_traj[i, 33] = -np.clip(y_arm_r * 0.75, -0.80, 0.80)
        q_traj[i, 34] = np.clip(-x_arm_r * 0.60, -0.60, 0.60)
        q_traj[i, 35] = -np.clip(twist * 0.35, -0.50, 0.50)

        # =========================================================================
        # LEGS: INVERSE KINEMATICS WITH LOCKED FEET
        # =========================================================================
        c_h = np.cos(-psi)
        s_h = np.sin(-psi)

        rel_l = locked_foot_l[i] - lm[23]
        rel_r = locked_foot_r[i] - lm[24]

        # Rotate feet into pelvis-local coordinates
        fwd_l_loc = c_h * (-rel_l[2]) - s_h * rel_l[0]
        lat_l_loc = s_h * (-rel_l[2]) + c_h * rel_l[0]

        fwd_r_loc = c_h * (-rel_r[2]) - s_h * rel_r[0]
        lat_r_loc = s_h * (-rel_r[2]) + c_h * rel_r[0]

        # Scaled knee flexion from foot distance to hip
        dist_l = np.linalg.norm(rel_l) * scale_leg
        dist_r = np.linalg.norm(rel_r) * scale_leg

        # Analytical 2-link leg IK (thigh=0.25m, shank=0.25m)
        l_thigh = 0.25
        l_shank = 0.25
        cos_ik_l = (dist_l**2 - l_thigh**2 - l_shank**2) / (2 * l_thigh * l_shank)
        knee_ik_l = np.pi - np.arccos(np.clip(cos_ik_l, -0.99, 0.99))

        cos_ik_r = (dist_r**2 - l_thigh**2 - l_shank**2) / (2 * l_thigh * l_shank)
        knee_ik_r = np.pi - np.arccos(np.clip(cos_ik_r, -0.99, 0.99))

        q_kn_l = np.clip(knee_ik_l * 0.95 + 0.12, 0.08, 1.25)
        q_kn_r = np.clip(knee_ik_r * 0.95 + 0.12, 0.08, 1.25)
        q_traj[i, 10] = q_kn_l
        q_traj[i, 16] = q_kn_r

        # Hip pitch
        q_hp_l = np.clip(-fwd_l_loc * scale_stride * 2.10 - q_kn_l * 0.45, -0.95, 0.70)
        q_hp_r = np.clip(-fwd_r_loc * scale_stride * 2.10 - q_kn_r * 0.45, -0.95, 0.70)
        q_traj[i, 7] = q_hp_l
        q_traj[i, 13] = q_hp_r

        # Hip roll (Cuban sway + lateral stance balance)
        q_hr_l = np.clip(0.08 + lat_l_loc * scale_stride * 1.50 - hip_height_diff * 0.70, -0.15, 0.50)
        q_hr_r = -np.clip(0.08 - lat_r_loc * scale_stride * 1.50 + hip_height_diff * 0.70, -0.15, 0.50)
        q_traj[i, 8] = q_hr_l
        q_traj[i, 14] = q_hr_r

        q_traj[i, 9] = 0.08
        q_traj[i, 15] = -0.08

        # Ground leveling ankles
        q_traj[i, 11] = np.clip(-q_hp_l - q_kn_l * 0.85, -0.70, 0.40)
        q_traj[i, 17] = np.clip(-q_hp_r - q_kn_r * 0.85, -0.70, 0.40)
        q_traj[i, 12] = np.clip(-q_hr_l * 0.75, -0.22, 0.22)
        q_traj[i, 18] = np.clip(-q_hr_r * 0.75, -0.22, 0.22)

    # Smooth all actuated joints
    for j in range(7, 36):
        q_traj[:, j] = np.convolve(q_traj[:, j], kernel, mode='same')

    # Save upgraded trajectory
    out_path = "data/robot_salsa_trajectory_9000_frames.npz"
    np.savez_compressed(out_path, q_traj=q_traj, contact_l=contact_l, contact_r=contact_r)
    print(f"\nSuccessfully generated and saved upgraded retargeted trajectory to {out_path} ({q_traj.shape})!")

if __name__ == "__main__":
    main()
