import numpy as np
import time

def main():
    print("Loading 9,000-frame 3D MoCap landmarks...")
    data_npz = np.load("data/salsa_dl_mocap_9000_frames.npz")
    lms = data_npz["landmarks_3d_meters"].copy() # (9000, 33, 3)
    N = len(lms)
    print(f"Loaded {N} frames @ 300 FPS ({N/300.0:.2f}s duration).")

    # 1. Clean up any occluded arm landmarks where arm length < 0.20m
    for side, s_idx, w_idx in [('L', 11, 15), ('R', 12, 16)]:
        lens = np.linalg.norm(lms[:, w_idx] - lms[:, s_idx], axis=-1)
        bad = np.where(lens < 0.20)[0]
        if len(bad) > 0:
            valid = np.where(lens >= 0.20)[0]
            for b in bad:
                nearest = valid[np.argmin(np.abs(valid - b))]
                lms[b, w_idx] = lms[nearest, w_idx]
    # Handle elbow occlusion similarly
    for side, s_idx, e_idx in [('L', 11, 13), ('R', 12, 14)]:
        lens = np.linalg.norm(lms[:, e_idx] - lms[:, s_idx], axis=-1)
        bad = np.where(lens < 0.12)[0]
        if len(bad) > 0:
            valid = np.where(lens >= 0.12)[0]
            for b in bad:
                nearest = valid[np.argmin(np.abs(valid - b))]
                lms[b, e_idx] = lms[nearest, e_idx]

    q_traj = np.zeros((N, 36), dtype=np.float32)

    # Base stance reference
    mid_hip = 0.5 * (lms[:, 23] + lms[:, 24]) # (N, 3)
    l_foot = lms[:, 27]
    r_foot = lms[:, 28]
    feet_center = 0.5 * (l_foot + r_foot)
    hip_rel = mid_hip - feet_center

    # Temporal smoothing kernel (~100ms at 300 FPS)
    win = 31
    kernel = np.ones(win) / win

    # Pelvis translation in local/world frame
    q0 = np.clip(-hip_rel[:, 2] * 0.40, -0.16, 0.16) # Sagittal step shift
    q1 = np.clip(hip_rel[:, 0] * 0.70, -0.22, 0.22)  # Lateral Cuban sway
    base_z = -hip_rel[:, 1]
    q2 = 0.740 + np.clip((base_z - base_z.mean()) * 0.18, -0.04, 0.04)

    # Smooth pelvis translation
    q0 = np.convolve(q0, kernel, mode='same')
    q1 = np.convolve(q1, kernel, mode='same')
    q2 = np.convolve(q2, kernel, mode='same')

    # Body Axis (Full Dancer Hip Yaw Tracking)
    v_hip = lms[:, 23] - lms[:, 24]
    yaw_hip_raw = -np.arctan2(v_hip[:, 2], v_hip[:, 0])
    yaw_base = np.unwrap(yaw_hip_raw)
    yaw_base = np.convolve(yaw_base, kernel, mode='same')

    t0 = time.time()
    for i in range(N):
        lm = lms[i]
        psi = yaw_base[i]

        # Floating base (Matches dancer body axis!)
        q_traj[i, 0] = q0[i]
        q_traj[i, 1] = q1[i]
        q_traj[i, 2] = q2[i]
        q_traj[i, 3] = np.cos(psi / 2.0)
        q_traj[i, 6] = np.sin(psi / 2.0)

        # =========================================================================
        # TORSO ANATOMICAL FRAME, TWISTING & BENDING
        # =========================================================================
        v_sh = lm[11] - lm[12]
        yaw_sh = -np.arctan2(v_sh[2], v_sh[0])

        # Relative twist between shoulders and hips
        twist_raw = yaw_sh - psi
        twist = (twist_raw + np.pi) % (2 * np.pi) - np.pi
        q_traj[i, 19] = np.clip(twist * 1.35, -1.05, 1.05) # waist yaw: dynamic spine twist

        # Spine basis vectors in camera space
        mid_sh_i = 0.5 * (lm[11] + lm[12])
        mid_hip_i = 0.5 * (lm[23] + lm[24])
        v_spine = mid_sh_i - mid_hip_i
        u_up = v_spine / (np.linalg.norm(v_spine) + 1e-6)

        # Orthonormalize lateral vector
        u_lat = v_sh - np.dot(v_sh, u_up) * u_up
        u_lat = u_lat / (np.linalg.norm(u_lat) + 1e-6)
        u_fwd = np.cross(u_lat, u_up) # True chest forward vector
        u_fwd = u_fwd / (np.linalg.norm(u_fwd) + 1e-6)

        # Lateral & Sagittal Spine Bending
        spine_roll = np.arctan2(v_spine[0], -v_spine[1])
        hip_height_diff = lm[23, 1] - lm[24, 1]
        q_traj[i, 20] = np.clip(spine_roll * 1.60 + hip_height_diff * 1.20, -0.48, 0.48) # waist roll

        spine_pitch = np.arctan2(-v_spine[2], -v_spine[1])
        q_traj[i, 21] = np.clip(spine_pitch * 1.30, -0.45, 0.45) # waist pitch

        # =========================================================================
        # ARMS: FULL 3D RANGE OF MOTION (SPREAD OPEN WIDE & CLOSE IN TIGHT)
        # =========================================================================
        # Left Arm: 11: sh, 13: el, 15: wr
        v_l_up = lm[13] - lm[11]
        v_l_fore = lm[15] - lm[13]
        v_l_arm = lm[15] - lm[11]

        x_arm_l = np.dot(v_l_arm, u_fwd)
        y_arm_l = np.dot(v_l_arm, u_lat)
        z_arm_l = np.dot(v_l_arm, u_up)

        # 1. Left Elbow: full extension (straight out +1.52) to flexion (bent in -0.95)
        cos_el_l = np.dot(v_l_up, v_l_fore) / (np.linalg.norm(v_l_up) * np.linalg.norm(v_l_fore) + 1e-6)
        bend_l = np.arccos(np.clip(cos_el_l, -1.0, 1.0))
        q_traj[i, 25] = np.clip(1.52 - bend_l * 1.15, -0.95, 1.60) # left elbow

        # 2. Left Shoulder Roll: lateral abduction (spread out wide & high)
        elev_l = np.arctan2(y_arm_l, -z_arm_l)
        q_traj[i, 23] = np.clip(elev_l * 1.25 + 0.15, -0.40, 2.15) # left shoulder roll

        # 3. Left Shoulder Yaw: horizontal sweep (opening wide vs closing across chest)
        phi_yaw_l = np.arctan2(y_arm_l, x_arm_l)
        q_traj[i, 24] = np.clip(phi_yaw_l - 0.70, -0.95, 1.45)     # left shoulder yaw

        # 4. Left Shoulder Pitch: forward reach (neg) vs backward behind back (pos)
        q_traj[i, 22] = np.clip(-x_arm_l * 1.65, -1.25, 1.10)      # left shoulder pitch

        # 5. Left Wrist: articulation & styling
        q_traj[i, 26] = np.clip(y_arm_l * 0.75, -0.80, 0.80)       # left wrist roll
        q_traj[i, 27] = np.clip(-x_arm_l * 0.60, -0.60, 0.60)     # left wrist pitch
        q_traj[i, 28] = np.clip(twist * 0.35, -0.50, 0.50)         # left wrist yaw

        # Right Arm: 12: sh, 14: el, 16: wr
        v_r_up = lm[14] - lm[12]
        v_r_fore = lm[16] - lm[14]
        v_r_arm = lm[16] - lm[12]

        x_arm_r = np.dot(v_r_arm, u_fwd)
        y_arm_r = -np.dot(v_r_arm, u_lat) # positive outward to right
        z_arm_r = np.dot(v_r_arm, u_up)

        # 1. Right Elbow
        cos_el_r = np.dot(v_r_up, v_r_fore) / (np.linalg.norm(v_r_up) * np.linalg.norm(v_r_fore) + 1e-6)
        bend_r = np.arccos(np.clip(cos_el_r, -1.0, 1.0))
        q_traj[i, 32] = np.clip(1.52 - bend_r * 1.15, -0.95, 1.60) # right elbow

        # 2. Right Shoulder Roll (negative is outward/abduction in G1)
        elev_r = np.arctan2(y_arm_r, -z_arm_r)
        q_traj[i, 30] = -np.clip(elev_r * 1.25 + 0.15, -0.40, 2.15) # right shoulder roll

        # 3. Right Shoulder Yaw (negative is outward wide, positive is closing across chest)
        phi_yaw_r = np.arctan2(y_arm_r, x_arm_r)
        q_traj[i, 31] = -np.clip(phi_yaw_r - 0.70, -0.95, 1.45)    # right shoulder yaw

        # 4. Right Shoulder Pitch
        q_traj[i, 29] = np.clip(-x_arm_r * 1.65, -1.25, 1.10)      # right shoulder pitch

        # 5. Right Wrist
        q_traj[i, 33] = -np.clip(y_arm_r * 0.75, -0.80, 0.80)      # right wrist roll
        q_traj[i, 34] = np.clip(-x_arm_r * 0.60, -0.60, 0.60)     # right wrist pitch
        q_traj[i, 35] = -np.clip(twist * 0.35, -0.50, 0.50)        # right wrist yaw

        # =========================================================================
        # LEGS: RELATIVE TO TURNED PELVIS FRAME (CROSS STEPS & BENDS)
        # =========================================================================
        c_h = np.cos(-psi)
        s_h = np.sin(-psi)

        rel_l = lm[27] - lm[23]
        rel_r = lm[28] - lm[24]

        # Rotate feet into pelvis-local coordinates
        fwd_l_loc = c_h * (-rel_l[2]) - s_h * rel_l[0]
        lat_l_loc = s_h * (-rel_l[2]) + c_h * rel_l[0]

        fwd_r_loc = c_h * (-rel_r[2]) - s_h * rel_r[0]
        lat_r_loc = s_h * (-rel_r[2]) + c_h * rel_r[0]

        # Knees
        thigh_l = lm[25] - lm[23]
        shin_l = lm[27] - lm[25]
        thigh_r = lm[26] - lm[24]
        shin_r = lm[28] - lm[26]

        cos_kn_l = np.dot(thigh_l, shin_l) / (np.linalg.norm(thigh_l)*np.linalg.norm(shin_l) + 1e-6)
        bend_kn_l = np.arccos(np.clip(cos_kn_l, -1.0, 1.0))
        cos_kn_r = np.dot(thigh_r, shin_r) / (np.linalg.norm(thigh_r)*np.linalg.norm(shin_r) + 1e-6)
        bend_kn_r = np.arccos(np.clip(cos_kn_r, -1.0, 1.0))

        q_kn_l = np.clip(bend_kn_l * 1.05 + 0.08, 0.08, 1.25)
        q_kn_r = np.clip(bend_kn_r * 1.05 + 0.08, 0.08, 1.25)
        q_traj[i, 10] = q_kn_l
        q_traj[i, 16] = q_kn_r

        # Hip pitch: forward step (negative) & back step (positive) along line of dance
        q_hp_l = np.clip(-fwd_l_loc * 1.45 - q_kn_l * 0.45, -0.95, 0.70)
        q_hp_r = np.clip(-fwd_r_loc * 1.45 - q_kn_r * 0.45, -0.95, 0.70)
        q_traj[i, 7] = q_hp_l
        q_traj[i, 13] = q_hp_r

        # Hip roll: Cuban motion
        q_hr_l = np.clip(0.08 + lat_l_loc * 0.90 - hip_height_diff * 0.70, -0.15, 0.50)
        q_hr_r = -np.clip(0.08 - lat_r_loc * 0.90 + hip_height_diff * 0.70, -0.15, 0.50)
        q_traj[i, 8] = q_hr_l
        q_traj[i, 14] = q_hr_r

        # Hip yaw: slight turnout
        q_traj[i, 9] = 0.08
        q_traj[i, 15] = -0.08

        # Ankles
        q_traj[i, 11] = np.clip(-q_hp_l - q_kn_l * 0.85, -0.70, 0.40)
        q_traj[i, 17] = np.clip(-q_hp_r - q_kn_r * 0.85, -0.70, 0.40)
        q_traj[i, 12] = np.clip(-q_hr_l * 0.75, -0.22, 0.22)
        q_traj[i, 18] = np.clip(-q_hr_r * 0.75, -0.22, 0.22)

    # Smooth all actuated joint trajectories across time
    for j in range(7, 36):
        q_traj[:, j] = np.convolve(q_traj[:, j], kernel, mode='same')

    print(f"Computed accurate 9,000-frame joint trajectories in {time.time()-t0:.2f}s!")
    out_path = "data/robot_salsa_trajectory_9000_frames.npz"
    np.savez_compressed(out_path, q_traj=q_traj)
    print(f"Saved to {out_path} ({q_traj.shape})")

if __name__ == "__main__":
    main()
