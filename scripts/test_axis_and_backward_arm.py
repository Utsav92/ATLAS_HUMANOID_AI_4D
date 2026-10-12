import mujoco
import cv2
import numpy as np

def main():
    m = mujoco.MjModel.from_xml_path("assets/unitree_g1/scene.xml")
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=592, width=640)
    cam = mujoco.MjvCamera()
    cam.lookat = [0.0, 0.0, 0.65]
    cam.distance = 2.45
    cam.elevation = 6
    cam.azimuth = 180

    data_npz = np.load("data/salsa_dl_mocap_9000_frames.npz")
    lms = data_npz["landmarks_3d_meters"]

    def retarget_dancer_axis(lm):
        q = np.zeros(36, dtype=np.float32)
        q[2] = 0.74

        # 1. Body Orientation (Matching Hip Yaw from Dancer)
        v_hip = lm[23] - lm[24]
        yaw_hip = -np.arctan2(v_hip[2], v_hip[0])
        q_yaw = yaw_hip
        q[3] = np.cos(q_yaw / 2.0)
        q[6] = np.sin(q_yaw / 2.0)

        mid_hip = 0.5 * (lm[23] + lm[24])
        l_foot = lm[27]
        r_foot = lm[28]
        feet_center = 0.5 * (l_foot + r_foot)
        hip_rel = mid_hip - feet_center
        q[1] = np.clip(hip_rel[0] * 0.70, -0.22, 0.22)
        q[0] = np.clip(-hip_rel[2] * 0.40, -0.15, 0.15)

        # 2. Torso Anatomical Frame & Twisting
        v_sh = lm[11] - lm[12]
        yaw_sh = -np.arctan2(v_sh[2], v_sh[0])

        twist_raw = yaw_sh - yaw_hip
        twist = (twist_raw + np.pi) % (2 * np.pi) - np.pi
        q[19] = np.clip(twist * 1.35, -1.05, 1.05) # waist yaw: dynamic spine twist

        mid_sh = 0.5 * (lm[11] + lm[12])
        v_spine = mid_sh - mid_hip
        u_up = v_spine / (np.linalg.norm(v_spine) + 1e-6)
        u_lat = v_sh / (np.linalg.norm(v_sh) + 1e-6)
        u_fwd = np.cross(u_lat, u_up) # True chest forward vector
        u_fwd = u_fwd / (np.linalg.norm(u_fwd) + 1e-6)

        # Lateral & Sagittal Spine Bending
        spine_roll = np.arctan2(v_spine[0], -v_spine[1])
        hip_height_diff = lm[23, 1] - lm[24, 1]
        q[20] = np.clip(spine_roll * 1.60 + hip_height_diff * 1.20, -0.48, 0.48) # waist roll

        spine_pitch = np.arctan2(-v_spine[2], -v_spine[1])
        q[21] = np.clip(spine_pitch * 1.30, -0.45, 0.45) # waist pitch

        # 3. Arms (Projected in Torso Anatomical Frame)
        v_l_up = lm[13] - lm[11]
        v_l_fore = lm[15] - lm[13]
        v_l_hand = lm[15] - lm[11]

        l_fwd = np.dot(v_l_hand, u_fwd) # positive = in front of chest, negative = behind back
        l_lat = np.dot(v_l_up, u_lat)
        l_vert = np.dot(v_l_up, u_up)

        elev_l = np.arctan2(l_lat, -l_vert)
        q[23] = np.clip(elev_l * 1.15 + 0.15, -0.10, 1.80)
        # When l_fwd < 0 (reaching backward), shoulder_pitch is POSITIVE!
        q[22] = np.clip(-l_fwd * 1.50, -0.75, 0.75)

        cos_el_l = np.dot(v_l_up, v_l_fore) / (np.linalg.norm(v_l_up)*np.linalg.norm(v_l_fore) + 1e-6)
        bend_l = np.arccos(np.clip(cos_el_l, -1.0, 1.0))
        q[25] = np.clip(0.12 - bend_l * 1.05, -0.95, 0.35)

        fore_inward_l = -np.dot(v_l_fore, u_lat) / (np.linalg.norm(v_l_fore) + 1e-6)
        q[24] = np.clip(0.15 + fore_inward_l * 0.45 + l_fwd * 0.40, -0.35, 0.85)

        v_r_up = lm[14] - lm[12]
        v_r_fore = lm[16] - lm[14]
        v_r_hand = lm[16] - lm[12]

        r_fwd = np.dot(v_r_hand, u_fwd)
        r_lat = -np.dot(v_r_up, u_lat)
        r_vert = np.dot(v_r_up, u_up)

        elev_r = np.arctan2(r_lat, -r_vert)
        q[30] = -np.clip(elev_r * 1.15 + 0.15, -0.10, 1.80)
        q[29] = np.clip(-r_fwd * 1.50, -0.75, 0.75)

        cos_el_r = np.dot(v_r_up, v_r_fore) / (np.linalg.norm(v_r_up)*np.linalg.norm(v_r_fore) + 1e-6)
        bend_r = np.arccos(np.clip(cos_el_r, -1.0, 1.0))
        q[32] = np.clip(0.12 - bend_r * 1.05, -0.95, 0.35)

        fore_inward_r = np.dot(v_r_fore, u_lat) / (np.linalg.norm(v_r_fore) + 1e-6)
        q[31] = -np.clip(0.15 + fore_inward_r * 0.45 + r_fwd * 0.40, -0.35, 0.85)

        # 4. Legs (Relative to Turned Pelvis Frame)
        c_h = np.cos(-yaw_hip)
        s_h = np.sin(-yaw_hip)

        rel_l = lm[27] - lm[23]
        rel_r = lm[28] - lm[24]

        fwd_l_loc = c_h * (-rel_l[2]) - s_h * rel_l[0]
        lat_l_loc = s_h * (-rel_l[2]) + c_h * rel_l[0]

        fwd_r_loc = c_h * (-rel_r[2]) - s_h * rel_r[0]
        lat_r_loc = s_h * (-rel_r[2]) + c_h * rel_r[0]

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
        q[10] = q_kn_l
        q[16] = q_kn_r

        q_hp_l = np.clip(-fwd_l_loc * 1.45 - q_kn_l * 0.45, -0.95, 0.70)
        q_hp_r = np.clip(-fwd_r_loc * 1.45 - q_kn_r * 0.45, -0.95, 0.70)
        q[7] = q_hp_l
        q[13] = q_hp_r

        q_hr_l = np.clip(0.08 + lat_l_loc * 0.90 - hip_height_diff * 0.70, -0.15, 0.50)
        q_hr_r = -np.clip(0.08 - lat_r_loc * 0.90 + hip_height_diff * 0.70, -0.15, 0.50)
        q[8] = q_hr_l
        q[14] = q_hr_r

        q[9] = 0.08
        q[15] = -0.08

        q[11] = np.clip(-q_hp_l - q_kn_l * 0.85, -0.70, 0.40)
        q[17] = np.clip(-q_hp_r - q_kn_r * 0.85, -0.70, 0.40)
        q[12] = np.clip(-q_hr_l * 0.75, -0.22, 0.22)
        q[18] = np.clip(-q_hr_r * 0.75, -0.22, 0.22)

        return q

    for f, label in [(1500, 't05'), (4500, 't15'), (7500, 't25')]:
        q_val = retarget_dancer_axis(lms[f])
        d.qpos[:] = q_val
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        cv2.imwrite(f'videos/test_axis_match_{label}.png', img)
        print(f"Rendered {label}: L_pitch={q_val[22]:.2f}")

    print("All test axis matches rendered!")

if __name__ == "__main__":
    main()
