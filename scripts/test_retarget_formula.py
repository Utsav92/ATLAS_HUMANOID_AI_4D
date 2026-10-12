import mujoco
import numpy as np
import cv2

model = mujoco.MjModel.from_xml_path("assets/unitree_g1/scene.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=592, width=640)

camera = mujoco.MjvCamera()
camera.lookat = [0.0, 0.0, 0.72]
camera.distance = 2.4
camera.elevation = -10
camera.azimuth = 0

npz_data = np.load("data/salsa_dl_mocap_3d.npz")
lms = npz_data["landmarks_3d_meters"]

def retarget(lm):
    q = np.zeros(36, dtype=np.float32)
    q[2] = 0.78 # base z
    q[3] = 1.0  # quat w

    # --- LEFT ARM ---
    # 11: sh, 13: el, 15: wr
    v_l_up = lm[13] - lm[11]
    v_l_fore = lm[15] - lm[13]
    # In MP: dx is rightward (+ = dancer's left), dy is downward (+ = down, - = up), dz is forward (- = towards camera)
    # When dancer lifts arm: dy becomes negative or small, dx is positive
    # Roll: abduction angle from straight down
    # atan2(dx, dy): if dy > 0, dx > 0 -> angle between 0 and pi/2. If dy < 0, dx > 0 -> angle between pi/2 and pi.
    l_roll = np.arctan2(v_l_up[0], v_l_up[1])
    # Limit to joint range [-1.58, 2.25]
    q[23] = np.clip(l_roll, -0.2, 2.2) # left_shoulder_roll

    # Pitch: forward/backward
    l_pitch = np.arctan2(-v_l_up[2], np.sqrt(v_l_up[0]**2 + v_l_up[1]**2))
    q[22] = np.clip(l_pitch * 1.5, -2.5, 2.5) # left_shoulder_pitch

    # Elbow flexion:
    cos_el_l = np.dot(v_l_up, v_l_fore) / (np.linalg.norm(v_l_up)*np.linalg.norm(v_l_fore) + 1e-6)
    q[25] = np.clip(np.pi - np.arccos(np.clip(cos_el_l, -1.0, 1.0)), 0.0, 2.0)

    # --- RIGHT ARM ---
    # 12: sh, 14: el, 16: wr
    v_r_up = lm[14] - lm[12]
    v_r_fore = lm[16] - lm[14]
    r_roll = np.arctan2(v_r_up[0], v_r_up[1]) # will be negative for right arm!
    q[30] = np.clip(r_roll, -2.2, 0.2) # right_shoulder_roll

    r_pitch = np.arctan2(-v_r_up[2], np.sqrt(v_r_up[0]**2 + v_r_up[1]**2))
    q[29] = np.clip(r_pitch * 1.5, -2.5, 2.5) # right_shoulder_pitch

    cos_el_r = np.dot(v_r_up, v_r_fore) / (np.linalg.norm(v_r_up)*np.linalg.norm(v_r_fore) + 1e-6)
    q[32] = np.clip(np.pi - np.arccos(np.clip(cos_el_r, -1.0, 1.0)), 0.0, 2.0)

    # --- WAIST ---
    sh_line = lm[12] - lm[11]
    hip_line = lm[24] - lm[23]
    twist = np.arctan2(sh_line[2], sh_line[0]) - np.arctan2(hip_line[2], hip_line[0])
    q[19] = np.clip(twist * 1.5, -1.5, 1.5) # waist yaw
    q[20] = np.clip((lm[11, 1] - lm[12, 1]) * 1.2, -0.4, 0.4) # waist roll

    # --- LEGS ---
    # 23: L_hip, 25: L_knee, 27: L_ankle
    # 24: R_hip, 26: R_knee, 28: R_ankle
    v_l_thigh = lm[25] - lm[23]
    v_l_shin = lm[27] - lm[25]
    v_r_thigh = lm[26] - lm[24]
    v_r_shin = lm[28] - lm[26]

    # Hip roll (lateral leg abduction)
    q[8] = np.clip(np.arctan2(v_l_thigh[0], v_l_thigh[1]) * 1.5, -0.3, 0.8) # left hip roll
    q[14] = np.clip(np.arctan2(v_r_thigh[0], v_r_thigh[1]) * 1.5, -0.8, 0.3) # right hip roll

    # Knees
    cos_kn_l = np.dot(v_l_thigh, v_l_shin) / (np.linalg.norm(v_l_thigh)*np.linalg.norm(v_l_shin) + 1e-6)
    cos_kn_r = np.dot(v_r_thigh, v_r_shin) / (np.linalg.norm(v_r_thigh)*np.linalg.norm(v_r_shin) + 1e-6)
    q[10] = np.clip(np.pi - np.arccos(np.clip(cos_kn_l, -1.0, 1.0)), 0.05, 1.5) # left knee
    q[16] = np.clip(np.pi - np.arccos(np.clip(cos_kn_r, -1.0, 1.0)), 0.05, 1.5) # right knee

    # Hip pitch
    q[7] = np.clip(-q[10] * 0.5 + v_l_thigh[2] * 2.0, -1.0, 1.0)
    q[13] = np.clip(-q[16] * 0.5 + v_r_thigh[2] * 2.0, -1.0, 1.0)

    # Ankles
    q[11] = -q[7] - q[10] * 0.5
    q[17] = -q[13] - q[16] * 0.5

    # Lateral root shift
    mid_hip = (lm[23] + lm[24]) * 0.5
    q[1] = np.clip(-mid_hip[0] * 0.8, -0.4, 0.4)

    return q

# Test on frame 450 (15s) and frame 750 (25s)
for f_idx, label in [(150, "5s"), (450, "15s"), (750, "25s")]:
    q_val = retarget(lms[f_idx])
    data.qpos[:] = q_val
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    img = renderer.render()
    cv2.imwrite(f"videos/test_exact_mocap_pose_{label}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Rendered exact pose {label} to videos/test_exact_mocap_pose_{label}.png")
