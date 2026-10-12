import mujoco
import numpy as np
import cv2

# Load model
model = mujoco.MjModel.from_xml_path("assets/unitree_g1/scene.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=592, width=640)

# Camera looking at robot from front
camera = mujoco.MjvCamera()
camera.lookat = [0.0, 0.0, 0.72]
camera.distance = 2.4
camera.elevation = -10
camera.azimuth = 0 # front facing

npz_data = np.load("data/salsa_dl_mocap_3d.npz")
lms = npz_data["landmarks_3d_meters"] # (900, 33, 3)

def retarget_frame(lm):
    # Joint mapping for Unitree G1 in MuJoCo:
    # 0: floating base (qpos 0..6)
    # 7..12: left leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    # 13..18: right leg (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    # 19..21: waist (yaw, roll, pitch)
    # 22..28: left arm (sh_pitch, sh_roll, sh_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)
    # 29..35: right arm (sh_pitch, sh_roll, sh_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw)

    q = np.zeros(36, dtype=np.float32)

    # Base height & position
    # In mediapipe: Y is down, X is right, Z is forward/back
    # Mid hip
    mid_hip = (lm[23] + lm[24]) * 0.5
    q[0] = 0.0 # x
    q[1] = -mid_hip[0] * 0.6 # lateral sway
    q[2] = 0.76 - mid_hip[1] * 0.1 # z height
    q[3] = 1.0 # quat w

    # Waist rotation (torso twist & sway)
    sh_diff = lm[12] - lm[11]
    hip_diff = lm[24] - lm[23]
    waist_yaw = np.arctan2(sh_diff[2], sh_diff[0]) - np.arctan2(hip_diff[2], hip_diff[0])
    waist_roll = (lm[11, 1] - lm[12, 1]) * 0.6
    q[19] = np.clip(waist_yaw * 1.2, -0.8, 0.8) # waist yaw
    q[20] = np.clip(waist_roll, -0.3, 0.3)      # waist roll
    q[21] = 0.0

    # Left Arm
    # In MediaPipe: 11 is Left Shoulder, 13 is Left Elbow, 15 is Left Wrist
    # Notice: In camera view, dancer's left arm is to the right of her body (positive X in image)
    l_arm_vec = lm[13] - lm[11] # shoulder to elbow
    l_fore_vec = lm[15] - lm[13] # elbow to wrist

    # Shoulder roll (abduction): if elbow is higher/further out
    # lm Y is down, so if lm[13, 1] < lm[11, 1], arm is raised up!
    l_arm_up = (lm[11, 1] - lm[13, 1]) # positive when raised
    l_arm_out = -l_arm_vec[0] # positive when out to the side
    q[22] = np.clip(l_arm_vec[2] * 2.0, -1.5, 1.5) # left sh pitch
    q[23] = np.clip(0.3 + l_arm_out * 2.5 + l_arm_up * 1.8, -0.2, 2.0) # left sh roll (lifting)
    q[24] = np.clip(-l_arm_vec[0] * 1.0, -1.0, 1.0) # left sh yaw

    # Left Elbow flexion
    # Angle between upper arm and fore arm
    cos_l_el = np.dot(l_arm_vec, l_fore_vec) / (np.linalg.norm(l_arm_vec)*np.linalg.norm(l_fore_vec) + 1e-6)
    l_elbow_angle = np.pi - np.arccos(np.clip(cos_l_el, -1.0, 1.0))
    q[25] = np.clip(l_elbow_angle, 0.1, 1.8) # elbow flex

    # Right Arm
    # 12 is Right Shoulder, 14 is Right Elbow, 16 is Right Wrist
    r_arm_vec = lm[14] - lm[12]
    r_fore_vec = lm[16] - lm[14]

    r_arm_up = (lm[12, 1] - lm[14, 1])
    r_arm_out = r_arm_vec[0]
    q[29] = np.clip(r_arm_vec[2] * 2.0, -1.5, 1.5) # right sh pitch
    q[30] = np.clip(-(0.3 + r_arm_out * 2.5 + r_arm_up * 1.8), -2.0, 0.2) # right sh roll (negative for up)
    q[31] = np.clip(r_arm_vec[0] * 1.0, -1.0, 1.0) # right sh yaw

    cos_r_el = np.dot(r_arm_vec, r_fore_vec) / (np.linalg.norm(r_arm_vec)*np.linalg.norm(r_fore_vec) + 1e-6)
    r_elbow_angle = np.pi - np.arccos(np.clip(cos_r_el, -1.0, 1.0))
    q[32] = np.clip(r_elbow_angle, 0.1, 1.8) # elbow flex

    # Legs (Salsa side-steps and cross-steps)
    # 23: L_hip, 25: L_knee, 27: L_ankle
    # 24: R_hip, 26: R_knee, 28: R_ankle
    l_thigh = lm[25] - lm[23]
    r_thigh = lm[26] - lm[24]
    l_shin = lm[27] - lm[25]
    r_shin = lm[28] - lm[26]

    # Knee flexion
    cos_l_kn = np.dot(l_thigh, l_shin) / (np.linalg.norm(l_thigh)*np.linalg.norm(l_shin) + 1e-6)
    cos_r_kn = np.dot(r_thigh, r_shin) / (np.linalg.norm(r_thigh)*np.linalg.norm(r_shin) + 1e-6)
    q[10] = np.clip(np.pi - np.arccos(np.clip(cos_l_kn, -1.0, 1.0)), 0.05, 1.2) # left knee
    q[16] = np.clip(np.pi - np.arccos(np.clip(cos_r_kn, -1.0, 1.0)), 0.05, 1.2) # right knee

    # Hip pitch
    q[7] = np.clip(-q[10] * 0.5 + l_thigh[2] * 1.5, -0.8, 0.8) # left hip pitch
    q[13] = np.clip(-q[16] * 0.5 + r_thigh[2] * 1.5, -0.8, 0.8) # right hip pitch

    # Hip roll (lateral leg extension)
    q[8] = np.clip(-l_thigh[0] * 1.5, -0.3, 0.6) # left hip roll
    q[14] = np.clip(-r_thigh[0] * 1.5, -0.6, 0.3) # right hip roll

    # Ankles
    q[11] = -q[7] - q[10] * 0.5 # left ankle pitch
    q[17] = -q[13] - q[16] * 0.5 # right ankle pitch

    return q

# Test render at frames: 150 (5s), 450 (15s), 750 (25s)
for f_idx, label in [(150, "5s"), (450, "15s"), (750, "25s")]:
    q_retargeted = retarget_frame(lms[f_idx])
    data.qpos[:] = q_retargeted
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=camera)
    img = renderer.render()
    cv2.imwrite(f"videos/exact_retarget_test_{label}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"Rendered exact retargeted pose at {label} to videos/exact_retarget_test_{label}.png")
