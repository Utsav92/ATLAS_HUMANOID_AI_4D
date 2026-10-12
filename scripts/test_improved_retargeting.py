"""
Test script for highly accurate 3D kinematic retargeting to Unitree G1 in MuJoCo.
Evaluates joint accuracy on keyframe 180 (the dynamic TikTok dance pose).
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mujoco
import numpy as np
import os

def angle_3pts(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    return np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))

def extract_g1_qpos_accurate(lm, prev_qpos=None, alpha=0.8):
    qpos = np.zeros(36, dtype=np.float64)
    qpos[3] = 1.0  # quat w
    qpos[2] = 0.793 # default height

    if lm is None:
        if prev_qpos is not None:
            return prev_qpos.copy()
        return qpos

    # 1. Pelvis Orientation & Position
    # 23: left hip, 24: right hip
    dx_hip = lm[24].x - lm[23].x
    dy_hip = lm[24].y - lm[23].y
    dz_hip = lm[24].z - lm[23].z

    # Pelvis yaw angle (around vertical z in MuJoCo)
    # Notice: camera faces dancer. When dancer turns left, right hip comes forward (dz < 0).
    pelvis_yaw = np.arctan2(-dz_hip, dx_hip) * 0.8
    pelvis_yaw = np.clip(pelvis_yaw, -0.6, 0.6)

    # Convert yaw to quaternion [qw, qx, qy, qz]
    qpos[3] = np.cos(pelvis_yaw / 2.0)
    qpos[6] = np.sin(pelvis_yaw / 2.0)

    # Pelvis center
    pelvis_x = (lm[23].x + lm[24].x) / 2.0
    pelvis_y = (lm[23].y + lm[24].y) / 2.0
    pelvis_z = (lm[23].z + lm[24].z) / 2.0

    qpos[0] = np.clip((pelvis_z) * 0.5, -0.2, 0.2)          # forward/backward shift
    qpos[1] = np.clip(-(pelvis_x - 0.5) * 0.5, -0.25, 0.25)  # lateral sway

    # 2. Torso / Waist Kinematics
    # 11: left shoulder, 12: right shoulder
    dx_sh = lm[12].x - lm[11].x
    dz_sh = lm[12].z - lm[11].z
    shoulder_yaw = np.arctan2(-dz_sh, dx_sh)
    waist_yaw = np.clip((shoulder_yaw - pelvis_yaw) * 0.9, -0.6, 0.6)
    qpos[19] = waist_yaw

    # Torso lean (pitch and roll)
    sh_mid_x = (lm[11].x + lm[12].x) / 2.0
    sh_mid_y = (lm[11].y + lm[12].y) / 2.0
    sh_mid_z = (lm[11].z + lm[12].z) / 2.0
    torso_lean_pitch = np.clip((sh_mid_z - pelvis_z) * 1.5, -0.3, 0.3)
    torso_lean_roll  = np.clip(-(sh_mid_x - pelvis_x) * 1.5, -0.3, 0.3)
    qpos[20] = torso_lean_roll   # waist roll
    qpos[21] = torso_lean_pitch  # waist pitch

    # 3. Leg Kinematics (Hips, Knees, Ankles)
    # Left leg: hip(23), knee(25), ankle(27)
    # Right leg: hip(24), knee(26), ankle(28)
    
    # Left thigh vector
    l_th_x = lm[25].x - lm[23].x
    l_th_y = lm[25].y - lm[23].y
    l_th_z = lm[25].z - lm[23].z

    # Right thigh vector
    r_th_x = lm[26].x - lm[24].x
    r_th_y = lm[26].y - lm[24].y
    r_th_z = lm[26].z - lm[24].z

    # Hip Pitch (Sagittal stepping)
    # In G1: negative = step forward (+x), positive = step backward (-x)
    # In BlazePose: dz < 0 is forward, dz > 0 is backward
    l_hip_pitch = np.clip((l_th_z / (abs(l_th_y) + 1e-4)) * 1.6, -1.2, 1.2)
    r_hip_pitch = np.clip((r_th_z / (abs(r_th_y) + 1e-4)) * 1.6, -1.2, 1.2)

    # Hip Roll (Coronal abduction)
    # G1: left hip roll positive abducts left; right hip roll negative abducts right
    l_hip_roll = np.clip((l_th_x / (abs(l_th_y) + 1e-4)) * 1.5, -0.2, 0.8)
    r_hip_roll = np.clip((r_th_x / (abs(r_th_y) + 1e-4)) * 1.5, -0.8, 0.2)

    # Knee Flexion
    l_knee = np.clip((np.pi - angle_3pts(lm[23], lm[25], lm[27])) * 1.3, 0.0, 2.0)
    r_knee = np.clip((np.pi - angle_3pts(lm[24], lm[26], lm[28])) * 1.3, 0.0, 2.0)

    # Ankle pitch & roll to keep feet flat
    l_ankle_pitch = np.clip(-(l_hip_pitch + l_knee * 0.55), -0.8, 0.45)
    r_ankle_pitch = np.clip(-(r_hip_pitch + r_knee * 0.55), -0.8, 0.45)
    l_ankle_roll = np.clip(-l_hip_roll * 0.8, -0.25, 0.25)
    r_ankle_roll = np.clip(-r_hip_roll * 0.8, -0.25, 0.25)

    # Squat height compensation
    qpos[2] = 0.793 - (l_knee + r_knee) * 0.055

    # Assign legs
    # Left leg
    qpos[7]  = l_hip_pitch
    qpos[8]  = l_hip_roll
    qpos[9]  = 0.0          # hip yaw
    qpos[10] = l_knee
    qpos[11] = l_ankle_pitch
    qpos[12] = l_ankle_roll

    # Right leg
    qpos[13] = r_hip_pitch
    qpos[14] = r_hip_roll
    qpos[15] = 0.0          # hip yaw
    qpos[16] = r_knee
    qpos[17] = r_ankle_pitch
    qpos[18] = r_ankle_roll

    # 4. Upper Body Kinematics (Arms, Elbows, Wrists)
    # Left arm: 11(sh), 13(el), 15(wr), 19(ind)
    # Right arm: 12(sh), 14(el), 16(wr), 20(ind)

    # Upper arm vectors (shoulder -> elbow)
    l_ua_x = lm[13].x - lm[11].x
    l_ua_y = lm[13].y - lm[11].y
    l_ua_z = lm[13].z - lm[11].z

    r_ua_x = lm[14].x - lm[12].x
    r_ua_y = lm[14].y - lm[12].y
    r_ua_z = lm[14].z - lm[12].z

    # Forearm vectors (elbow -> wrist)
    l_fa_x = lm[15].x - lm[13].x
    l_fa_y = lm[15].y - lm[13].y
    l_fa_z = lm[15].z - lm[13].z

    r_fa_x = lm[16].x - lm[14].x
    r_fa_y = lm[16].y - lm[14].y
    r_fa_z = lm[16].z - lm[14].z

    # Left Shoulder Pitch: negative = forward/up, positive = back
    l_sh_pitch = np.clip((l_ua_z / (abs(l_ua_y) + 1e-4)) * 1.8, -2.5, 2.0)
    # Left Shoulder Roll: positive = abduct outward to left
    l_sh_roll  = np.clip((l_ua_x / (abs(l_ua_y) + 1e-4)) * 1.8, -0.4, 2.0)
    # Left Shoulder Yaw: inward across chest
    l_sh_yaw   = np.clip((l_fa_x / (abs(l_fa_y) + 1e-4)) * 0.8, -1.2, 1.2)

    # Right Shoulder Pitch: negative = forward/up
    r_sh_pitch = np.clip((r_ua_z / (abs(r_ua_y) + 1e-4)) * 1.8, -2.5, 2.0)
    # Right Shoulder Roll: negative = abduct outward to right
    r_sh_roll  = np.clip((r_ua_x / (abs(r_ua_y) + 1e-4)) * 1.8, -2.0, 0.4)
    # Right Shoulder Yaw: inward across chest
    r_sh_yaw   = np.clip((-r_fa_x / (abs(r_fa_y) + 1e-4)) * 0.8, -1.2, 1.2)

    # Elbow Flexion
    l_elbow = np.clip((np.pi - angle_3pts(lm[11], lm[13], lm[15])) * 1.4, 0.0, 2.0)
    r_elbow = np.clip((np.pi - angle_3pts(lm[12], lm[14], lm[16])) * 1.4, 0.0, 2.0)

    # Wrist Kinematics (using index fingers 19 and 20)
    if len(lm) > 20:
        l_hand_x = lm[19].x - lm[15].x
        l_hand_y = lm[19].y - lm[15].y
        l_hand_z = lm[19].z - lm[15].z
        l_wrist_pitch = np.clip((l_hand_z / (abs(l_hand_y) + 1e-4)) * 1.2, -1.2, 1.2)
        l_wrist_yaw   = np.clip((l_hand_x / (abs(l_hand_y) + 1e-4)) * 1.2, -1.2, 1.2)

        r_hand_x = lm[20].x - lm[16].x
        r_hand_y = lm[20].y - lm[16].y
        r_hand_z = lm[20].z - lm[16].z
        r_wrist_pitch = np.clip((r_hand_z / (abs(r_hand_y) + 1e-4)) * 1.2, -1.2, 1.2)
        r_wrist_yaw   = np.clip((-r_hand_x / (abs(r_hand_y) + 1e-4)) * 1.2, -1.2, 1.2)
    else:
        l_wrist_pitch = l_wrist_yaw = r_wrist_pitch = r_wrist_yaw = 0.0

    # Assign arms
    qpos[22] = l_sh_pitch
    qpos[23] = l_sh_roll
    qpos[24] = l_sh_yaw
    qpos[25] = l_elbow
    qpos[26] = 0.0          # wrist roll
    qpos[27] = l_wrist_pitch
    qpos[28] = l_wrist_yaw

    qpos[29] = r_sh_pitch
    qpos[30] = r_sh_roll
    qpos[31] = r_sh_yaw
    qpos[32] = r_elbow
    qpos[33] = 0.0          # wrist roll
    qpos[34] = r_wrist_pitch
    qpos[35] = r_wrist_yaw

    if prev_qpos is not None:
        qpos = alpha * qpos + (1.0 - alpha) * prev_qpos
        # Normalize quat
        norm = np.linalg.norm(qpos[3:7])
        if norm > 1e-6:
            qpos[3:7] /= norm

    return qpos

def main():
    root = r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D"
    video_src = os.path.join(root, "videos", "demo_real_human_robot.mp4")
    model_path = os.path.join(root, "models", "pose_landmarker_lite.task")
    out_eval = os.path.join(root, "videos", "eval_accurate_retarget.png")

    detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path))
    )

    mj_model = mujoco.MjModel.from_xml_path(os.path.join(root, "assets", "unitree_g1", "scene.xml"))
    mj_data = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, width=480, height=480)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = [0.0, 0.0, 0.72]
    cam.distance = 2.0
    cam.elevation = -5.0
    cam.azimuth = 180.0

    cap = cv2.VideoCapture(video_src)
    for _ in range(180):
        ret, frame = cap.read()
    ret, frame = cap.read() # frame 180
    cap.release()

    crop = frame[580:1005, 110:605]
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
    lm = res.pose_landmarks[0]

    qpos = extract_g1_qpos_accurate(lm)
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    renderer.update_scene(mj_data, camera=cam)
    mj_pix = renderer.render()
    mj_bgr = cv2.cvtColor(mj_pix, cv2.COLOR_RGB2BGR)

    # Create composite comparison
    h, w = 480, 480
    p1 = cv2.resize(crop, (w, h))
    p3 = cv2.resize(mj_bgr, (w, h))

    comp = np.hstack([p1, p3])
    cv2.putText(comp, "HUMAN DEMONSTRATOR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(comp, "ACCURATE MUJOCO G1", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imwrite(out_eval, comp)
    print(f"Saved evaluation comparison to: {out_eval}")

if __name__ == "__main__":
    main()
