"""
ATLAS Precision Retargeting Test - Frame-by-frame validation across multiple choreography moments.
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

def extract_g1_qpos_precision(lm, prev_qpos=None, alpha=0.8):
    qpos = np.zeros(36, dtype=np.float64)
    qpos[3] = 1.0
    qpos[2] = 0.793

    if lm is None:
        if prev_qpos is not None:
            return prev_qpos.copy()
        return qpos

    # 1. Pelvis Yaw (orientation of pelvis in horizontal X-Z plane)
    dx_hip = lm[24].x - lm[23].x
    dz_hip = lm[24].z - lm[23].z
    # When dancer faces left, right hip comes forward (dz < 0), angle is negative
    pelvis_yaw = np.arctan2(dz_hip, dx_hip) * 1.4
    pelvis_yaw = np.clip(pelvis_yaw, -0.7, 0.7)

    c_y = np.cos(pelvis_yaw / 2.0)
    s_y = np.sin(pelvis_yaw / 2.0)
    qpos[3] = c_y
    qpos[6] = s_y

    # Pelvis translation
    pelvis_x = (lm[23].x + lm[24].x) / 2.0
    pelvis_y = (lm[23].y + lm[24].y) / 2.0
    pelvis_z = (lm[23].z + lm[24].z) / 2.0
    qpos[0] = np.clip(pelvis_z * 0.4, -0.15, 0.15)
    qpos[1] = np.clip(-(pelvis_x - 0.5) * 0.4, -0.2, 0.2)

    # 2. Torso / Waist Kinematics
    dx_sh = lm[12].x - lm[11].x
    dz_sh = lm[12].z - lm[11].z
    shoulder_yaw = np.arctan2(dz_sh, dx_sh) * 1.4
    waist_yaw = np.clip((shoulder_yaw - pelvis_yaw) * 0.8, -0.5, 0.5)
    qpos[19] = waist_yaw

    sh_mid_x = (lm[11].x + lm[12].x) / 2.0
    sh_mid_y = (lm[11].y + lm[12].y) / 2.0
    sh_mid_z = (lm[11].z + lm[12].z) / 2.0
    torso_lean_pitch = np.clip((sh_mid_z - pelvis_z) * 1.2, -0.3, 0.3)
    torso_lean_roll  = np.clip(-(sh_mid_x - pelvis_x) * 1.2, -0.3, 0.3)
    qpos[20] = torso_lean_roll
    qpos[21] = torso_lean_pitch

    # Rotate vector into robot local yaw frame
    cos_a = np.cos(-pelvis_yaw)
    sin_a = np.sin(-pelvis_yaw)
    def to_local(dx, dy, dz):
        # x is fwd, y is lat, z is up in robot frame
        # in mediapipe: x is screen right, y is down, z is away
        # local fwd (along dancer facing):
        fwd = -(dz * cos_a - dx * sin_a)
        lat = -(dx * cos_a + dz * sin_a)
        up  = -dy
        return fwd, lat, up

    # 3. Leg Kinematics
    # Right Leg: 24(hip), 26(knee), 28(ankle)
    r_th_dx = lm[26].x - lm[24].x
    r_th_dy = lm[26].y - lm[24].y
    r_th_dz = lm[26].z - lm[24].z
    r_fwd, r_lat, r_up = to_local(r_th_dx, r_th_dy, r_th_dz)

    # In G1: negative hip pitch is leg swinging FORWARD
    r_hip_pitch = np.clip(-np.arctan2(r_fwd, -r_up) * 1.8, -1.2, 1.2)
    # Right hip roll: negative abducts outward to right
    r_hip_roll  = np.clip(-np.arctan2(r_lat, -r_up) * 1.5, -0.8, 0.3)
    # Knee flexion
    r_knee = np.clip((np.pi - angle_3pts(lm[24], lm[26], lm[28])) * 1.4, 0.0, 2.2)
    # Ankle compensation
    r_ankle_pitch = np.clip(-(r_hip_pitch + r_knee * 0.55), -0.8, 0.45)
    r_ankle_roll  = np.clip(-r_hip_roll * 0.8, -0.25, 0.25)

    # Left Leg: 23(hip), 25(knee), 27(ankle)
    l_th_dx = lm[25].x - lm[23].x
    l_th_dy = lm[25].y - lm[23].y
    l_th_dz = lm[25].z - lm[23].z
    l_fwd, l_lat, l_up = to_local(l_th_dx, l_th_dy, l_th_dz)

    l_hip_pitch = np.clip(-np.arctan2(l_fwd, -l_up) * 1.8, -1.2, 1.2)
    # Left hip roll: positive abducts outward to left
    l_hip_roll  = np.clip(np.arctan2(l_lat, -l_up) * 1.5, -0.3, 0.8)
    l_knee = np.clip((np.pi - angle_3pts(lm[23], lm[25], lm[27])) * 1.4, 0.0, 2.2)
    l_ankle_pitch = np.clip(-(l_hip_pitch + l_knee * 0.55), -0.8, 0.45)
    l_ankle_roll  = np.clip(-l_hip_roll * 0.8, -0.25, 0.25)

    # Squat height compensation
    qpos[2] = 0.793 - (l_knee + r_knee) * 0.05

    # Assign legs
    qpos[7]  = l_hip_pitch
    qpos[8]  = l_hip_roll
    qpos[10] = l_knee
    qpos[11] = l_ankle_pitch
    qpos[12] = l_ankle_roll

    qpos[13] = r_hip_pitch
    qpos[14] = r_hip_roll
    qpos[16] = r_knee
    qpos[17] = r_ankle_pitch
    qpos[18] = r_ankle_roll

    # 4. Arm Kinematics
    # Right arm: 12(sh), 14(el), 16(wr), 20(ind)
    r_ua_dx = lm[14].x - lm[12].x
    r_ua_dy = lm[14].y - lm[12].y
    r_ua_dz = lm[14].z - lm[12].z
    r_ua_fwd, r_ua_lat, r_ua_up = to_local(r_ua_dx, r_ua_dy, r_ua_dz)

    # Forearm
    r_fa_dx = lm[16].x - lm[14].x
    r_fa_dy = lm[16].y - lm[14].y
    r_fa_dz = lm[16].z - lm[14].z
    r_fa_fwd, r_fa_lat, r_fa_up = to_local(r_fa_dx, r_fa_dy, r_fa_dz)

    # In G1: negative shoulder pitch moves arm forward/up
    r_sh_pitch = np.clip(-np.arctan2(r_ua_fwd, -r_ua_up) * 1.8, -2.5, 2.0)
    # Right shoulder roll: negative abducts arm outward to right
    r_sh_roll  = np.clip(-np.arctan2(r_ua_lat, np.sqrt(r_ua_fwd**2 + r_ua_up**2) + 1e-4) * 1.8, -2.0, 0.4)
    # Right shoulder yaw: brings forearm across chest
    r_sh_yaw   = np.clip(-np.arctan2(r_fa_lat, np.sqrt(r_fa_fwd**2 + r_fa_up**2) + 1e-4) * 1.4, -1.2, 1.2)
    # Elbow flexion
    r_elbow = np.clip((np.pi - angle_3pts(lm[12], lm[14], lm[16])) * 1.4, 0.0, 2.0)

    # Left arm: 11(sh), 13(el), 15(wr), 19(ind)
    l_ua_dx = lm[13].x - lm[11].x
    l_ua_dy = lm[13].y - lm[11].y
    l_ua_dz = lm[13].z - lm[11].z
    l_ua_fwd, l_ua_lat, l_ua_up = to_local(l_ua_dx, l_ua_dy, l_ua_dz)

    l_fa_dx = lm[15].x - lm[13].x
    l_fa_dy = lm[15].y - lm[13].y
    l_fa_dz = lm[15].z - lm[13].z
    l_fa_fwd, l_fa_lat, l_fa_up = to_local(l_fa_dx, l_fa_dy, l_fa_dz)

    l_sh_pitch = np.clip(-np.arctan2(l_ua_fwd, -l_ua_up) * 1.8, -2.5, 2.0)
    # Left shoulder roll: positive abducts arm outward to left
    l_sh_roll  = np.clip(np.arctan2(l_ua_lat, np.sqrt(l_ua_fwd**2 + l_ua_up**2) + 1e-4) * 1.8, -0.4, 2.0)
    l_sh_yaw   = np.clip(np.arctan2(l_fa_lat, np.sqrt(l_fa_fwd**2 + l_fa_up**2) + 1e-4) * 1.4, -1.2, 1.2)
    l_elbow = np.clip((np.pi - angle_3pts(lm[11], lm[13], lm[15])) * 1.4, 0.0, 2.0)

    # 5. Hand / Wrist Articulation
    if len(lm) > 20:
        r_hd_dx = lm[20].x - lm[16].x
        r_hd_dy = lm[20].y - lm[16].y
        r_hd_dz = lm[20].z - lm[16].z
        r_hd_fwd, r_hd_lat, r_hd_up = to_local(r_hd_dx, r_hd_dy, r_hd_dz)
        r_wrist_pitch = np.clip(np.arctan2(r_hd_fwd, -r_hd_up) * 1.2, -1.2, 1.2)
        r_wrist_yaw   = np.clip(-np.arctan2(r_hd_lat, abs(r_hd_up) + 1e-4) * 1.2, -1.2, 1.2)

        l_hd_dx = lm[19].x - lm[15].x
        l_hd_dy = lm[19].y - lm[15].y
        l_hd_dz = lm[19].z - lm[15].z
        l_hd_fwd, l_hd_lat, l_hd_up = to_local(l_hd_dx, l_hd_dy, l_hd_dz)
        l_wrist_pitch = np.clip(np.arctan2(l_hd_fwd, -l_hd_up) * 1.2, -1.2, 1.2)
        l_wrist_yaw   = np.clip(np.arctan2(l_hd_lat, abs(l_hd_up) + 1e-4) * 1.2, -1.2, 1.2)
    else:
        r_wrist_pitch = r_wrist_yaw = l_wrist_pitch = l_wrist_yaw = 0.0

    # Assign arms
    qpos[22] = l_sh_pitch
    qpos[23] = l_sh_roll
    qpos[24] = l_sh_yaw
    qpos[25] = l_elbow
    qpos[27] = l_wrist_pitch
    qpos[28] = l_wrist_yaw

    qpos[29] = r_sh_pitch
    qpos[30] = r_sh_roll
    qpos[31] = r_sh_yaw
    qpos[32] = r_elbow
    qpos[34] = r_wrist_pitch
    qpos[35] = r_wrist_yaw

    if prev_qpos is not None:
        qpos = alpha * qpos + (1.0 - alpha) * prev_qpos
        norm = np.linalg.norm(qpos[3:7])
        if norm > 1e-6:
            qpos[3:7] /= norm

    return qpos

def main():
    root = r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D"
    video_src = os.path.join(root, "videos", "demo_real_human_robot.mp4")
    model_path = os.path.join(root, "models", "pose_landmarker_lite.task")

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
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()

    for target_idx in [50, 180, 240, 320]:
        crop = frames[target_idx][580:1005, 110:605]
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        lm = res.pose_landmarks[0]

        qpos = extract_g1_qpos_precision(lm)
        mj_data.qpos[:] = qpos
        mj_data.qvel[:] = 0.0
        mujoco.mj_forward(mj_model, mj_data)

        renderer.update_scene(mj_data, camera=cam)
        mj_pix = renderer.render()
        mj_bgr = cv2.cvtColor(mj_pix, cv2.COLOR_RGB2BGR)

        h, w = 480, 480
        p1 = cv2.resize(crop, (w, h))
        p3 = cv2.resize(mj_bgr, (w, h))
        comp = np.hstack([p1, p3])
        cv2.putText(comp, f"FRAME {target_idx} : DEMO", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(comp, f"FRAME {target_idx} : PRECISION G1", (w + 20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        out_path = os.path.join(root, "videos", f"precision_eval_f{target_idx}.png")
        cv2.imwrite(out_path, comp)
        print(f"Saved evaluation frame to: {out_path}")

if __name__ == "__main__":
    main()
