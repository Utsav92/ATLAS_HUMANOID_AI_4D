"""
Test Kinematic Retargeting of Deep Learning Landmarks to MuJoCo G1 Robot
"""
import mujoco
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def compute_angle_3pts(p1, p2, p3):
    """Compute angle at p2 between vector (p1-p2) and (p3-p2) in radians."""
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.arccos(cosine)

def retarget_landmarks_to_g1(landmarks, qpos):
    """
    Maps BlazePose 33 landmarks to G1 joint coordinates in qpos.
    qpos: array of length 36
    """
    if landmarks is None or len(landmarks) < 33:
        return

    # Joint landmarks
    ls = landmarks[11] # left shoulder
    rs = landmarks[12] # right shoulder
    le = landmarks[13] # left elbow
    re = landmarks[14] # right elbow
    lw = landmarks[15] # left wrist
    rw = landmarks[16] # right wrist
    lh = landmarks[23] # left hip
    rh = landmarks[24] # right hip
    lk = landmarks[25] # left knee
    rk = landmarks[26] # right knee
    la = landmarks[27] # left ankle
    ra = landmarks[28] # right ankle

    # 1. Base translation
    hip_center_x = (lh.x + rh.x) / 2.0 - 0.5
    qpos[0] = hip_center_x * 0.4  # lateral sway
    qpos[1] = 0.0

    # 2. Left arm
    # Elbow flexion: angle between upper arm and forearm
    l_elbow_angle = np.pi - compute_angle_3pts(ls, le, lw)
    qpos[25] = np.clip(l_elbow_angle, 0.0, 2.0)
    
    # Left shoulder pitch: angle of upper arm in YZ
    # In MediaPipe, y is down, z is depth
    l_arm_dy = le.y - ls.y
    l_arm_dz = le.z - ls.z
    l_arm_dx = le.x - ls.x
    
    # Shoulder pitch (forward/back)
    qpos[22] = np.clip(-l_arm_dz * 3.0, -1.5, 1.5)
    # Shoulder roll (abduction)
    qpos[23] = np.clip((l_arm_dx) * 2.5, -0.2, 1.8)

    # 3. Right arm
    r_elbow_angle = np.pi - compute_angle_3pts(rs, re, rw)
    qpos[32] = np.clip(r_elbow_angle, 0.0, 2.0)

    r_arm_dy = re.y - rs.y
    r_arm_dz = re.z - rs.z
    r_arm_dx = re.x - rs.x

    qpos[29] = np.clip(-r_arm_dz * 3.0, -1.5, 1.5)
    qpos[30] = np.clip((r_arm_dx) * 2.5, -1.8, 0.2)

    # 4. Legs
    l_knee_flex = np.pi - compute_angle_3pts(lh, lk, la)
    r_knee_flex = np.pi - compute_angle_3pts(rh, rk, ra)
    
    qpos[10] = np.clip(l_knee_flex * 1.2, 0.0, 1.5)  # left knee
    qpos[16] = np.clip(r_knee_flex * 1.2, 0.0, 1.5)  # right knee
    
    # Hip pitch compensates knee flexion to keep torso upright
    qpos[7] = -qpos[10] * 0.5  # left hip pitch
    qpos[13] = -qpos[16] * 0.5 # right hip pitch
    
    # Ankle pitch compensates knee flexion
    qpos[11] = -qpos[10] * 0.5 # left ankle pitch
    qpos[17] = -qpos[16] * 0.5 # right ankle pitch

    # Lateral stepping (hip roll)
    l_step_dx = lk.x - lh.x
    r_step_dx = rk.x - rh.x
    qpos[8] = np.clip(l_step_dx * 1.5, -0.4, 0.4)
    qpos[14] = np.clip(r_step_dx * 1.5, -0.4, 0.4)

    # Adjust root height based on knee flexion
    avg_knee = (qpos[10] + qpos[16]) / 2.0
    qpos[2] = 0.78 - avg_knee * 0.08

def main():
    video_path = "videos/demo_real_human_robot.mp4"
    cap = cv2.VideoCapture(video_path)
    
    base_options = python.BaseOptions(model_asset_path="models/pose_landmarker_lite.task")
    detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    )

    model = mujoco.MjModel.from_xml_path("assets/unitree_g1/scene.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=480, height=480)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = [0.0, 0.0, 0.72]
    cam.distance = 2.3
    cam.elevation = -8.0
    cam.azimuth = 180.0

    # Test at t = 6.0 seconds
    cap.set(cv2.CAP_PROP_POS_MSEC, 6000)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video frame")
        return

    woman_crop = frame[580:1005, 110:605]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(woman_crop, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_image)

    if res.pose_landmarks and len(res.pose_landmarks) > 0:
        lm = res.pose_landmarks[0]
        # Reset qpos
        data.qpos[:] = 0.0
        data.qpos[2] = 0.78
        data.qpos[3] = 1.0  # quat w
        retarget_landmarks_to_g1(lm, data.qpos)
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=cam)
        pixels = renderer.render()

        cv2.imwrite("videos/mujoco_retarget_t6.png", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
        print("Successfully rendered MuJoCo retargeted frame at t=6s!")
    else:
        print("No landmarks detected")

    cap.release()

if __name__ == "__main__":
    main()
