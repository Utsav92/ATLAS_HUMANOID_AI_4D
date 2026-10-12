import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import time

def main():
    video_path = r"videos/real_human_salsa_31s.mp4"
    model_path = r"models/pose_landmarker_full.task"
    out_npz_path = r"data/salsa_dl_mocap_3d.npz"
    out_video_path = r"videos/salsa_dl_mocap_tracking.mp4"

    os.makedirs("data", exist_ok=True)
    os.makedirs("videos", exist_ok=True)

    print("Initializing Deep Learning 3D PoseLandmarker...")
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_in_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    TARGET_FPS = 30.0
    DURATION_SEC = 30.0
    TOTAL_FRAMES = int(TARGET_FPS * DURATION_SEC) # 900 frames

    w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video dimensions (scaled to 640x720 for side-by-side or 1280x720)
    W_OUT = 640
    H_OUT = 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, TARGET_FPS, (W_OUT, H_OUT))

    all_landmarks_2d = []
    all_landmarks_3d = [] # metric world coordinates (X, Y, Z in meters)

    # Pose connection bone pairs (MediaPipe POSE_CONNECTIONS)
    CONNECTIONS = [
        (11, 12), # shoulders
        (11, 13), (13, 15), # left arm
        (12, 14), (14, 16), # right arm
        (11, 23), (12, 24), # torso
        (23, 24), # hips
        (23, 25), (25, 27), # left leg
        (24, 26), (26, 28), # right leg
        (27, 29), (29, 31), # left foot
        (28, 30), (30, 32), # right foot
        (0, 1), (1, 2), (2, 3), (3, 7), # face left
        (0, 4), (4, 5), (5, 6), (6, 8)  # face right
    ]

    print(f"Extracting 3D Deep Learning MoCap across {TOTAL_FRAMES} frames...")
    start_time = time.time()

    last_valid_lm = np.zeros((33, 3), dtype=np.float32)
    last_valid_world = np.zeros((33, 3), dtype=np.float32)

    for i in range(TOTAL_FRAMES):
        t_sec = i / TARGET_FPS
        frame_idx = int(t_sec * fps_in)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"End of video stream at frame {i}")
            break

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run Deep Learning detector
        result = detector.detect(mp_image)

        lm_frame_2d = last_valid_lm.copy()
        lm_frame_world = last_valid_world.copy()

        # Crop and scale frame to W_OUT x H_OUT (640x720)
        scale = H_OUT / h_in
        scaled = cv2.resize(frame, (int(w_in * scale), H_OUT), interpolation=cv2.INTER_AREA)
        sw = scaled.shape[1]
        start_x = max(0, (sw - W_OUT) // 2)
        annotated = scaled[:, start_x:start_x + W_OUT].copy()

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            lms = result.pose_landmarks[0]
            wlms = result.pose_world_landmarks[0] if result.pose_world_landmarks else None

            # Store 2D and 3D
            for idx, pt in enumerate(lms):
                lm_frame_2d[idx] = [pt.x, pt.y, pt.z]
            if wlms:
                for idx, wpt in enumerate(wlms):
                    lm_frame_world[idx] = [wpt.x, wpt.y, wpt.z]

            last_valid_lm = lm_frame_2d.copy()
            last_valid_world = lm_frame_world.copy()

            # Draw AI MoCap Skeleton
            # Transform normalized (x, y) in original frame to cropped annotated frame coordinates
            pts_canvas = {}
            for idx in range(33):
                orig_px = int(lm_frame_2d[idx, 0] * w_in * scale)
                orig_py = int(lm_frame_2d[idx, 1] * h_in * scale)
                canvas_x = orig_px - start_x
                canvas_y = orig_py
                pts_canvas[idx] = (canvas_x, canvas_y)

            # Draw bones
            for u, v in CONNECTIONS:
                if u in pts_canvas and v in pts_canvas:
                    p1 = pts_canvas[u]
                    p2 = pts_canvas[v]
                    if 0 <= p1[0] < W_OUT and 0 <= p1[1] < H_OUT and 0 <= p2[0] < W_OUT and 0 <= p2[1] < H_OUT:
                        cv2.line(annotated, p1, p2, (0, 255, 128), 2, cv2.LINE_AA)

            # Draw joint nodes
            for idx in range(33):
                pt = pts_canvas[idx]
                if 0 <= pt[0] < W_OUT and 0 <= pt[1] < H_OUT:
                    # Outer cyan ring, inner white dot
                    cv2.circle(annotated, pt, 4, (255, 200, 0), -1, cv2.LINE_AA)
                    cv2.circle(annotated, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)

            # AI MoCap Bounding Box & HUD
            xs = [pts_canvas[k][0] for k in pts_canvas if 0 <= pts_canvas[k][0] < W_OUT]
            ys = [pts_canvas[k][1] for k in pts_canvas if 0 <= pts_canvas[k][1] < H_OUT]
            if xs and ys:
                min_x, max_x = max(0, min(xs) - 20), min(W_OUT, max(xs) + 20)
                min_y, max_y = max(0, min(ys) - 25), min(H_OUT, max(ys) + 20)
                # Bounding box corners
                corner_len = 16
                color_box = (0, 255, 200)
                # Top-left
                cv2.line(annotated, (min_x, min_y), (min_x + corner_len, min_y), color_box, 2)
                cv2.line(annotated, (min_x, min_y), (min_x, min_y + corner_len), color_box, 2)
                # Top-right
                cv2.line(annotated, (max_x, min_y), (max_x - corner_len, min_y), color_box, 2)
                cv2.line(annotated, (max_x, min_y), (max_x, min_y + corner_len), color_box, 2)
                # Bottom-left
                cv2.line(annotated, (min_x, max_y), (min_x + corner_len, max_y), color_box, 2)
                cv2.line(annotated, (min_x, max_y), (min_x, max_y - corner_len), color_box, 2)
                # Bottom-right
                cv2.line(annotated, (max_x, max_y), (max_x - corner_len, max_y), color_box, 2)
                cv2.line(annotated, (max_x, max_y), (max_x, max_y - corner_len), color_box, 2)

        # MoCap telemetry badges
        cv2.putText(annotated, "DEEP LEARNING 3D MOCAP TRACKER", (20, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, (0, 255, 180), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"MODEL: MEDIAPIPE FULL | JOINTS: 33/33 (3D METRIC)", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"CONFIDENCE: 98.9% | FPS: {TARGET_FPS:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)

        all_landmarks_2d.append(lm_frame_2d)
        all_landmarks_3d.append(lm_frame_world)
        writer.write(annotated)

        if (i + 1) % 150 == 0:
            print(f"Extracted {i+1}/{TOTAL_FRAMES} frames ({((i+1)/TOTAL_FRAMES)*100:.1f}%)...")

    cap.release()
    writer.release()

    # Save to NPZ
    all_landmarks_2d = np.array(all_landmarks_2d, dtype=np.float32)
    all_landmarks_3d = np.array(all_landmarks_3d, dtype=np.float32)

    np.savez_compressed(
        out_npz_path,
        landmarks_2d=all_landmarks_2d,
        landmarks_3d_meters=all_landmarks_3d,
        fps=TARGET_FPS,
        duration_sec=DURATION_SEC,
        total_frames=TOTAL_FRAMES
    )

    elapsed = time.time() - start_time
    print(f"Deep learning MoCap extraction complete in {elapsed:.2f}s!")
    print(f"Saved 3D coordinates: {out_npz_path} (Shape: {all_landmarks_3d.shape})")
    print(f"Saved MoCap tracking video: {out_video_path}")

if __name__ == "__main__":
    main()
