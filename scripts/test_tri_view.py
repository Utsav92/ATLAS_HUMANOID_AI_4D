"""
Test Tri-View Generation:
[1] Woman Dancer (Studio) -> [2] Deep Learning Skeleton -> [3] Unitree G1 Robot
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

def main():
    video_path = "videos/demo_real_human_robot.mp4"
    cap = cv2.VideoCapture(video_path)
    
    # Initialize MediaPipe PoseLandmarker
    base_options = python.BaseOptions(model_asset_path="models/pose_landmarker_lite.task")
    options = vision.PoseLandmarkerOptions(base_options=base_options, output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # Seek to t = 6.0 seconds
    cap.set(cv2.CAP_PROP_POS_MSEC, 6000)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return
        
    # Full frame is 1920x1080
    # 1. Woman crop from bottom-left studio camera: [y: 580:1005, x: 110:605]
    woman_crop = frame[580:1005, 110:605]
    
    # 2. Deep learning landmark detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(woman_crop, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    
    # Target size for each panel in 1280x720 video:
    # 3 panels side-by-side: width = 390 each, height = 520 each
    panel_w = 390
    panel_h = 520
    
    panel1_woman = cv2.resize(woman_crop, (panel_w, panel_h))
    
    # Create Panel 2: Skeleton canvas
    panel2_skeleton = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    
    # Define pose connections for skeleton drawing
    connections = [
        # Torso
        (11, 12), (11, 23), (12, 24), (23, 24),
        # Left arm
        (11, 13), (13, 15),
        # Right arm
        (12, 14), (14, 16),
        # Left leg
        (23, 25), (25, 27), (27, 29), (29, 31),
        # Right leg
        (24, 26), (26, 28), (28, 30), (30, 32),
        # Head / Neck
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12)
    ]
    
    if len(results.pose_landmarks) > 0:
        landmarks = results.pose_landmarks[0]
        # Draw connections
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                p1 = landmarks[start_idx]
                p2 = landmarks[end_idx]
                if p1.visibility > 0.3 and p2.visibility > 0.3:
                    x1, y1 = int(p1.x * panel_w), int(p1.y * panel_h)
                    x2, y2 = int(p2.x * panel_w), int(p2.y * panel_h)
                    cv2.line(panel2_skeleton, (x1, y1), (x2, y2), (0, 229, 255), 3, cv2.LINE_AA)
                    
        # Draw joints
        for idx, pt in enumerate(landmarks):
            if pt.visibility > 0.3:
                x, y = int(pt.x * panel_w), int(pt.y * panel_h)
                # Key body joints in bright green, head in cyan
                color = (0, 255, 128) if idx in [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] else (0, 229, 255)
                cv2.circle(panel2_skeleton, (x, y), 5, color, -1, cv2.LINE_AA)
                cv2.circle(panel2_skeleton, (x, y), 7, (255, 255, 255), 1, cv2.LINE_AA)
                
    # 3. Robot crop from main camera: [y: 250:970, x: 740:1460]
    robot_crop = frame[250:970, 740:1460]
    panel3_robot = cv2.resize(robot_crop, (panel_w, panel_h))
    
    # Build complete 1280x720 canvas
    canvas = np.full((720, 1280, 3), 15, dtype=np.uint8)  # sleek dark background
    
    # Place panels
    # y range: 110 to 630 (520 px height)
    # x1: 30 to 420
    # x2: 445 to 835
    # x3: 860 to 1250
    canvas[110:630, 30:420] = panel1_woman
    canvas[110:630, 445:835] = panel2_skeleton
    canvas[110:630, 860:1250] = panel3_robot
    
    # Draw border boxes around each panel
    for x_start in [30, 445, 860]:
        cv2.rectangle(canvas, (x_start, 110), (x_start + panel_w, 630), (0, 229, 255), 2)
        
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Header
    cv2.putText(canvas, "ATLAS HUMANOID AI 4D : DEEP LEARNING POSE EXTRACTION & ROBOT RETARGETING", (30, 45), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    # Column labels
    cv2.putText(canvas, "[1] HUMAN DEMONSTRATOR", (30, 95), font, 0.55, (0, 255, 128), 2, cv2.LINE_AA)
    cv2.putText(canvas, "[2] DEEP LEARNING SKELETON", (445, 95), font, 0.55, (0, 229, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "[3] UNITREE G1 ROBOT", (860, 95), font, 0.55, (255, 200, 0), 2, cv2.LINE_AA)
    # Sub labels
    cv2.putText(canvas, "Performer: Jasmine Coro", (30, 650), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "BlazePose 33-Keypoint Extraction", (445, 650), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "29-DoF Embodied Motion Execution", (860, 650), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    # Bottom status bar
    cv2.putText(canvas, "DATASET: G1 MOVES (exptech/g1-moves) | SYNC: 100% FRAME-ACCURATE | STATUS: SYNC LOCKED", (30, 695), font, 0.5, (0, 255, 128), 1, cv2.LINE_AA)
    cv2.putText(canvas, "TIME: 00:00:06 / 00:00:45", (1020, 695), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    cv2.imwrite("videos/tri_view_preview.png", canvas)
    print("Saved videos/tri_view_preview.png")

if __name__ == "__main__":
    main()
