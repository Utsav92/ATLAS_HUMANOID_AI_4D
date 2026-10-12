import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

img = cv2.imread(r"videos/misaki_preview_10s.png")
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5) as pose:
    results = pose.process(rgb)
    if results.pose_landmarks:
        print(f"Deep learning pose detected successfully! Landmarks: {len(results.pose_landmarks.landmark)}")
        # Draw on image for verification
        annotated = img.copy()
        mp_drawing.draw_landmarks(annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        cv2.imwrite(r"videos/test_dl_pose_extracted.png", annotated)
        print("Saved annotated deep learning extraction to videos/test_dl_pose_extracted.png")
    else:
        print("No landmarks detected.")
