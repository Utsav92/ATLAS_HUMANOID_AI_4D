import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

model_path = r"models/pose_landmarker_full.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.PoseLandmarker.create_from_options(options)

image = mp.Image.create_from_file(r"videos/misaki_preview_10s.png")
detection_result = detector.detect(image)

if detection_result.pose_landmarks:
    print(f"Extracted {len(detection_result.pose_landmarks[0])} 3D landmarks!")
    for idx, lm in enumerate(detection_result.pose_landmarks[0][:5]):
        print(f"  Landmark {idx}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}, visibility={lm.visibility:.3f}")
    if detection_result.pose_world_landmarks:
        print(f"World landmarks (metric 3D): {len(detection_result.pose_world_landmarks[0])}")
        for idx, wlm in enumerate(detection_result.pose_world_landmarks[0][:5]):
            print(f"  World 3D {idx}: x={wlm.x:.3f}m, y={wlm.y:.3f}m, z={wlm.z:.3f}m")
else:
    print("No landmarks detected.")
