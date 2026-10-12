"""
Test Deep Learning Pose Extraction on Woman's Dance Video
Using MediaPipe BlazePose deep learning architecture.
"""
import cv2
import mediapipe as mp
import numpy as np

def main():
    video_path = "videos/demo_real_human_robot.mp4"
    cap = cv2.VideoCapture(video_path)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Read frame at ~6 seconds
    cap.set(cv2.CAP_PROP_POS_MSEC, 6000)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        return
        
    # Crop the woman in the studio: [y: 580 to 1005, x: 110 to 605]
    woman_crop = frame[580:1005, 110:605]
    h, w, _ = woman_crop.shape
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(woman_crop, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    
    if results.pose_landmarks:
        print(f"Pose detected successfully with {len(results.pose_landmarks.landmark)} landmarks!")
        
        # Draw on black canvas for skeleton replication
        skeleton_img = np.zeros((h, w, 3), dtype=np.uint8)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            skeleton_img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=4, circle_radius=4),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 229, 255), thickness=2)
        )
        
        # Save both
        cv2.imwrite("videos/dl_woman_test.png", woman_crop)
        cv2.imwrite("videos/dl_skeleton_test.png", skeleton_img)
        print("Saved dl_woman_test.png and dl_skeleton_test.png")
    else:
        print("No landmarks detected")
        
    cap.release()

if __name__ == "__main__":
    main()
