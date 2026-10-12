import urllib.request
import os

url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
out_path = r"models/pose_landmarker_full.task"
os.makedirs("models", exist_ok=True)

print("Downloading MediaPipe PoseLandmarker model...")
try:
    urllib.request.urlretrieve(url, out_path)
    print(f"Downloaded model to {out_path}, size: {os.path.getsize(out_path)} bytes")
except Exception as e:
    print(f"Download failed: {e}")
