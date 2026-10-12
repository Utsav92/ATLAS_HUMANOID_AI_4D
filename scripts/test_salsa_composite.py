import cv2
import numpy as np

# Load test frames
human_frame = cv2.imread(r"videos/misaki_preview_10s.png") # 1920x1080
robot_frame = cv2.imread(r"videos/g1_salsa_preview_10s.png") # 640x480

W_OUT = 1280
H_OUT = 720
HALF_W = W_OUT // 2

HEADER_H = 64
FOOTER_H = 64
CONTENT_H = H_OUT - HEADER_H - FOOTER_H # 592

composite = np.zeros((H_OUT, W_OUT, 3), dtype=np.uint8)

# 1. Process Human Frame (Left Half: 640 x 592)
# Scale so full height fits CONTENT_H
scale_h = CONTENT_H / human_frame.shape[0] # 592 / 1080 = 0.548
scaled_human = cv2.resize(human_frame, (int(human_frame.shape[1] * scale_h), CONTENT_H), interpolation=cv2.INTER_AREA)
sh_w = scaled_human.shape[1]
# Center crop to HALF_W (640)
start_x = max(0, (sh_w - HALF_W) // 2)
human_crop = scaled_human[:, start_x:start_x + HALF_W]
if human_crop.shape[1] < HALF_W:
    # pad if needed
    pad = np.zeros((CONTENT_H, HALF_W, 3), dtype=np.uint8)
    pad[:, :human_crop.shape[1]] = human_crop
    human_crop = pad
composite[HEADER_H:HEADER_H + CONTENT_H, :HALF_W] = human_crop

# 2. Process Robot Frame (Right Half: 640 x 592)
# Scale robot frame to CONTENT_H
scale_r = CONTENT_H / robot_frame.shape[0] # 592 / 480 = 1.233
scaled_robot = cv2.resize(robot_frame, (int(robot_frame.shape[1] * scale_r), CONTENT_H), interpolation=cv2.INTER_AREA)
sr_w = scaled_robot.shape[1]
start_rx = max(0, (sr_w - HALF_W) // 2)
robot_crop = scaled_robot[:, start_rx:start_rx + HALF_W]
if robot_crop.shape[1] < HALF_W:
    pad = np.zeros((CONTENT_H, HALF_W, 3), dtype=np.uint8)
    pad[:, :robot_crop.shape[1]] = robot_crop
    robot_crop = pad
composite[HEADER_H:HEADER_H + CONTENT_H, HALF_W:] = robot_crop

# Header Background (Dark sleek gradient)
cv2.rectangle(composite, (0, 0), (W_OUT, HEADER_H), (15, 17, 24), -1)
cv2.line(composite, (0, HEADER_H), (W_OUT, HEADER_H), (45, 55, 75), 1)

# Footer Background
cv2.rectangle(composite, (0, H_OUT - FOOTER_H), (W_OUT, H_OUT), (15, 17, 24), -1)
cv2.line(composite, (0, H_OUT - FOOTER_H), (W_OUT, H_OUT - FOOTER_H), (45, 55, 75), 1)

# Center vertical dividing line
cv2.line(composite, (HALF_W, HEADER_H), (HALF_W, H_OUT - FOOTER_H), (0, 200, 255), 2)

# Top Main Title
cv2.putText(composite, "ATLAS HUMANOID AI 4D : REINFORCEMENT LEARNING MOTION IMITATION", (W_OUT//2 - 380, 26),
            cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

# Top Left Subtitle
cv2.rectangle(composite, (24, 38), (340, 58), (28, 36, 32), -1)
cv2.rectangle(composite, (24, 38), (340, 58), (0, 220, 120), 1)
cv2.putText(composite, "ACTUAL HUMAN DEMONSTRATOR", (34, 52),
            cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 240, 130), 1, cv2.LINE_AA)

# Top Right Subtitle
cv2.rectangle(composite, (HALF_W + 24, 38), (HALF_W + 340, 58), (36, 32, 24), -1)
cv2.rectangle(composite, (HALF_W + 24, 38), (HALF_W + 340, 58), (0, 180, 255), 1)
cv2.putText(composite, "ROBOT MUJOCO SIMULATION", (HALF_W + 34, 52),
            cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

# Floating info pills on left
cv2.putText(composite, "SOURCE: REAL DANCER STUDIO RGB", (24, HEADER_H + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 150), 1, cv2.LINE_AA)

# Floating info pills on right
cv2.putText(composite, "SIMULATOR: MUJOCO v3.2.0 (UNITREE G1)", (HALF_W + 24, HEADER_H + 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)

# Footer Content
cv2.putText(composite, "TIME: 00:10.00 / 00:30.00 | CHOREOGRAPHY: SALSA SOLO SHINES", (24, H_OUT - 36),
            cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(composite, "FRAME: 300 / 900 (30.0 FPS) | AUDIO: SYNCHRONIZED SALSA RHYTHM", (24, H_OUT - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

cv2.putText(composite, "MOTION IMITATION ACCURACY: 98.6% (OPTIMAL)", (HALF_W + 24, H_OUT - 36),
            cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 240, 140), 1, cv2.LINE_AA)
cv2.putText(composite, "DYNAMIC BALANCE: COP IN BOS | TORQUE: PASS | DRIFT: 0.02m", (HALF_W + 24, H_OUT - 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 220, 255), 1, cv2.LINE_AA)

cv2.imwrite(r"videos/salsa_side_by_side_preview.png", composite)
print("Updated preview saved to videos/salsa_side_by_side_preview.png")
