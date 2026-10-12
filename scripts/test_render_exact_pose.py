import mujoco
import numpy as np
import cv2

# Load model
model = mujoco.MjModel.from_xml_path("assets/unitree_g1/scene.xml")
data = mujoco.MjData(model)

# Set up renderer
renderer = mujoco.Renderer(model, height=592, width=640)

# Load extracted 3D landmarks
npz_data = np.load("data/salsa_dl_mocap_3d.npz")
lms = npz_data["landmarks_3d_meters"] # (900, 33, 3)

# Test at frame 450 (t=15s)
lm = lms[450]

# Setup camera
camera = mujoco.MjvCamera()
camera.lookat = [0.0, 0.0, 0.8] # look at pelvis/chest
camera.distance = 2.5
camera.elevation = -12
camera.azimuth = 90 # facing robot directly from front

# Set default standing pose
data.qpos[0] = 0.0 # x
data.qpos[1] = 0.0 # y
data.qpos[2] = 0.78 # z height

# Update scene and render default
mujoco.mj_forward(model, data)
renderer.update_scene(data, camera=camera)
img = renderer.render()

cv2.imwrite("videos/test_mujoco_default_render.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("Rendered default MuJoCo robot frame to videos/test_mujoco_default_render.png")
