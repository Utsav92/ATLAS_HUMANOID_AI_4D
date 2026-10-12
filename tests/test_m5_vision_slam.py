"""Unit tests for Milestone M5: RGB-D Perception and SLAM."""

import pytest
import numpy as np
import cv2

from atlas.vision.detector import CameraModel, ColorObjectDetector
from atlas.slam.rgbd_slam import ICPRegistration, PoseGraphSLAM, TrajectoryEvaluator


def test_m5_camera_and_depth_projection():
    cam = CameraModel(width=320, height=240, fov_deg=60.0)
    assert cam.K.shape == (3, 3)

    # Synthetic depth map
    depth = np.ones((240, 320), dtype=np.float32) * 1.5  # 1.5 meters away
    pts = cam.backproject_depth(depth, stride=8)
    assert pts.ndim == 2
    assert pts.shape[1] == 3
    # Z coordinate should equal depth
    np.testing.assert_allclose(pts[:, 2], 1.5, rtol=1e-3)


def test_m5_color_object_detector():
    cam = CameraModel(width=320, height=240)
    detector = ColorObjectDetector(cam)

    # Create synthetic black image with red square in the center
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    # Red in RGB is [255, 0, 0]
    cv2.rectangle(rgb, (140, 100), (180, 140), (255, 20, 20), -1)
    depth = np.ones((240, 320), dtype=np.float32) * 1.0

    res = detector.detect_red_cube(rgb, depth)
    assert res is not None
    assert abs(res["centroid"][0] - 160.0) < 5.0
    assert abs(res["centroid"][1] - 120.0) < 5.0
    assert res["camera_3d_position"] is not None
    np.testing.assert_allclose(res["camera_3d_position"][2], 1.0, atol=0.05)


def test_m5_icp_registration():
    np.random.seed(42)
    # Create synthetic source point cloud
    src_pts = np.random.randn(100, 3).astype(np.float32) * 0.5

    # Known translation
    true_t = np.array([0.05, -0.03, 0.02], dtype=np.float32)
    tgt_pts = src_pts + true_t

    icp = ICPRegistration(max_iterations=15)
    R, t, err = icp.align(src_pts, tgt_pts)
    assert err < 0.05
    np.testing.assert_allclose(t, true_t, atol=0.03)


def test_m5_trajectory_evaluation():
    gt = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32)
    est = gt + np.array([0.02, 0.01, -0.01], dtype=np.float32)  # slight offset

    ate = TrajectoryEvaluator.compute_ate_rmse(est, gt)
    rpe = TrajectoryEvaluator.compute_rpe_rmse(est, gt)

    assert ate < 0.05
    assert rpe < 0.01
