"""Computer Vision Module for ATLAS HUMANOID AI 4D.
Performs color segmentation, contour detection, depth back-projection,
and 3D object pose estimation in robot/world coordinates.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2


class CameraModel:
    """Pinhole camera model with intrinsic matrix K."""
    def __init__(self, width: int = 320, height: int = 240, fov_deg: float = 60.0):
        self.width = width
        self.height = height
        self.fov_rad = np.deg2rad(fov_deg)
        self.fx = (width / 2.0) / np.tan(self.fov_rad / 2.0)
        self.fy = self.fx
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.K = np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

    def pixel_to_ray(self, u: float, v: float) -> np.ndarray:
        """Convert pixel coordinates to normalized ray in camera RDF frame."""
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        ray = np.array([x, y, 1.0], dtype=np.float32)
        return ray / np.linalg.norm(ray)

    def backproject_depth(self, depth_map: np.ndarray, stride: int = 4) -> np.ndarray:
        """Back-project depth map to 3D point cloud in camera frame (N, 3)."""
        h, w = depth_map.shape
        v_coords, u_coords = np.mgrid[0:h:stride, 0:w:stride]
        u_flat = u_coords.flatten()
        v_flat = v_coords.flatten()
        z_flat = depth_map[v_flat, u_flat]

        valid = (z_flat > 0.05) & (z_flat < 5.0)
        u_val = u_flat[valid]
        v_val = v_flat[valid]
        z_val = z_flat[valid]

        x_pts = (u_val - self.cx) * z_val / self.fx
        y_pts = (v_val - self.cy) * z_val / self.fy
        pts = np.stack([x_pts, y_pts, z_val], axis=-1)
        return pts.astype(np.float32)


class ColorObjectDetector:
    """Detects colored primitive objects (e.g. red cube) using HSV segmentation and depth."""
    def __init__(self, camera: CameraModel):
        self.camera = camera

    def detect_red_cube(
        self,
        rgb_img: np.ndarray,
        depth_map: Optional[np.ndarray] = None
    ) -> Optional[Dict[str, Any]]:
        """Detect red cube in RGB frame and estimate its 3D position if depth is provided."""
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

        # Red HSV ranges wrap around 0/180
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([168, 70, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 15:  # Minimum pixel area threshold
            return None

        x, y, w, h = cv2.boundingRect(largest)
        u_center = float(x + w / 2.0)
        v_center = float(y + h / 2.0)

        pos_3d = None
        if depth_map is not None:
            # Sample median depth in box center
            u_min, u_max = int(max(0, u_center - 2)), int(min(rgb_img.shape[1], u_center + 3))
            v_min, v_max = int(max(0, v_center - 2)), int(min(rgb_img.shape[0], v_center + 3))
            crop = depth_map[v_min:v_max, u_min:u_max]
            valid_depths = crop[crop > 0.05]
            if len(valid_depths) > 0:
                z = float(np.median(valid_depths))
                x_3d = (u_center - self.camera.cx) * z / self.camera.fx
                y_3d = (v_center - self.camera.cy) * z / self.camera.fy
                pos_3d = np.array([x_3d, y_3d, z], dtype=np.float32)

        return {
            "bbox": [x, y, w, h],
            "centroid": [u_center, v_center],
            "area": float(area),
            "camera_3d_position": pos_3d,
            "confidence": min(1.0, area / 200.0)
        }
