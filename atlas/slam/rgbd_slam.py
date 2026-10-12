"""RGB-D SLAM, Point Cloud Registration, and Trajectory Evaluation for ATLAS HUMANOID AI 4D.
Implements ICP registration, visual odometry, pose graph management, and ATE/RPE benchmarking.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class ICPRegistration:
    """Iterative Closest Point (ICP) rigid transformation solver using SVD."""
    def __init__(self, max_iterations: int = 25, tolerance: float = 1e-4):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def find_best_rigid_transform(self, source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optimal R (3, 3) and t (3,) aligning source to target using Kabsch-Umeyama SVD."""
        centroid_src = np.mean(source, axis=0)
        centroid_tgt = np.mean(target, axis=0)

        src_centered = source - centroid_src
        tgt_centered = target - centroid_tgt

        # Cross-covariance matrix
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Reflection correction
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_tgt - R @ centroid_src
        return R.astype(np.float32), t.astype(np.float32)

    def align(
        self,
        source_pts: np.ndarray,
        target_pts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Align source point cloud to target point cloud iteratively.
        Returns:
            (R, t, fitness_rmse)
        """
        if len(source_pts) == 0 or len(target_pts) == 0:
            return np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 1e6

        # Subsample for speed
        n_pts = min(300, len(source_pts), len(target_pts))
        src = source_pts[np.random.choice(len(source_pts), n_pts, replace=False)].copy()
        tgt = target_pts[np.random.choice(len(target_pts), n_pts, replace=False)].copy()

        curr_R = np.eye(3, dtype=np.float32)
        curr_t = np.zeros(3, dtype=np.float32)
        prev_err = float("inf")

        for _ in range(self.max_iterations):
            # Find nearest neighbors (brute-force for small subsampled point set)
            dists = np.linalg.norm(src[:, None, :] - tgt[None, :, :], axis=-1)
            nn_idx = np.argmin(dists, axis=1)
            matched_tgt = tgt[nn_idx]

            # Solve step transform
            dR, dt = self.find_best_rigid_transform(src, matched_tgt)
            src = (dR @ src.T).T + dt

            curr_R = dR @ curr_R
            curr_t = dR @ curr_t + dt

            err = float(np.mean(np.linalg.norm(src - matched_tgt, axis=1)))
            if abs(prev_err - err) < self.tolerance:
                break
            prev_err = err

        return curr_R, curr_t, prev_err


class PoseGraphSLAM:
    """Maintains keyframe poses, accumulates map points, and evaluates trajectory."""
    def __init__(self):
        self.poses: List[np.ndarray] = [np.eye(4, dtype=np.float32)]  # 4x4 SE(3) poses
        self.timestamps: List[float] = [0.0]
        self.map_points: List[np.ndarray] = []

    def add_keyframe(self, relative_R: np.ndarray, relative_t: np.ndarray, timestamp: float, pts: Optional[np.ndarray] = None):
        """Append relative transformation from previous keyframe."""
        T_rel = np.eye(4, dtype=np.float32)
        T_rel[:3, :3] = relative_R
        T_rel[:3, 3] = relative_t

        prev_T = self.poses[-1]
        new_T = prev_T @ T_rel
        self.poses.append(new_T)
        self.timestamps.append(timestamp)

        if pts is not None and len(pts) > 0:
            # Transform pts to world map frame
            world_pts = (new_T[:3, :3] @ pts.T).T + new_T[:3, 3]
            self.map_points.append(world_pts[:min(500, len(world_pts))])

    def get_trajectory_positions(self) -> np.ndarray:
        """Return (N, 3) estimated trajectory positions."""
        return np.array([T[:3, 3] for T in self.poses], dtype=np.float32)


class TrajectoryEvaluator:
    """Evaluates SLAM estimated trajectory against ground truth (ATE and RPE)."""
    @staticmethod
    def compute_ate_rmse(estimated_pos: np.ndarray, gt_pos: np.ndarray) -> float:
        """Compute Absolute Trajectory Error (ATE) Root Mean Squared Error."""
        n = min(len(estimated_pos), len(gt_pos))
        if n == 0:
            return 0.0
        err = estimated_pos[:n] - gt_pos[:n]
        sq_err = np.sum(err ** 2, axis=-1)
        return float(np.sqrt(np.mean(sq_err)))

    @staticmethod
    def compute_rpe_rmse(estimated_pos: np.ndarray, gt_pos: np.ndarray, delta: int = 1) -> float:
        """Compute Relative Pose Error (RPE) translation drift per delta steps."""
        n = min(len(estimated_pos), len(gt_pos))
        if n <= delta:
            return 0.0
        est_rel = estimated_pos[delta:n] - estimated_pos[:n-delta]
        gt_rel = gt_pos[delta:n] - gt_pos[:n-delta]
        diff = est_rel - gt_rel
        return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=-1))))
