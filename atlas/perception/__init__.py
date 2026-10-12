"""Atlas Perception Package: RTMPose, MediaPipe, and Depth Anything V2."""

from atlas.perception.pose_estimator import (
    PoseEstimationResult,
    RTMPoseEstimator,
    MediaPipePoseEstimator,
    HybridPoseEstimator,
    COCO_17_NAMES,
    MEDIAPIPE_33_NAMES
)
from atlas.perception.depth_anything_v2 import (
    DepthEstimationResult,
    DepthAnythingV2Estimator
)

__all__ = [
    "PoseEstimationResult",
    "RTMPoseEstimator",
    "MediaPipePoseEstimator",
    "HybridPoseEstimator",
    "COCO_17_NAMES",
    "MEDIAPIPE_33_NAMES",
    "DepthEstimationResult",
    "DepthAnythingV2Estimator",
]
