"""Canonical schemas for ATLAS HUMANOID AI 4D Dataset Hub.
Enforces typed metadata, coordinate frames, embodiment constraints, and train/val/test splits.
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class DatasetManifest:
    name: str
    source_id: str
    version: str
    license_spdx: str
    robot_embodiment: str  # e.g., "Unitree G1", "Generic Biped", "Franka Panda", "Sensor Rig"
    num_episodes: int
    num_samples: int
    features: List[str]
    split_ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    coordinate_frame: str = "FLU"  # Forward-Left-Up (MuJoCo/ROS standard)
    citation: str = ""
    url: str = ""
    github_url: str = ""
    hf_url: str = ""
    purpose: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectorySample:
    episode_id: str
    step_id: int
    timestamp: float
    joint_names: List[str]
    joint_positions: np.ndarray  # (D,)
    joint_velocities: np.ndarray  # (D,)
    actions: np.ndarray  # (A,)
    base_pose: np.ndarray  # (7,) [x, y, z, qw, qx, qy, qz]
    reward: float = 0.0
    done: bool = False


@dataclass
class SLAMSample:
    sequence_id: str
    frame_id: int
    timestamp: float
    rgb_shape: Tuple[int, int, int]
    depth_shape: Tuple[int, int]
    ground_truth_pose: np.ndarray  # (7,) [tx, ty, tz, qx, qy, qz, qw]
    estimated_pose: Optional[np.ndarray] = None


@dataclass
class TactileSample:
    sample_id: str
    timestamp: float
    sensor_type: str  # "GelSight", "YCB-Sight", "Sparsh"
    taxel_shape: Tuple[int, ...]
    contact_force: float
    slip_detected: bool
    contact_state: int  # 0: no contact, 1: stable contact, 2: slipping, 3: lift


@dataclass
class LanguageActionSample:
    sample_id: str
    instruction: str
    intent: str
    target_object: str
    destination: str
    subgoals: List[str]
    initial_world_state: Dict[str, Any]
    target_world_state: Dict[str, Any]
    success: bool = True
