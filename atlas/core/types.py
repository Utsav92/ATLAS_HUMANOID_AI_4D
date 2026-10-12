"""Canonical types and data representations for ATLAS HUMANOID AI 4D.
Provides unified state schemas, observations, task representations, and logs.
"""

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class RobotState:
    timestamp: float = 0.0
    qpos: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    qvel: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    ctrl: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    base_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    base_linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    joint_positions: Dict[str, float] = field(default_factory=dict)
    joint_velocities: Dict[str, float] = field(default_factory=dict)
    feet_contacts: Dict[str, bool] = field(default_factory=lambda: {"left_foot": False, "right_foot": False})
    hand_contacts: Dict[str, bool] = field(default_factory=lambda: {"left_hand": False, "right_hand": False})
    com_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    zmp: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "base_position": self.base_position.tolist(),
            "base_orientation": self.base_orientation.tolist(),
            "base_linear_velocity": self.base_linear_velocity.tolist(),
            "base_angular_velocity": self.base_angular_velocity.tolist(),
            "joint_positions": {k: float(v) for k, v in self.joint_positions.items()},
            "joint_velocities": {k: float(v) for k, v in self.joint_velocities.items()},
            "feet_contacts": self.feet_contacts,
            "hand_contacts": self.hand_contacts,
            "com_position": self.com_position.tolist(),
            "zmp": self.zmp.tolist(),
        }


@dataclass
class MultimodalObservation:
    timestamp: float = 0.0
    robot_state: Optional[RobotState] = None
    rgb_image: Optional[np.ndarray] = None  # (H, W, 3) uint8
    depth_map: Optional[np.ndarray] = None  # (H, W) float32 in meters
    point_cloud: Optional[np.ndarray] = None  # (N, 3) float32
    tactile_features: Optional[np.ndarray] = None  # (D,) float32
    estimated_pose: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))  # [x, y, yaw]
    detected_objects: List[Dict[str, Any]] = field(default_factory=list)
    missing_modalities: List[str] = field(default_factory=list)


@dataclass
class TaskPlan:
    task_id: str = "task_0"
    instruction: str = ""
    task: str = "IDLE"  # e.g., "PICK_AND_PLACE", "LOCOMOTION", "REACH"
    target_object: str = ""
    destination: str = ""
    steps: List[str] = field(default_factory=list)
    current_step_index: int = 0
    status: str = "PENDING"  # PENDING, EXECUTING, COMPLETED, FAILED
    metadata: Dict[str, Any] = field(default_factory=dict)

    def current_step(self) -> Optional[str]:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    def advance(self) -> bool:
        self.current_step_index += 1
        if self.current_step_index >= len(self.steps):
            self.status = "COMPLETED"
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "task": self.task,
            "target_object": self.target_object,
            "destination": self.destination,
            "steps": self.steps,
            "current_step_index": self.current_step_index,
            "status": self.status,
            "metadata": self.metadata,
        }


@dataclass
class PolicyOutput:
    action: np.ndarray
    policy_name: str = "default_pd"
    value_estimate: float = 0.0
    reward: float = 0.0
    safety_override: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeLog:
    episode_id: str
    instruction: str = ""
    task_plan: Optional[TaskPlan] = None
    timestamps: List[float] = field(default_factory=list)
    states: List[RobotState] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    success: bool = False
    failure_reason: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)

    def record_step(self, t: float, state: RobotState, action: np.ndarray, reward: float):
        self.timestamps.append(t)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
