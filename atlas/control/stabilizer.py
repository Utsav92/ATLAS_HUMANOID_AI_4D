"""Humanoid Posture Stabilizer for Unitree G1.
Maintains balance by adjusting ankle and hip joints using CoM / ZMP feedback.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from atlas.core.types import RobotState


class PostureStabilizer:
    def __init__(
        self,
        target_com_height: float = 0.69,
        k_com_x: float = 0.15,
        k_com_y: float = 0.20,
        k_vel_x: float = 0.05,
        k_vel_y: float = 0.08
    ):
        self.target_com_height = target_com_height
        self.k_com_x = k_com_x
        self.k_com_y = k_com_y
        self.k_vel_x = k_vel_x
        self.k_vel_y = k_vel_y

        # Indices in actuator vector for ankle corrections:
        # 4: left_ankle_pitch, 5: left_ankle_roll
        # 10: right_ankle_pitch, 11: right_ankle_roll
        self.left_ankle_pitch_idx = 4
        self.left_ankle_roll_idx = 5
        self.right_ankle_pitch_idx = 10
        self.right_ankle_roll_idx = 11

    def compute_stabilization_offset(self, state: RobotState) -> np.ndarray:
        """Compute joint angle offsets to restore center of mass balance."""
        nu = len(state.ctrl)
        delta_q = np.zeros(nu, dtype=np.float32)

        # Deviation of CoM from base position in horizontal plane
        com_err_x = state.com_position[0] - state.base_position[0]
        com_err_y = state.com_position[1] - state.base_position[1]

        # Velocity feedback
        vel_x = state.base_linear_velocity[0]
        vel_y = state.base_linear_velocity[1]

        # Corrective ankle pitch (sagittal) and roll (coronal)
        pitch_correction = -(self.k_com_x * com_err_x + self.k_vel_x * vel_x)
        roll_correction = -(self.k_com_y * com_err_y + self.k_vel_y * vel_y)

        # Apply symmetric/differential offsets
        if self.left_ankle_pitch_idx < nu:
            delta_q[self.left_ankle_pitch_idx] = pitch_correction
        if self.right_ankle_pitch_idx < nu:
            delta_q[self.right_ankle_pitch_idx] = pitch_correction

        if self.left_ankle_roll_idx < nu:
            delta_q[self.left_ankle_roll_idx] = roll_correction
        if self.right_ankle_roll_idx < nu:
            delta_q[self.right_ankle_roll_idx] = roll_correction

        return delta_q
