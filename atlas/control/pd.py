"""Joint-Space Proportional-Derivative (PD) Controller for Unitree G1.
Supports feedforward nominal standing control and per-actuator gains.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Tuple


class JointPDController:
    def __init__(
        self,
        kp_legs: float = 80.0,
        kd_legs: float = 4.0,
        kp_waist: float = 60.0,
        kd_waist: float = 3.0,
        kp_arms: float = 30.0,
        kd_arms: float = 1.5,
        kp_hands: float = 15.0,
        kd_hands: float = 0.5,
        nu: int = 43
    ):
        self.nu = nu
        self.kp = np.full(nu, kp_arms, dtype=np.float32)
        self.kd = np.full(nu, kd_arms, dtype=np.float32)

        # Assign stiffer gains to legs (indices 0..11) and waist (indices 12..14)
        for i in range(min(12, nu)):
            self.kp[i] = kp_legs
            self.kd[i] = kd_legs
        for i in range(12, min(15, nu)):
            self.kp[i] = kp_waist
            self.kd[i] = kd_waist
        for i in range(22, min(29, nu)):
            self.kp[i] = kp_hands
            self.kd[i] = kd_hands
        for i in range(36, min(43, nu)):
            self.kp[i] = kp_hands
            self.kd[i] = kd_hands

    def compute_torque(
        self,
        target_qpos: np.ndarray,
        current_qpos: np.ndarray,
        current_qvel: np.ndarray,
        feedforward_torque: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute control signal: tau = Kp * (q_des - q) - Kd * q_vel + tau_ff."""
        err_pos = target_qpos - current_qpos
        tau = self.kp * err_pos - self.kd * current_qvel
        if feedforward_torque is not None:
            tau += feedforward_torque
        return tau
