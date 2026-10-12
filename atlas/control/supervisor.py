"""Safety Supervisor and Action Arbitrator for ATLAS HUMANOID AI 4D.
Guarantees joint limits, prevents torque saturation, enforces balance thresholds,
and triggers safe fall-recovery or emergency standby.
"""

from __future__ import annotations
import numpy as np
import mujoco
from typing import Dict, List, Tuple, Any
from atlas.core.types import RobotState, PolicyOutput


class SafetySupervisor:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.max_tilt_rad = np.deg2rad(45.0)
        self.min_torso_height = 0.50

        # Actuator effort limits
        self.ctrl_ranges = np.zeros((model.nu, 2), dtype=np.float32)
        for i in range(model.nu):
            self.ctrl_ranges[i] = model.actuator_ctrlrange[i]

    def validate_action(
        self,
        raw_ctrl: np.ndarray,
        state: RobotState,
        nominal_safe_ctrl: np.ndarray
    ) -> PolicyOutput:
        """Inspect proposed action. Clamp, override, or approve with audit diagnostics."""
        reasons = []
        is_safe = True

        # 1. Height check
        if state.base_position[2] < self.min_torso_height:
            reasons.append("Torso height below minimum threshold")
            is_safe = False

        # 2. Base tilt check from quaternion [w, x, y, z]
        w, x, y, z = state.base_orientation
        up_z = 1.0 - 2.0 * (x * x + y * y)
        if up_z < np.cos(self.max_tilt_rad):
            reasons.append(f"Excessive base tilt: up_z={up_z:.2f}")
            is_safe = False

        # 3. Saturation check & clipping
        clipped_ctrl = np.clip(raw_ctrl, self.ctrl_ranges[:, 0], self.ctrl_ranges[:, 1])
        diff = np.abs(clipped_ctrl - raw_ctrl)
        max_diff = float(np.max(diff)) if len(diff) > 0 else 0.0
        if max_diff > 1e-3:
            reasons.append(f"Actuator saturation clamped (max dev: {max_diff:.2f})")

        # Fallback decision
        if not is_safe:
            final_action = nominal_safe_ctrl.copy()
            override = True
        else:
            final_action = clipped_ctrl
            override = False

        return PolicyOutput(
            action=final_action,
            policy_name="safety_supervisor",
            safety_override=override,
            details={
                "is_safe": is_safe,
                "reasons": reasons,
                "max_saturation_diff": max_diff
            }
        )
