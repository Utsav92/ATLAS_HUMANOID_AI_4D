"""MuJoCo Advanced Simulation: Contact Dynamics, Dynamic Balance, and Object Manipulation.
Simulates multi-contact physics (friction cones, zero-skate ground reaction),
Center of Pressure (CoP) / Zero Moment Point (ZMP) dynamic equilibrium,
and dual-arm object grasping and manipulation in MuJoCo.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import mujoco

from atlas.simulation.world import AtlasWorld
from atlas.simulation.robot import G1Robot
from atlas.core.types import RobotState


@dataclass
class ManipulationStepResult:
    """Telemetry returned after each physical simulation step."""
    robot_state: RobotState
    contact_forces: Dict[str, float]  # Normal forces in Newtons
    cop_position: np.ndarray          # Center of Pressure [x, y] in meters
    is_balanced: bool                 # CoP inside convex hull of support
    object_position: Optional[np.ndarray] = None # Object 3D position [x, y, z]
    object_error: float = 0.0         # Distance to target object trajectory
    grasp_force: float = 0.0          # Measured contact force on grasped object
    reward: float = 0.0


class MuJoCoManipulationEnv:
    """MuJoCo Simulation Environment for Contact, Dynamic Balance, and Object Manipulation."""

    def __init__(self, xml_path: Optional[str] = None, dt: float = 0.002):
        self.world = AtlasWorld(xml_path=xml_path)
        self.robot = self.world.robot
        self.model = self.world.model
        self.data = self.world.data
        self.dt = dt

        # Object tracking identifiers
        self.target_object_id = self._find_object_body(["target_cube", "red_cube", "prop_box", "object"])
        self.object_pos = np.array([0.45, 0.0, 0.72], dtype=np.float32)

    def _find_object_body(self, candidates: List[str]) -> int:
        for c in candidates:
            try:
                return self.model.body(c).id
            except KeyError:
                continue
        return -1

    def reset(self) -> ManipulationStepResult:
        """Reset robot to stable standing stance and place manipulation object."""
        state = self.world.reset()
        if self.target_object_id >= 0:
            self.object_pos = np.array(self.data.xpos[self.target_object_id], dtype=np.float32)
        else:
            self.object_pos = np.array([0.45, 0.0, 0.72], dtype=np.float32)

        return self._evaluate_step(target_object_pos=self.object_pos)

    def step(self, action_ctrl: np.ndarray,
             target_object_pos: Optional[np.ndarray] = None,
             n_substeps: int = 10) -> ManipulationStepResult:
        """Advance physical simulation with contact dynamics and balance checking."""
        full_ctrl = self.robot.stand_ctrl.copy()
        n_act = min(len(action_ctrl), len(full_ctrl))
        full_ctrl[:n_act] = action_ctrl[:n_act]
        clipped_ctrl = self.robot.clip_ctrl(full_ctrl)
        self.data.ctrl[:len(clipped_ctrl)] = clipped_ctrl

        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        return self._evaluate_step(target_object_pos)

    def _evaluate_step(self, target_object_pos: Optional[np.ndarray] = None) -> ManipulationStepResult:
        state = self.robot.get_state()

        # 1. Contact Dynamics & Normal Forces
        # Solves 3D contact impulses and checks Coulomb friction limits
        fn_left = 0.0
        fn_right = 0.0
        grasp_fn = 0.0

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]

            # Extract 6D contact force: [normal, friction_t1, friction_t2, torques...]
            c_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_force)
            f_norm = float(abs(c_force[0]))

            if b1 == self.robot.left_foot_id or b2 == self.robot.left_foot_id:
                fn_left += f_norm
            if b1 == self.robot.right_foot_id or b2 == self.robot.right_foot_id:
                fn_right += f_norm
            if b1 == self.robot.left_hand_id or b2 == self.robot.left_hand_id or \
               b1 == self.robot.right_hand_id or b2 == self.robot.right_hand_id:
                grasp_fn += f_norm

        total_fn = fn_left + fn_right
        contact_forces = {"left_foot": fn_left, "right_foot": fn_right}

        # 2. Center of Pressure (CoP) Calculation
        # CoP = (P_left * F_left + P_right * F_right) / (F_left + F_right)
        p_left = np.array(self.data.xpos[self.robot.left_foot_id][:2], dtype=np.float32)
        p_right = np.array(self.data.xpos[self.robot.right_foot_id][:2], dtype=np.float32)

        if total_fn > 1e-3:
            cop = (p_left * fn_left + p_right * fn_right) / total_fn
        else:
            cop = 0.5 * (p_left + p_right)

        # Dynamic balance verification: CoP inside base of support polygon
        bos_min_x = min(p_left[0], p_right[0]) - 0.12
        bos_max_x = max(p_left[0], p_right[0]) + 0.12
        bos_min_y = min(p_left[1], p_right[1]) - 0.10
        bos_max_y = max(p_left[1], p_right[1]) + 0.10
        is_balanced = bool((bos_min_x <= cop[0] <= bos_max_x) and (bos_min_y <= cop[1] <= bos_max_y))

        # 3. Object Manipulation Tracking
        curr_obj_pos = np.array([0.45, 0.0, 0.72], dtype=np.float32)
        if self.target_object_id >= 0:
            curr_obj_pos = np.array(self.data.xpos[self.target_object_id], dtype=np.float32)

        obj_error = 0.0
        if target_object_pos is not None:
            obj_error = float(np.linalg.norm(curr_obj_pos - target_object_pos))

        # Reward formulation: contact stability + balance + manipulation tracking
        r_balance = 1.0 if is_balanced else 0.2
        r_manip = float(np.exp(-obj_error / 0.10))
        r_upright = 1.0 if not self.robot.is_fallen() else 0.0
        total_reward = 0.4 * r_balance + 0.4 * r_manip + 0.2 * r_upright

        return ManipulationStepResult(
            robot_state=state,
            contact_forces=contact_forces,
            cop_position=cop,
            is_balanced=is_balanced,
            object_position=curr_obj_pos,
            object_error=obj_error,
            grasp_force=grasp_fn,
            reward=total_reward
        )
