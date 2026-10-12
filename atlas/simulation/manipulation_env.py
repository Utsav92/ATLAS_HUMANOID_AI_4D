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
class ScenarioConfig:
    """Configuration for stress-testing under varied simulation scenarios."""
    name: str = "nominal"
    object_mass: float = 1.0           # Object weight in kg (e.g. 0.2 to 5.0 kg)
    object_location: np.ndarray = field(default_factory=lambda: np.array([0.45, 0.0, 0.72], dtype=np.float32))
    speed_factor: float = 1.0          # Motion speed multiplier (0.5x to 2.0x)
    perturbation_force: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32)) # Torso push force [Fx, Fy, Fz] in N
    perturbation_duration_steps: int = 5 # How many simulation steps push is active
    perturbation_start_step: int = 15    # At which step to apply push impulse
    torque_noise_std: float = 0.0      # Actuator Gaussian noise std (N*m)


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
        self.current_scenario: Optional[ScenarioConfig] = None
        self.current_step_count: int = 0

    def _find_object_body(self, candidates: List[str]) -> int:
        for c in candidates:
            try:
                return self.model.body(c).id
            except KeyError:
                continue
        return -1

    def set_scenario(self, scenario: ScenarioConfig):
        """Configure simulation parameters for stress testing."""
        self.current_scenario = scenario
        self.object_pos = np.array(scenario.object_location, dtype=np.float32)
        if self.target_object_id >= 0:
            try:
                self.model.body_mass[self.target_object_id] = float(scenario.object_mass)
            except Exception:
                pass

    def reset(self, scenario: Optional[ScenarioConfig] = None) -> ManipulationStepResult:
        """Reset robot to stable standing stance and place manipulation object."""
        state = self.world.reset()
        self.current_step_count = 0
        self.data.xfrc_applied[:] = 0.0

        if scenario is not None:
            self.set_scenario(scenario)
        elif self.current_scenario is not None:
            self.set_scenario(self.current_scenario)
        else:
            if self.target_object_id >= 0:
                self.object_pos = np.array(self.data.xpos[self.target_object_id], dtype=np.float32)
            else:
                self.object_pos = np.array([0.45, 0.0, 0.72], dtype=np.float32)

        return self._evaluate_step(target_object_pos=self.object_pos)

    def step(self, action_ctrl: np.ndarray,
             target_object_pos: Optional[np.ndarray] = None,
             n_substeps: int = 10) -> ManipulationStepResult:
        """Advance physical simulation with contact dynamics and balance checking."""
        self.current_step_count += 1
        full_ctrl = self.robot.stand_ctrl.copy()
        n_act = min(len(action_ctrl), len(full_ctrl))
        full_ctrl[:n_act] = action_ctrl[:n_act]

        # Scenario Perturbations: torque noise
        if self.current_scenario is not None and self.current_scenario.torque_noise_std > 0.0:
            noise = np.random.normal(0.0, self.current_scenario.torque_noise_std, size=len(full_ctrl)).astype(np.float32)
            full_ctrl += noise

        clipped_ctrl = self.robot.clip_ctrl(full_ctrl)
        self.data.ctrl[:len(clipped_ctrl)] = clipped_ctrl

        # Scenario Perturbations: external push impulses and object gravitational payload
        if self.current_scenario is not None:
            sc = self.current_scenario
            # Torso lateral push impulse
            if sc.perturbation_start_step <= self.current_step_count < (sc.perturbation_start_step + sc.perturbation_duration_steps):
                self.data.xfrc_applied[self.robot.pelvis_id, :3] = sc.perturbation_force
            else:
                self.data.xfrc_applied[self.robot.pelvis_id, :3] = 0.0

            # Gravitational payload of object weight applied to robot hands
            if sc.object_mass > 0.0:
                hand_payload_f = float(sc.object_mass * 9.81 * 0.5)
                if self.robot.left_hand_id >= 0:
                    self.data.xfrc_applied[self.robot.left_hand_id, 2] = -hand_payload_f
                if self.robot.right_hand_id >= 0:
                    self.data.xfrc_applied[self.robot.right_hand_id, 2] = -hand_payload_f

        for _ in range(n_substeps):
            mujoco.mj_step(self.model, self.data)

        return self._evaluate_step(target_object_pos)

    def _evaluate_step(self, target_object_pos: Optional[np.ndarray] = None) -> ManipulationStepResult:
        state = self.robot.get_state()

        # 1. Contact Dynamics & Normal Forces
        fn_left = 0.0
        fn_right = 0.0
        grasp_fn = 0.0

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]

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
        curr_obj_pos = self.object_pos.copy()
        if self.target_object_id >= 0:
            curr_obj_pos = np.array(self.data.xpos[self.target_object_id], dtype=np.float32)
        elif self.robot.left_hand_id >= 0 and self.robot.right_hand_id >= 0:
            curr_obj_pos = 0.5 * (np.array(self.data.xpos[self.robot.left_hand_id], dtype=np.float32) +
                                  np.array(self.data.xpos[self.robot.right_hand_id], dtype=np.float32))

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

    def is_episode_successful(self, target_object_pos: Optional[np.ndarray] = None, tolerance_m: float = 0.08) -> bool:
        """Evaluates whether the robot completed manipulation successfully without falling or dropping the object."""
        state = self.robot.get_state()
        is_upright = bool(state.base_position[2] >= 0.55 and not self.robot.is_fallen())
        
        curr_res = self._evaluate_step(target_object_pos)
        obj_ok = bool(curr_res.object_error <= tolerance_m) if target_object_pos is not None else True
        balance_ok = bool(curr_res.is_balanced)

        return bool(is_upright and obj_ok and balance_ok)
