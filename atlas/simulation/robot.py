"""Unitree G1 Robot Kinematic and State Interface for MuJoCo.
Handles joint mapping, actuator limits, standing posture, contact sensing, and state extraction.
"""

from __future__ import annotations
import numpy as np
import mujoco
from typing import Dict, List, Optional, Tuple, Any
from atlas.core.types import RobotState


class G1Robot:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Extract joint names and actuator mappings
        self.joint_names = [model.joint(i).name for i in range(model.njnt)]
        self.actuator_names = [model.actuator(i).name for i in range(model.nu)]
        
        # Build lookup tables
        self.actuator_name_to_id = {name: i for i, name in enumerate(self.actuator_names)}
        self.joint_name_to_id = {name: i for i, name in enumerate(self.joint_names)}

        # Identify body IDs for kinematics and contacts
        self.pelvis_id = model.body("pelvis").id if "pelvis" in [model.body(i).name for i in range(model.nbody)] else 1
        self.left_foot_id = self._find_body_id(["left_ankle_roll_link", "left_foot", "left_ankle"])
        self.right_foot_id = self._find_body_id(["right_ankle_roll_link", "right_foot", "right_ankle"])
        self.left_hand_id = self._find_body_id(["left_wrist_yaw_link", "left_hand", "left_palm"])
        self.right_hand_id = self._find_body_id(["right_wrist_yaw_link", "right_hand", "right_palm"])

        # Extract stand keyframe if present
        self.stand_qpos = self._extract_keyframe_qpos("stand")
        self.stand_ctrl = self._extract_keyframe_ctrl("stand")

        # Actuator effort limits
        self.ctrl_limits = np.zeros((model.nu, 2), dtype=np.float32)
        for i in range(model.nu):
            self.ctrl_limits[i] = model.actuator_ctrlrange[i]

    def _find_body_id(self, candidate_names: List[str]) -> int:
        for name in candidate_names:
            try:
                return self.model.body(name).id
            except KeyError:
                continue
        return -1

    def _extract_keyframe_qpos(self, key_name: str) -> np.ndarray:
        for i in range(self.model.nkey):
            if self.model.key(i).name == key_name:
                return np.array(self.model.key(i).qpos, dtype=np.float32)
        # Default fallback
        q = np.zeros(self.model.nq, dtype=np.float32)
        q[2] = 0.79  # Z height
        q[3] = 1.0   # Quaternion w
        return q

    def _extract_keyframe_ctrl(self, key_name: str) -> np.ndarray:
        for i in range(self.model.nkey):
            if self.model.key(i).name == key_name:
                return np.array(self.model.key(i).ctrl, dtype=np.float32)
        return np.zeros(self.model.nu, dtype=np.float32)

    def reset_to_stand(self):
        """Reset the robot state to the stable standing keyframe while preserving scene objects."""
        self.data.qpos[:] = self.model.qpos0
        # G1 robot floating base + body joints are the first 50 coordinates (or len of stand if no scene objects)
        robot_qpos_len = min(len(self.stand_qpos), 50 if self.model.nq >= 50 else len(self.stand_qpos))
        self.data.qpos[:robot_qpos_len] = self.stand_qpos[:robot_qpos_len]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:len(self.stand_ctrl)] = self.stand_ctrl
        mujoco.mj_forward(self.model, self.data)

    def get_state(self) -> RobotState:
        """Extract canonical RobotState from MuJoCo data."""
        # Base pose (assumes freejoint at pelvis)
        base_pos = np.array(self.data.qpos[0:3], dtype=np.float32)
        base_quat = np.array(self.data.qpos[3:7], dtype=np.float32)
        base_linvel = np.array(self.data.qvel[0:3], dtype=np.float32)
        base_angvel = np.array(self.data.qvel[3:6], dtype=np.float32)

        # Joint positions & velocities (excluding floating base 7 qpos / 6 qvel)
        joint_positions = {}
        joint_velocities = {}
        for j_idx, j_name in enumerate(self.joint_names):
            if j_name == "floating_base_joint":
                continue
            qpos_adr = self.model.jnt_qposadr[j_idx]
            dof_adr = self.model.jnt_dofadr[j_idx]
            joint_positions[j_name] = float(self.data.qpos[qpos_adr])
            joint_velocities[j_name] = float(self.data.qvel[dof_adr])

        # Contacts detection
        feet_contacts = {"left_foot": False, "right_foot": False}
        hand_contacts = {"left_hand": False, "right_hand": False}

        for c_idx in range(self.data.ncon):
            con = self.data.contact[c_idx]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            
            if b1 == self.left_foot_id or b2 == self.left_foot_id:
                feet_contacts["left_foot"] = True
            if b1 == self.right_foot_id or b2 == self.right_foot_id:
                feet_contacts["right_foot"] = True
            if b1 == self.left_hand_id or b2 == self.left_hand_id:
                hand_contacts["left_hand"] = True
            if b1 == self.right_hand_id or b2 == self.right_hand_id:
                hand_contacts["right_hand"] = True

        # Center of Mass (sub-tree CoM at pelvis body)
        com = np.array(self.data.subtree_com[self.pelvis_id], dtype=np.float32)

        # Zero Moment Point approximation from ground reaction
        # ZMP = [x_com - (z_com / g) * x_acc, y_com - (z_com / g) * y_acc]
        g = 9.81
        zmp = np.array([com[0] - (com[2] / g) * self.data.qacc[0],
                        com[1] - (com[2] / g) * self.data.qacc[1]], dtype=np.float32)

        return RobotState(
            timestamp=float(self.data.time),
            qpos=np.array(self.data.qpos, copy=True),
            qvel=np.array(self.data.qvel, copy=True),
            ctrl=np.array(self.data.ctrl, copy=True),
            base_position=base_pos,
            base_orientation=base_quat,
            base_linear_velocity=base_linvel,
            base_angular_velocity=base_angvel,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            feet_contacts=feet_contacts,
            hand_contacts=hand_contacts,
            com_position=com,
            zmp=zmp,
        )

    def is_fallen(self) -> bool:
        """Detect if the robot has fallen based on height and orientation."""
        torso_z = self.data.qpos[2]
        if torso_z < 0.45:  # Standing nominal is ~0.79m
            return True
        # Check quaternion tilt (roll / pitch > 50 degrees)
        w, x, y, z = self.data.qpos[3:7]
        # Up-vector z-component in world coords: 1 - 2*(x^2 + y^2)
        up_z = 1.0 - 2.0 * (x * x + y * y)
        if up_z < 0.6:  # Tilt exceeds ~53 degrees
            return True
        return False

    def clip_ctrl(self, ctrl: np.ndarray) -> np.ndarray:
        """Safely clip control inputs to actuator limits."""
        clipped = np.clip(ctrl, self.ctrl_limits[:, 0], self.ctrl_limits[:, 1])
        return clipped
