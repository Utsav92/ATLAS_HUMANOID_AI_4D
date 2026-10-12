"""Kinematics and Damped Least-Squares (DLS) Inverse Kinematics for G1.
Enables reaching, grasping, and Cartesian end-effector tracking.
"""

from __future__ import annotations
import numpy as np
import mujoco
from typing import Tuple, Optional


class G1Kinematics:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Body IDs
        self.left_hand_id = self._find_body(["left_wrist_yaw_link", "left_hand", "left_palm"])
        self.right_hand_id = self._find_body(["right_wrist_yaw_link", "right_hand", "right_palm"])

    def get_arm_dofs(self, side: str = "right") -> np.ndarray:
        """Return array of DOF indices for the specified arm."""
        prefix = f"{side}_shoulder"
        elbow = f"{side}_elbow"
        wrist = f"{side}_wrist"
        dofs = []
        for i in range(self.model.njnt):
            jname = self.model.joint(i).name
            if prefix in jname or elbow in jname or wrist in jname:
                dofs.append(int(self.model.jnt_dofadr[i]))
        return np.array(dofs, dtype=np.int32)

    def _find_body(self, candidate_names) -> int:
        for name in candidate_names:
            try:
                return self.model.body(name).id
            except KeyError:
                continue
        return 1

    def get_forward_kinematics(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute end-effector 3D position [x, y, z] and 3x3 rotation matrix."""
        mujoco.mj_forward(self.model, self.data)
        pos = np.array(self.data.xpos[body_id], dtype=np.float32)
        rot = np.array(self.data.xmat[body_id].reshape(3, 3), dtype=np.float32)
        return pos, rot

    def solve_dls_ik(
        self,
        body_id: int,
        target_pos: np.ndarray,
        arm_dof_indices: np.ndarray,
        damping: float = 0.05,
        max_iters: int = 50,
        tolerance: float = 0.01,
        step_size: float = 0.5
    ) -> Tuple[np.ndarray, bool, float]:
        """Solve Damped Least-Squares IK for specified body to target position.
        
        Returns:
            (target_qpos, success_flag, final_error_norm)
        """
        # Work on a local copy of data to avoid corrupting simulation state during IK iterations
        test_data = mujoco.MjData(self.model)
        test_data.qpos[:] = self.data.qpos[:]
        test_data.qvel[:] = 0.0

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        success = False
        final_err = 1e6

        for _ in range(max_iters):
            mujoco.mj_forward(self.model, test_data)
            current_pos = test_data.xpos[body_id]
            err = target_pos - current_pos
            err_norm = float(np.linalg.norm(err))
            final_err = err_norm

            if err_norm < tolerance:
                success = True
                break

            # Compute Jacobian at body position
            mujoco.mj_jac(self.model, test_data, jacp, jacr, current_pos, body_id)

            # Sub-Jacobian for arm DOFs only
            J = jacp[:, arm_dof_indices]

            # DLS formulation: J_dls = J^T (J J^T + lambda^2 I)^(-1)
            JJT = J @ J.T + (damping ** 2) * np.eye(3)
            J_dls = J.T @ np.linalg.inv(JJT)

            # Joint displacement
            dq = step_size * (J_dls @ err)

            # Apply displacement to test qpos
            for idx, dof_idx in enumerate(arm_dof_indices):
                # find corresponding qposadr
                # For revolute joints, dof_idx maps directly to qposadr
                qpos_adr = self.model.jnt_qposadr[self.model.dof_jntid[dof_idx]]
                test_data.qpos[qpos_adr] += dq[idx]

        return np.array(test_data.qpos, copy=True), success, final_err
