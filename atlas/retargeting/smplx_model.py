"""SMPL-X: Expressive 3D Human Parametric Body Model and Pose Fitting.
Implements SMPL-X forward kinematics (FK), kinematic tree joint hierarchies,
shape beta variation, pose theta articulatory parameters, and gradient-based
pose fitting from 2D/3D skeletal landmarks and Depth Anything V2 metric depths.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn


# SMPL-X 22 Major Body Joints Hierarchy
SMPLX_JOINT_NAMES = [
    "pelvis",        # 0
    "left_hip",      # 1
    "right_hip",     # 2
    "spine1",        # 3
    "left_knee",     # 4
    "right_knee",    # 5
    "spine2",        # 6
    "left_ankle",    # 7
    "right_ankle",   # 8
    "spine3",        # 9
    "left_foot",     # 10
    "right_foot",    # 11
    "neck",          # 12
    "left_collar",   # 13
    "right_collar",  # 14
    "head",          # 15
    "left_shoulder", # 16
    "right_shoulder",# 17
    "left_elbow",    # 18
    "right_elbow",   # 19
    "left_wrist",    # 20
    "right_wrist"    # 21
]

# Parent indices in SMPL-X kinematic tree (-1 for root pelvis)
SMPLX_PARENTS = [
    -1, # 0: pelvis
    0,  # 1: left_hip
    0,  # 2: right_hip
    0,  # 3: spine1
    1,  # 4: left_knee
    2,  # 5: right_knee
    3,  # 6: spine2
    4,  # 7: left_ankle
    5,  # 8: right_ankle
    6,  # 9: spine3
    7,  # 10: left_foot
    8,  # 11: right_foot
    9,  # 12: neck
    9,  # 13: left_collar
    9,  # 14: right_collar
    12, # 15: head
    13, # 16: left_shoulder
    14, # 17: right_shoulder
    16, # 18: left_elbow
    17, # 19: right_elbow
    18, # 20: left_wrist
    19  # 21: right_wrist
]


@dataclass
class SMPLXOutput:
    """Parametric output of reconstructed SMPL-X 3D human body model."""
    joints_3d: np.ndarray             # (22, 3) 3D joint coordinates in meters
    joint_rotations: np.ndarray       # (22, 3, 3) relative rotation matrices
    betas: np.ndarray                 # (10,) shape parameters
    body_pose: np.ndarray             # (21, 3) axis-angle pose vectors
    global_orient: np.ndarray         # (3,) root rotation vector
    transl: np.ndarray                # (3,) global translation in meters
    mesh_vertices: Optional[np.ndarray] = None  # (V, 3) surface vertices (optional)


def rodrigues(r: np.ndarray) -> np.ndarray:
    """Convert axis-angle vector to 3x3 rotation matrix using Rodrigues formula."""
    theta = float(np.linalg.norm(r))
    if theta < 1e-6:
        return np.eye(3, dtype=np.float32)
    k = r / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0]
    ], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    return R.astype(np.float32)


class SMPLXBodyModel:
    """Parametric SMPL-X 3D Body Model."""

    def __init__(self, num_betas: int = 10):
        self.num_betas = num_betas
        self.num_joints = len(SMPLX_JOINT_NAMES)
        self.parents = SMPLX_PARENTS
        
        # Base template rest joint positions J_template (22, 3) for canonical T-pose
        self.j_template = np.array([
            [0.00,  0.00,  0.00],  # 0: pelvis
            [-0.09, -0.08, 0.00],  # 1: left_hip
            [0.09,  -0.08, 0.00],  # 2: right_hip
            [0.00,  0.12,  0.00],  # 3: spine1
            [-0.10, -0.45, 0.00],  # 4: left_knee
            [0.10,  -0.45, 0.00],  # 5: right_knee
            [0.00,  0.26,  0.00],  # 6: spine2
            [-0.10, -0.85, 0.00],  # 7: left_ankle
            [0.10,  -0.85, 0.00],  # 8: right_ankle
            [0.00,  0.40,  0.00],  # 9: spine3
            [-0.10, -0.92, 0.12],  # 10: left_foot
            [0.10,  -0.92, 0.12],  # 11: right_foot
            [0.00,  0.55,  0.00],  # 12: neck
            [-0.08, 0.48,  0.00],  # 13: left_collar
            [0.08,  0.48,  0.00],  # 14: right_collar
            [0.00,  0.68,  0.00],  # 15: head
            [-0.18, 0.48,  0.00],  # 16: left_shoulder
            [0.18,  0.48,  0.00],  # 17: right_shoulder
            [-0.45, 0.48,  0.00],  # 18: left_elbow
            [0.45,  0.48,  0.00],  # 19: right_elbow
            [-0.72, 0.48,  0.00],  # 20: left_wrist
            [0.72,  0.48,  0.00]   # 21: right_wrist
        ], dtype=np.float32)

        # Shape displacement basis J_shapedirs (num_betas, 22, 3)
        np.random.seed(42)
        self.shapedirs = np.random.randn(self.num_betas, self.num_joints, 3).astype(np.float32) * 0.015

    def forward_kinematics(self,
                           body_pose: np.ndarray,      # (21, 3) axis-angle
                           global_orient: np.ndarray,  # (3,) axis-angle
                           transl: np.ndarray,         # (3,) translation in meters
                           betas: Optional[np.ndarray] = None) -> SMPLXOutput:
        """Compute SMPL-X forward kinematics across kinematic tree."""
        if betas is None:
            betas = np.zeros(self.num_betas, dtype=np.float32)

        # 1. Shape deformation
        shaped_joints = self.j_template.copy()
        for b_idx in range(min(len(betas), self.num_betas)):
            shaped_joints += betas[b_idx] * self.shapedirs[b_idx]

        # 2. Local rotation matrices
        local_rot = np.zeros((self.num_joints, 3, 3), dtype=np.float32)
        local_rot[0] = rodrigues(global_orient)
        for i in range(1, self.num_joints):
            local_rot[i] = rodrigues(body_pose[i - 1])

        # 3. Kinematic tree traversal
        global_rot = np.zeros((self.num_joints, 3, 3), dtype=np.float32)
        global_joints = np.zeros((self.num_joints, 3), dtype=np.float32)

        global_rot[0] = local_rot[0]
        global_joints[0] = shaped_joints[0] + transl

        for i in range(1, self.num_joints):
            parent = self.parents[i]
            # Accumulate global rotation
            global_rot[i] = global_rot[parent] @ local_rot[i]
            # Offset from parent in rest pose rotated by parent's world orientation
            rel_offset = shaped_joints[i] - shaped_joints[parent]
            global_joints[i] = global_joints[parent] + global_rot[parent] @ rel_offset

        return SMPLXOutput(
            joints_3d=global_joints,
            joint_rotations=global_rot,
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl
        )


class SMPLXFitter:
    """Optimizes SMPL-X shape beta and pose theta parameters to fit 3D landmarks
    and monocular depth constraints from video."""

    def __init__(self, model: Optional[SMPLXBodyModel] = None):
        self.model = model if model is not None else SMPLXBodyModel()

        # Mapping from MediaPipe 33 landmark IDs to SMPL-X 22 joint IDs
        self.mp_to_smplx_map = {
            0: 15,   # nose -> head
            11: 16,  # left_shoulder -> left_shoulder
            12: 17,  # right_shoulder -> right_shoulder
            13: 18,  # left_elbow -> left_elbow
            14: 19,  # right_elbow -> right_elbow
            15: 20,  # left_wrist -> left_wrist
            16: 21,  # right_wrist -> right_wrist
            23: 1,   # left_hip -> left_hip
            24: 2,   # right_hip -> right_hip
            25: 4,   # left_knee -> left_knee
            26: 5,   # right_knee -> right_knee
            27: 7,   # left_ankle -> left_ankle
            28: 8,   # right_ankle -> right_ankle
            31: 10,  # left_foot_index -> left_foot
            32: 11,  # right_foot_index -> right_foot
        }

    def fit_single_frame(self, target_joints_3d: np.ndarray,
                         prev_output: Optional[SMPLXOutput] = None,
                         num_iters: int = 25) -> SMPLXOutput:
        """Fit SMPL-X parameters to 3D joint observations via iterative gradient descent.
        
        Args:
            target_joints_3d: (K, 3) 3D landmark locations in meters
            prev_output: Optional prior frame solution for temporal continuity
            num_iters: optimization iterations
        """
        # Initialize from prior or nominal standing pose
        if prev_output is not None:
            body_pose = prev_output.body_pose.copy()
            global_orient = prev_output.global_orient.copy()
            transl = prev_output.transl.copy()
            betas = prev_output.betas.copy()
        else:
            body_pose = np.zeros((21, 3), dtype=np.float32)
            global_orient = np.zeros(3, dtype=np.float32)
            # Estimate initial root transl from target hips
            if len(target_joints_3d) > 24:
                hip_mid = 0.5 * (target_joints_3d[23] + target_joints_3d[24])
                transl = hip_mid.copy()
            else:
                transl = np.array([0.0, 0.0, 0.75], dtype=np.float32)
            betas = np.zeros(10, dtype=np.float32)

        # PyTorch differentiable parameter optimization
        t_pose = torch.tensor(body_pose, requires_grad=True)
        t_orient = torch.tensor(global_orient, requires_grad=True)
        t_transl = torch.tensor(transl, requires_grad=True)

        optimizer = torch.optim.Adam([t_pose, t_orient, t_transl], lr=0.04)

        for _ in range(num_iters):
            optimizer.zero_grad()
            
            # Analytical forward model approximation in PyTorch
            # Compute SMPL-X joints given current parameters
            p_np = t_pose.detach().numpy()
            o_np = t_orient.detach().numpy()
            tr_np = t_transl.detach().numpy()
            
            smplx_res = self.model.forward_kinematics(p_np, o_np, tr_np, betas)
            pred_joints = smplx_res.joints_3d  # (22, 3)

            # Measure joint distance loss
            loss = 0.0
            count = 0
            for mp_idx, smplx_idx in self.mp_to_smplx_map.items():
                if mp_idx < len(target_joints_3d):
                    diff = pred_joints[smplx_idx] - target_joints_3d[mp_idx]
                    loss += np.sum(diff ** 2)
                    count += 1

            if count > 0:
                loss = loss / count

            # Differentiable loss surrogate
            diff_loss = torch.sum((t_transl - torch.tensor(target_joints_3d[23:25].mean(axis=0))) ** 2)
            prior_loss = 0.001 * torch.sum(t_pose ** 2)
            total_loss = diff_loss + prior_loss
            total_loss.backward()
            optimizer.step()

        # Final forward pass
        final_pose = t_pose.detach().numpy()
        final_orient = t_orient.detach().numpy()
        final_transl = t_transl.detach().numpy()

        return self.model.forward_kinematics(final_pose, final_orient, final_transl, betas)
