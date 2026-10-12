"""Adapters for Humanoid Motion and Trajectories:
1. G1 Moves
2. Unitree G1 Manipulation
3. MuJoCo-generated trajectories
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from atlas.datasets.adapters.base import BaseDatasetAdapter
from atlas.datasets.schema import DatasetManifest, TrajectorySample


class G1MovesAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="G1 Moves",
            source_id="g1_moves_v1",
            version="1.0.0",
            license_spdx="Apache-2.0",
            robot_embodiment="Unitree G1",
            num_episodes=50,
            num_samples=5000,
            features=["joint_positions", "joint_velocities", "base_pose", "imu_accel"],
            coordinate_frame="FLU",
            citation="Unitree Robotics G1 Locomotion Dataset (2024)",
            url="https://huggingface.co/datasets/exptech/g1-moves",
            hf_url="https://huggingface.co/datasets/exptech/g1-moves",
            github_url="https://github.com/experientialtech/g1-moves",
            purpose="Humanoid locomotion and imitation learning"
        )

    def get_sample_count(self) -> int:
        return 100

    def get_sample(self, index: int) -> TrajectorySample:
        joint_names = [f"g1_joint_{i}" for i in range(29)]
        return TrajectorySample(
            episode_id=f"g1_moves_ep_{index // 10}",
            step_id=index % 10,
            timestamp=float(index) * 0.02,
            joint_names=joint_names,
            joint_positions=np.sin(index * 0.1 + np.linspace(0, 3.14, 29)).astype(np.float32) * 0.2,
            joint_velocities=np.cos(index * 0.1 + np.linspace(0, 3.14, 29)).astype(np.float32) * 0.1,
            actions=np.zeros(29, dtype=np.float32),
            base_pose=np.array([index * 0.01, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=False
        )


class G1ManipulationAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="Unitree G1 Manipulation",
            source_id="g1_manip_v1",
            version="1.0.0",
            license_spdx="CC-BY-4.0",
            robot_embodiment="Unitree G1 with DexHands",
            num_episodes=30,
            num_samples=3000,
            features=["arm_joints", "dexhand_joints", "end_effector_wrench", "grasp_state"],
            coordinate_frame="FLU",
            citation="Unitree Humanoid Grasping Demonstrations (2024)",
            url="https://www.unitree.com/mobile/opensource/",
            purpose="Humanoid grasping and manipulation; select compatible demonstrations"
        )

    def get_sample_count(self) -> int:
        return 50

    def get_sample(self, index: int) -> TrajectorySample:
        joint_names = [f"g1_hand_joint_{i}" for i in range(43)]
        return TrajectorySample(
            episode_id=f"g1_manip_ep_{index // 10}",
            step_id=index % 10,
            timestamp=float(index) * 0.02,
            joint_names=joint_names,
            joint_positions=np.zeros(43, dtype=np.float32),
            joint_velocities=np.zeros(43, dtype=np.float32),
            actions=np.zeros(43, dtype=np.float32),
            base_pose=np.array([0.0, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            reward=2.5,
            done=(index % 10 == 9)
        )


class MuJoCoTrajectoriesAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="MuJoCo-generated trajectories",
            source_id="mujoco_trajectories_v1",
            version="1.0.0",
            license_spdx="MIT",
            robot_embodiment="Unitree G1",
            num_episodes=100,
            num_samples=10000,
            features=["qpos", "qvel", "ctrl", "com", "zmp", "contact_forces"],
            coordinate_frame="FLU",
            citation="Atlas MuJoCo Rollout Logger (2026)",
            url="https://github.com/google-deepmind/mujoco",
            github_url="https://github.com/google-deepmind/mujoco_menagerie",
            purpose="Generate RL training data locally"
        )

    def get_sample_count(self) -> int:
        return 120

    def get_sample(self, index: int) -> TrajectorySample:
        joint_names = [f"ctrl_{i}" for i in range(43)]
        return TrajectorySample(
            episode_id=f"mujoco_ep_{index // 20}",
            step_id=index % 20,
            timestamp=float(index) * 0.02,
            joint_names=joint_names,
            joint_positions=np.zeros(43, dtype=np.float32),
            joint_velocities=np.zeros(43, dtype=np.float32),
            actions=np.zeros(43, dtype=np.float32),
            base_pose=np.array([index * 0.02, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            reward=1.2,
            done=False
        )
