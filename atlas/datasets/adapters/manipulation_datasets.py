"""Adapters for Cross-Robot Learning & Offline Demonstration Frameworks:
10. DROID
11. Open X-Embodiment
12. LeRobot
13. Minari
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
from atlas.datasets.adapters.base import BaseDatasetAdapter
from atlas.datasets.schema import DatasetManifest, TrajectorySample


class DROIDAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="DROID",
            source_id="droid_manipulation",
            version="1.0",
            license_spdx="CC-BY-4.0",
            robot_embodiment="Franka Emika Panda",
            num_episodes=76,
            num_samples=7600,
            features=["cartesian_pose", "joint_positions", "gripper_action", "exterior_rgb", "wrist_rgb"],
            coordinate_frame="FLU",
            citation="Khazatsky et al., DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset (2024)",
            url="https://droid-dataset.github.io/droid/the-droid-dataset/",
            purpose="Robot manipulation and imitation learning"
        )

    def get_sample_count(self) -> int:
        return 50

    def get_sample(self, index: int) -> TrajectorySample:
        panda_joints = [f"panda_joint_{i}" for i in range(7)] + ["panda_finger_joint"]
        return TrajectorySample(
            episode_id=f"droid_ep_{index // 10}",
            step_id=index % 10,
            timestamp=float(index) * 0.05,
            joint_names=panda_joints,
            joint_positions=np.zeros(8, dtype=np.float32),
            joint_velocities=np.zeros(8, dtype=np.float32),
            actions=np.zeros(8, dtype=np.float32),
            base_pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=(index % 10 == 9)
        )


class OpenXEmbodimentAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="Open X-Embodiment",
            source_id="open_x_embodiment_core",
            version="1.0",
            license_spdx="Apache-2.0",
            robot_embodiment="Heterogeneous Multi-Robot (22 embodiments)",
            num_episodes=500,
            num_samples=50000,
            features=["standardized_action_ee", "world_rgb", "language_instruction", "gripper_closedness"],
            coordinate_frame="FLU",
            citation="Open X-Embodiment Collaboration, Open X-Embodiment: Robotic Learning Datasets and RT-X, ICRA 2024",
            url="https://github.com/google-deepmind/open_x_embodiment",
            github_url="https://github.com/google-deepmind/open_x_embodiment",
            purpose="Cross-robot policy learning"
        )

    def get_sample_count(self) -> int:
        return 60

    def get_sample(self, index: int) -> TrajectorySample:
        return TrajectorySample(
            episode_id=f"open_x_ep_{index // 10}",
            step_id=index % 10,
            timestamp=float(index) * 0.1,
            joint_names=["ee_dx", "ee_dy", "ee_dz", "ee_droll", "ee_dpitch", "ee_dyaw", "gripper"],
            joint_positions=np.zeros(7, dtype=np.float32),
            joint_velocities=np.zeros(7, dtype=np.float32),
            actions=np.zeros(7, dtype=np.float32),
            base_pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            reward=1.0,
            done=(index % 10 == 9)
        )


class LeRobotAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="LeRobot",
            source_id="lerobot_hf_datasets",
            version="0.1.0",
            license_spdx="Apache-2.0",
            robot_embodiment="ALOHA / Koch Low-Cost Arms",
            num_episodes=100,
            num_samples=10000,
            features=["observation.state", "action", "next.reward", "next.done"],
            coordinate_frame="FLU",
            citation="Cadene et al., LeRobot: State-of-the-art Machine Learning for Real-World Robotics, Hugging Face 2024",
            url="https://github.com/huggingface/lerobot",
            github_url="https://github.com/huggingface/lerobot",
            hf_url="https://huggingface.co/lerobot",
            purpose="Robot learning and demonstration datasets"
        )

    def get_sample_count(self) -> int:
        return 40

    def get_sample(self, index: int) -> TrajectorySample:
        return TrajectorySample(
            episode_id=f"lerobot_ep_{index // 10}",
            step_id=index % 10,
            timestamp=float(index) * 0.033,
            joint_names=[f"aloha_joint_{i}" for i in range(14)],
            joint_positions=np.zeros(14, dtype=np.float32),
            joint_velocities=np.zeros(14, dtype=np.float32),
            actions=np.zeros(14, dtype=np.float32),
            base_pose=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            reward=0.5,
            done=False
        )


class MinariAdapter(BaseDatasetAdapter):
    def _build_manifest(self) -> DatasetManifest:
        return DatasetManifest(
            name="Minari",
            source_id="minari_offline_rl",
            version="0.4.2",
            license_spdx="Apache-2.0",
            robot_embodiment="Gymnasium Standard Environments (Humanoid / Ant / HalfCheetah)",
            num_episodes=120,
            num_samples=12000,
            features=["observations", "actions", "rewards", "terminations", "truncations"],
            coordinate_frame="FLU",
            citation="Farrahi et al., Minari: Standardizing Offline Reinforcement Learning Datasets, NeurIPS 2023",
            url="https://minari.farama.org/",
            github_url="https://github.com/Farama-Foundation/Minari",
            purpose="Offline reinforcement learning"
        )

    def get_sample_count(self) -> int:
        return 50

    def get_sample(self, index: int) -> TrajectorySample:
        return TrajectorySample(
            episode_id=f"minari_humanoid_ep_{index // 15}",
            step_id=index % 15,
            timestamp=float(index) * 0.02,
            joint_names=[f"minari_act_{i}" for i in range(17)],
            joint_positions=np.zeros(17, dtype=np.float32),
            joint_velocities=np.zeros(17, dtype=np.float32),
            actions=np.zeros(17, dtype=np.float32),
            base_pose=np.array([index * 0.03, 0.0, 1.2, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            reward=1.8,
            done=(index % 15 == 14)
        )
