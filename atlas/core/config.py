"""System configuration and resource constraints for ATLAS HUMANOID AI 4D.
Tailored for Windows 10/11, 8GB RAM, CPU-first PyTorch and ONNX Runtime execution.
"""

from __future__ import annotations
import os
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SystemLimits:
    max_ram_gb: float = 8.0
    use_cuda: bool = False
    device: str = "cpu"
    num_threads: int = 4
    batch_size: int = 16


@dataclass
class TouchDesignerConfig:
    width: int = 640
    height: int = 360
    fps: int = 15
    max_particles: int = 500
    max_point_cloud_points: int = 2000
    history_poses: int = 5
    enable_shadows: bool = False
    enable_bloom: bool = False
    enable_antialiasing: bool = False
    osc_port: int = 9000
    ws_port: int = 9982
    ipc_dir: Path = Path(r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D\data\ipc")


@dataclass
class RobotConfig:
    model_name: str = "unitree_g1"
    model_xml_path: Path = Path(r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D\assets\unitree_g1\scene.xml")
    hands_model_xml_path: Path = Path(r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D\assets\unitree_g1\scene_with_hands.xml")
    task_scene_xml_path: Path = Path(r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D\assets\unitree_g1\atlas_task_scene.xml")
    simulation_dt: float = 0.002
    control_dt: float = 0.02  # 50 Hz control loop
    standing_height: float = 0.79
    base_mass: float = 35.0  # kg approximate total


@dataclass
class MasterConfig:
    limits: SystemLimits = field(default_factory=SystemLimits)
    td: TouchDesignerConfig = field(default_factory=TouchDesignerConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    repo_root: Path = Path(r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D")


CONFIG = MasterConfig()
