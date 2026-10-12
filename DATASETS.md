# ATLAS HUMANOID AI 4D — Complete Dataset Directory

This directory provides canonical schemas, source-specific adapters, and split management for all 13 selected dataset/framework sources plus the locally generated ATLAS Language-to-Action dataset.

## Dataset Checklist & Manifest

| # | Dataset / Framework | Official Links & Downloads | Primary Purpose | Embodiment | Frame | License |
|---|---------------------|---------------------------|-----------------|------------|-------|---------|
| 1 | **G1 Moves** | [Hugging Face](https://huggingface.co/datasets/exptech/g1-moves) · [GitHub](https://github.com/experientialtech/g1-moves) | Humanoid locomotion and imitation learning | Unitree G1 | FLU | Apache-2.0 |
| 2 | **Unitree G1 Manipulation** | [Unitree Open Source](https://www.unitree.com/mobile/opensource/) | Humanoid grasping and manipulation demonstrations | Unitree G1 DexHands | FLU | CC-BY-4.0 |
| 3 | **MuJoCo-generated Trajectories** | [MuJoCo](https://github.com/google-deepmind/mujoco) · [G1 Models](https://github.com/google-deepmind/mujoco_menagerie) | Reinforcement learning & balance control | Unitree G1 | FLU | MIT |
| 4 | **TUM RGB-D** | [Official Dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) | RGB-D SLAM & 3D reconstruction | Sensor Rig | RDF | CC-BY-3.0 |
| 5 | **EuRoC MAV** | [Official Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) | Visual-inertial odometry & SLAM | MAV Rig | RDF | CC-BY-4.0 |
| 6 | **KITTI Odometry** | [Official Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | LiDAR & visual odometry | Ground Vehicle | FLU | CC-BY-NC-SA-3.0 |
| 7 | **TartanAir** | [Official Dataset](https://tartanair.org/) | Synthetic 3D vision, depth & SLAM | Simulated Agent | RDF | BSD-3-Clause |
| 8 | **YCB-Sight** | [GitHub & Downloads](https://github.com/Robo-Touch/YCB-Sight) | Tactile sensing & object understanding | GelSight Sensor | SensorSurface | MIT |
| 9 | **Sparsh** | [GitHub](https://github.com/facebookresearch/sparsh) · [Hugging Face](https://huggingface.co/facebook/sparsh) | Tactile representation learning | Multi-Sensor Rig | SensorSurface | Apache-2.0 |
| 10 | **DROID** | [Official Dataset](https://droid-dataset.github.io/droid/the-droid-dataset/) | Robot manipulation demonstrations | Franka Panda | FLU | CC-BY-4.0 |
| 11 | **Open X-Embodiment** | [GitHub](https://github.com/google-deepmind/open_x_embodiment) | Cross-robot policy learning | 22 Embodiments | FLU | Apache-2.0 |
| 12 | **LeRobot** | [GitHub](https://github.com/huggingface/lerobot) · [Hugging Face](https://huggingface.co/lerobot) | Robot learning & tooling | ALOHA / Koch | FLU | Apache-2.0 |
| 13 | **Minari** | [Official Website](https://minari.farama.org/) · [GitHub](https://github.com/Farama-Foundation/Minari) | Offline reinforcement learning | Gymnasium Standard | FLU | Apache-2.0 |
| 14 | **ATLAS Language-to-Action** | Generated locally in `atlas/datasets/` | NLP & Generative AI task planning | Unitree G1 Multimodal | FLU | Apache-2.0 |

---

## Canonical Adapter API

Every dataset source is wrapped in a subclass of `BaseDatasetAdapter`:

```python
from atlas.datasets.registry import DatasetRegistry

registry = DatasetRegistry()
adapter = registry.get_adapter("g1_moves_v1")

# 1. Metadata inspection
manifest = adapter.manifest
print(manifest.name, manifest.url, manifest.purpose)

# 2. Sample access
sample = adapter.get_sample(0)

# 3. Coordinate conversion (e.g. RDF camera to FLU robot frame)
flu_pos = adapter.transform_coordinates(sample.base_pose[:3], source_frame="RDF", target_frame="FLU")

# 4. Joint mapping validation (prevent direct incompatible merging)
mapped_q, mapped_names, missing = adapter.map_joints_to_g1(
    sample.joint_names, sample.joint_positions, target_g1_names
)
```
