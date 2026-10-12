# ATLAS HUMANOID AI 4D — Architecture Specification

## Overview

ATLAS HUMANOID AI 4D is a unified multimodal embodied artificial intelligence platform engineered for bipedal humanoid robotics (Unitree G1). The system integrates perception, spatial reasoning, language understanding, generative task planning, reinforcement learning, whole-body control, offline imitation learning, tactile intelligence, neural deployment, and a 4D Digital Twin in TouchDesigner.

```
                                  USER COMMAND (NLP / GenAI)
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │   NLP & GenAI Plan   │
                                  │ (8-Step FSM Pipeline)│
                                  └──────────┬───────────┘
                                             │
                     ┌───────────────────────┼────────────────────────┐
                     ▼                       ▼                        ▼
           ┌──────────────────┐    ┌───────────────────┐    ┌───────────────────┐
           │   RGB-D SLAM     │    │ Tactile Sensing   │    │  Proprioception   │
           │ (ICP & Odometry) │    │(GelSight Features)│    │(Joints, CoM, ZMP) │
           └─────────┬────────┘    └─────────┬─────────┘    └─────────┬─────────┘
                     │                       │                        │
                     └───────────────────────┼────────────────────────┘
                                             ▼
                                  ┌──────────────────────┐
                                  │  Multimodal Fusion   │
                                  │ (Concat / Attention) │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │  Whole-Body Control  │
                                  │(Stabilizer, DLS IK,  │
                                  │  PPO, Supervisor)    │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │   MuJoCo Simulator   │
                                  │ (Unitree G1 43-DOF)  │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │   TouchDesigner 4D   │
                                  │ (15-Node Telemetry)  │
                                  └──────────────────────┘
```

---

## 1. System Components

### 1.1 Core & Simulation (`atlas/core/`, `atlas/simulation/`)
- **Robot Model**: Unitree G1 (29-DoF biped + 14 dexterous hand actuators = 43 actuators total, 57 generalized coordinates).
- **Physics Engine**: MuJoCo 3.14.0 (2ms physics step, 20ms / 50Hz control loop).
- **Task Scene**: Table, red graspable cube with freejoint, and target placement shelf.

### 1.2 Dataset Hub (`atlas/datasets/`, `configs/dataset_registry.yaml`)
- 13 verified dataset sources + ATLAS Language-to-Action.
- Canonical lifecycle: `REGISTERED -> VERIFIED -> DOWNLOADED -> VALIDATED -> CONVERTED -> INTEGRATED -> TESTED`.
- Coordinate transforms between RDF (camera) and FLU (robot).

### 1.3 Machine Learning & Deep Learning (`atlas/ml/`, `atlas/deep_learning/`)
- Random Forest & Logistic Regression for contact state classification.
- Isolation Forest for sensor anomaly detection.
- K-Means & PCA for unsupervised gait clustering and dimensionality reduction.
- Compact PyTorch networks: MLP, CNN, GRU, Temporal Convolution (TCN), Autoencoder, and Transformer Encoder.

### 1.4 Perception & SLAM (`atlas/vision/`, `atlas/slam/`)
- RGB-D camera acquisition and pinhole projection matrix.
- Color segmentation & depth back-projection for 3D red cube localization.
- ICP registration using Kabsch-Umeyama SVD.
- Trajectory evaluation: ATE RMSE (12.12 mm) and RPE RMSE (16.47 mm).

### 1.5 Reinforcement & Imitation Learning (`atlas/rl/`, `atlas/imitation/`)
- Stable-Baselines3 PPO for continuous standing balance and locomotion.
- Gymnasium environments (`G1StandingEnv`, `G1WalkingEnv`).
- Behavior Cloning (BC) and Temporal BC with action chunking on G1 demonstrations.

### 1.6 Multimodal Sensor Fusion (`atlas/fusion/`, `atlas/tactile/`)
- Tactile GelSight deformation and contact state classification.
- Multimodal fusion network combining kinematics, vision, tactile, and instruction embeddings.
- Zero-NaN missing modality masks.

### 1.7 Planning & Whole-Body Control (`atlas/nlp/`, `atlas/planning/`, `atlas/control/`)
- Natural Language Instruction parsing into 8-step FSM (`LOCATE`, `APPROACH`, `STABILIZE`, `REACH`, `GRASP`, `LIFT`, `TRANSPORT`, `PLACE`).
- Damped Least-Squares (DLS) Inverse Kinematics for arm reaching.
- Posture stabilizer using Center of Mass (CoM) and Zero Moment Point (ZMP) feedback.
- Safety supervisor enforcing torque limits and base tilt boundaries.

### 1.8 Neural Deployment (`atlas/deployment/`)
- PyTorch to ONNX export (`models/onnx/g1_policy.onnx`).
- ONNX Runtime CPU inference with 4 threads.
- Measured batch-one inference latency: **p50 = 0.055 ms** (55 microseconds).

### 1.9 TouchDesigner 4D Digital Twin (`touchdesigner/`)
- Project `/project1/ATLAS_HUMANOID_AI_4D` containing all 15 components:
  `ROBOT_RENDERER`, `SLAM_MAP`, `POINT_CLOUD`, `SKELETON`, `JOINT_HEATMAP`, `BALANCE_MONITOR`, `TACTILE_CONTACT`, `POLICY_MONITOR`, `NLP_COMMAND`, `AI_REASONING`, `TASK_PLANNER`, `TEMPORAL_4D`, `CAMERA_SYSTEM`, `DASHBOARD`, `OUTPUT`.
- Rendered live in real-time at 640x360 15 FPS.
