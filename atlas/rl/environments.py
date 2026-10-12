"""Gymnasium Reinforcement Learning Environments for Unitree G1 Humanoid.
Supports PPO training for standing balance, locomotion, and reaching.
"""

from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any

from atlas.simulation.world import AtlasWorld
from atlas.core.types import RobotState


class G1HumanoidEnv(gym.Env):
    """Base Gymnasium environment for Unitree G1 Humanoid continuous control."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, xml_path: Optional[str] = None, max_episode_steps: int = 500):
        super().__init__()
        self.world = AtlasWorld(xml_path=xml_path)
        self.robot = self.world.robot
        self.nu = self.robot.model.nu
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        # Action space: normalized [-1, 1] offsets around nominal stand ctrl
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        # Observation space:
        # Base pos z (1), Base orientation quat (4), Base linvel (3), Base angvel (3)
        # Joint positions (nu) - floating base (7 qpos)
        # Joint velocities (nu)
        # Feet contacts (2)
        # Target velocity (2) [vx_cmd, vy_cmd]
        obs_dim = 1 + 4 + 3 + 3 + self.nu + self.nu + 2 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.target_vel = np.zeros(2, dtype=np.float32)

    def _get_obs(self, state: RobotState) -> np.ndarray:
        obs = np.concatenate([
            [state.base_position[2]],
            state.base_orientation,
            state.base_linear_velocity,
            state.base_angular_velocity,
            state.qpos[7:7 + self.nu] if len(state.qpos) >= 7 + self.nu else np.zeros(self.nu),
            state.qvel[6:6 + self.nu] if len(state.qvel) >= 6 + self.nu else np.zeros(self.nu),
            [float(state.feet_contacts["left_foot"]), float(state.feet_contacts["right_foot"])],
            self.target_vel
        ]).astype(np.float32)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        state = self.world.reset()
        obs = self._get_obs(state)
        return obs, {"state": state}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # Scale action into delta offsets from nominal stand
        scale = 0.25  # Max rad offset per action
        target_ctrl = self.robot.stand_ctrl + action * scale
        state = self.world.step(target_ctrl)

        obs = self._get_obs(state)

        # Reward decomposition
        # 1. Height reward
        r_height = float(np.exp(-10.0 * (state.base_position[2] - 0.79) ** 2))

        # 2. Upright reward (quaternion w should be close to 1)
        w, x, y, z = state.base_orientation
        up_z = 1.0 - 2.0 * (x * x + y * y)
        r_upright = float(max(0.0, up_z))

        # 3. Action penalty
        r_ctrl = -0.01 * float(np.sum(action ** 2))

        # 4. Velocity tracking reward
        vel_err = np.linalg.norm(state.base_linear_velocity[:2] - self.target_vel)
        r_vel = float(np.exp(-2.0 * vel_err))

        reward = 1.5 * r_height + 2.0 * r_upright + 1.0 * r_vel + r_ctrl

        # Fall termination
        terminated = self.robot.is_fallen()
        if terminated:
            reward -= 5.0

        truncated = (self.current_step >= self.max_episode_steps)

        info = {
            "r_height": r_height,
            "r_upright": r_upright,
            "r_vel": r_vel,
            "r_ctrl": r_ctrl,
            "fallen": terminated
        }

        return obs, float(reward), terminated, truncated, info


class G1StandingEnv(G1HumanoidEnv):
    """Environment focused on maintaining upright balance under zero command velocity."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_vel = np.array([0.0, 0.0], dtype=np.float32)


class G1WalkingEnv(G1HumanoidEnv):
    """Environment for forward locomotion tracking vx=0.4 m/s."""
    def __init__(self, target_vx: float = 0.4, **kwargs):
        super().__init__(**kwargs)
        self.target_vel = np.array([target_vx, 0.0], dtype=np.float32)
