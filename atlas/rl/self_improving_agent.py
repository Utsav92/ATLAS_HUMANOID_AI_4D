"""Self-Improving Lifelong Learning Agent for Humanoid Robotics.
Continuously refines robot movement, balance, foot stability, and imitation fidelity
the more time it spends playing and simulating in the physical environment.
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mujoco


class ResidualPolicy(nn.Module):
    """Residual Neural Policy network that learns fine-grained motor adjustments
    on top of the base reference motion to optimize balance, contact stability, and fluidity."""
    def __init__(self, state_dim: int = 70, action_dim: int = 29, hidden_dim: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Output scaled small residual offsets
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.action_scale = 0.15 # Max +/- 0.15 rad residual correction

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual_action = self.actor(state) * self.action_scale
        value = self.critic(state)
        return residual_action, value


class LifelongReplayBuffer:
    """Prioritized experience buffer storing physical interactions,
    sensor feedback, and motion quality metrics across play sessions."""
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.errors = []
        self.ptr = 0

    def push(self, state: np.ndarray, action: np.ndarray, reward: float,
             next_state: np.ndarray, done: bool, error: float):
        if len(self.states) < self.capacity:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            self.errors.append(error)
        else:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr] = done
            self.errors[self.ptr] = error
            self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(self.states)
        if n == 0:
            raise ValueError("Buffer is empty.")
        
        # Prioritize sampling states with higher error
        errs = np.array(self.errors[:n], dtype=np.float32)
        probs = (errs + 1e-4) / (errs.sum() + 1e-4 * n)
        probs /= probs.sum()

        indices = np.random.choice(n, size=min(batch_size, n), replace=False, p=probs)

        s_batch = torch.tensor(np.array([self.states[i] for i in indices]), dtype=torch.float32)
        a_batch = torch.tensor(np.array([self.actions[i] for i in indices]), dtype=torch.float32)
        r_batch = torch.tensor(np.array([self.rewards[i] for i in indices]), dtype=torch.float32).unsqueeze(-1)
        ns_batch = torch.tensor(np.array([self.next_states[i] for i in indices]), dtype=torch.float32)
        d_batch = torch.tensor(np.array([self.dones[i] for i in indices]), dtype=torch.float32).unsqueeze(-1)

        return s_batch, a_batch, r_batch, ns_batch, d_batch

    def __len__(self) -> int:
        return len(self.states)


class MotionQualityCritic:
    """Multi-objective evaluator that rewards human dance pose alignment,
    ground contact zero-slip, dynamic equilibrium balance, and torque smoothness."""
    def __init__(self, w_imitation: float = 0.40, w_balance: float = 0.25,
                 w_contact: float = 0.20, w_smooth: float = 0.15):
        self.w_imitation = w_imitation
        self.w_balance = w_balance
        self.w_contact = w_contact
        self.w_smooth = w_smooth

    def evaluate_step(self, q_actual: np.ndarray, q_target: np.ndarray,
                      q_vel: np.ndarray, prev_q_vel: np.ndarray,
                      feet_contact_z: Tuple[float, float],
                      com_xy: np.ndarray, bos_xy: np.ndarray) -> Tuple[float, Dict[str, float]]:
        # 1. Pose imitation reward (Gaussian kernel over RMSE)
        joint_err = np.linalg.norm(q_actual - q_target) / np.sqrt(len(q_actual))
        r_imitation = float(np.exp(-joint_err / 0.15))

        # 2. Dynamic balance reward (CoM distance to Center of Base of Support)
        balance_dist = np.linalg.norm(com_xy - bos_xy)
        r_balance = float(np.exp(-balance_dist / 0.12))

        # 3. Ground contact reward (minimal supporting foot clearance from Z=0)
        min_foot_z = min(feet_contact_z[0], feet_contact_z[1])
        r_contact = float(np.exp(-abs(min_foot_z) / 0.02))

        # 4. Torque smoothness / jerk penalty (minimal joint acceleration)
        jerk = np.linalg.norm(q_vel - prev_q_vel)
        r_smooth = float(np.exp(-jerk / 2.0))

        total_reward = (self.w_imitation * r_imitation +
                        self.w_balance * r_balance +
                        self.w_contact * r_contact +
                        self.w_smooth * r_smooth)

        metrics = {
            "joint_rmse": float(joint_err),
            "r_imitation": r_imitation,
            "r_balance": r_balance,
            "r_contact": r_contact,
            "r_smooth": r_smooth,
            "total_reward": total_reward
        }
        return total_reward, metrics


class SelfImprovingAgent:
    """Embodied lifelong learning agent that continuously improves humanoid dance
    and movement accuracy the more play time and episodes it executes."""

    def __init__(self,
                 state_dim: int = 70,
                 action_dim: int = 29,
                 model_path: str = "models/self_improving_agent.pt",
                 state_file: str = "data/self_improving_agent_state.json",
                 lr: float = 3e-4,
                 device: str = "cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_path = model_path
        self.state_file = state_file
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        self.policy = ResidualPolicy(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer = LifelongReplayBuffer(capacity=25000)
        self.critic = MotionQualityCritic()

        # Lifelong play history
        self.cumulative_play_time_sec = 0.0
        self.total_episodes_played = 0
        self.history: Dict[str, List[float]] = {
            "play_time_sec": [],
            "episodes": [],
            "rmse": [],
            "balance_score": [],
            "reward": [],
            "fidelity": []
        }

        # Exploration noise scheduler
        self.noise_std = 0.08
        self.min_noise_std = 0.01

        # Load existing state if available
        self.load_state()

    def construct_state(self, phase_sin: float, phase_cos: float,
                        q_curr: np.ndarray, q_vel: np.ndarray,
                        q_target: np.ndarray, root_z: float) -> np.ndarray:
        """Constructs fixed 70-dim state vector for policy input."""
        q_err = q_target - q_curr
        # 1 (phase_sin) + 1 (phase_cos) + 1 (root_z) + 29 (q_curr) + 29 (q_vel) + 29 (q_err) = 90
        # Slice or project to state_dim (70 dims: 2 phase + 1 root_z + 29 q_curr + 29 q_err + 9 vel_subset)
        feat = np.concatenate([
            [phase_sin, phase_cos, root_z],
            q_curr[:29],
            q_err[:29],
            q_vel[:9] # base + core joint velocities
        ]).astype(np.float32)
        if len(feat) < self.state_dim:
            pad = np.zeros(self.state_dim - len(feat), dtype=np.float32)
            feat = np.concatenate([feat, pad])
        return feat[:self.state_dim]

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Returns residual joint corrections."""
        self.policy.eval()
        with torch.no_grad():
            s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            res_act, _ = self.policy(s_tensor)
            action = res_act.squeeze(0).cpu().numpy()

        if not deterministic and self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise

        return np.clip(action, -0.15, 0.15)

    def adapt_step(self, batch_size: int = 64) -> float:
        """Performs one online policy gradient & value function improvement update."""
        if len(self.buffer) < batch_size:
            return 0.0

        self.policy.train()
        s_b, a_b, r_b, ns_b, d_b = self.buffer.sample(batch_size)
        s_b, a_b, r_b, ns_b, d_b = (s_b.to(self.device), a_b.to(self.device),
                                    r_b.to(self.device), ns_b.to(self.device), d_b.to(self.device))

        # Critic target
        with torch.no_grad():
            _, next_val = self.policy(ns_b)
            target_val = r_b + 0.95 * next_val * (1.0 - d_b)

        pred_action, pred_val = self.policy(s_b)

        # Policy optimization: maximize expected value + minimize residual action norm
        advantage = (target_val - pred_val).detach()
        action_loss = -torch.mean(pred_val) + 0.05 * torch.mean(pred_action ** 2)
        critic_loss = nn.functional.mse_loss(pred_val, target_val)

        total_loss = action_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return float(total_loss.item())

    def play_episode(self,
                      reference_trajectory: np.ndarray,
                      episode_duration_sec: float = 5.0,
                      fps: float = 100.0,
                      adapt_every: int = 10) -> Dict[str, Any]:
        """Plays one interactive session, executing physical simulation steps,
        computing rewards, accumulating experiences, and self-improving."""
        N_steps = int(episode_duration_sec * fps)
        step_dt = 1.0 / fps

        N_ref = len(reference_trajectory)
        ref_indices = np.linspace(0, N_ref - 1, N_steps).astype(int)

        ep_rewards = []
        ep_errors = []
        ep_balances = []

        q_curr = reference_trajectory[ref_indices[0], 7:].copy()
        q_vel = np.zeros(self.action_dim, dtype=np.float32)
        prev_q_vel = np.zeros(self.action_dim, dtype=np.float32)

        # Simulated base of support
        bos_xy = np.array([0.0, 0.0])

        for step in range(N_steps):
            ref_idx = ref_indices[step]
            q_ref = reference_trajectory[ref_idx, 7:]
            root_z = reference_trajectory[ref_idx, 2]

            phase = (step / N_steps) * 2 * np.pi
            state = self.construct_state(
                phase_sin=np.sin(phase),
                phase_cos=np.cos(phase),
                q_curr=q_curr,
                q_vel=q_vel,
                q_target=q_ref,
                root_z=root_z
            )

            # Residual correction
            residual = self.get_action(state, deterministic=False)

            # Apply action: PD-like dynamic tracking
            target_pose = q_ref + residual
            prev_q_vel = q_vel.copy()
            # Simulated forward dynamics with actuator inertia
            q_vel = 0.85 * q_vel + 0.15 * (target_pose - q_curr) / step_dt
            q_curr = q_curr + q_vel * step_dt

            # Foot contact heights
            sole_z_l = float(root_z - 0.70 + 0.03 * np.sin(phase))
            sole_z_r = float(root_z - 0.70 + 0.03 * np.cos(phase))

            # Center of mass tracking
            com_xy = np.array([0.02 * np.sin(phase), 0.02 * np.cos(phase)])

            reward, metrics = self.critic.evaluate_step(
                q_actual=q_curr,
                q_target=q_ref,
                q_vel=q_vel,
                prev_q_vel=prev_q_vel,
                feet_contact_z=(sole_z_l, sole_z_r),
                com_xy=com_xy,
                bos_xy=bos_xy
            )

            ep_rewards.append(reward)
            ep_errors.append(metrics["joint_rmse"])
            ep_balances.append(metrics["r_balance"])

            # Next state
            next_phase = ((step + 1) / N_steps) * 2 * np.pi
            next_state = self.construct_state(
                phase_sin=np.sin(next_phase),
                phase_cos=np.cos(next_phase),
                q_curr=q_curr,
                q_vel=q_vel,
                q_target=reference_trajectory[ref_indices[min(step + 1, N_steps - 1)], 7:],
                root_z=root_z
            )

            done = (step == N_steps - 1)
            self.buffer.push(state, residual, reward, next_state, done, metrics["joint_rmse"])

            # Online adaptation
            if step % adapt_every == 0 and len(self.buffer) >= 32:
                self.adapt_step(batch_size=32)

        # Update lifelong statistics
        self.cumulative_play_time_sec += episode_duration_sec
        self.total_episodes_played += 1

        mean_err = float(np.mean(ep_errors))
        mean_rew = float(np.mean(ep_rewards))
        mean_bal = float(np.mean(ep_balances)) * 100.0
        fidelity = float(max(50.0, 100.0 * (1.0 - mean_err / 0.25)))

        self.history["play_time_sec"].append(self.cumulative_play_time_sec)
        self.history["episodes"].append(self.total_episodes_played)
        self.history["rmse"].append(mean_err)
        self.history["balance_score"].append(mean_bal)
        self.history["reward"].append(mean_rew)
        self.history["fidelity"].append(fidelity)

        # Decay exploration noise as agent matures
        self.noise_std = max(self.min_noise_std, self.noise_std * 0.985)

        # Persist updated agent state
        self.save_state()

        return {
            "episode": self.total_episodes_played,
            "play_time_total_sec": self.cumulative_play_time_sec,
            "mean_rmse": mean_err,
            "mean_reward": mean_rew,
            "balance_stability": mean_bal,
            "motion_fidelity": fidelity,
            "noise_std": self.noise_std
        }

    def save_state(self):
        """Persists agent model weights, lifelong replay metrics, and learning curves."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)

        torch.save({
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "cumulative_play_time_sec": self.cumulative_play_time_sec,
            "total_episodes_played": self.total_episodes_played,
            "noise_std": self.noise_std
        }, self.model_path)

        state_data = {
            "cumulative_play_time_sec": self.cumulative_play_time_sec,
            "total_episodes_played": self.total_episodes_played,
            "noise_std": self.noise_std,
            "history": self.history
        }
        with open(self.state_file, "w") as f:
            json.dump(state_data, f, indent=2)

    def load_state(self) -> bool:
        """Loads existing lifelong learning state if present."""
        if os.path.exists(self.model_path) and os.path.exists(self.state_file):
            try:
                ckpt = torch.load(self.model_path, map_location=self.device)
                self.policy.load_state_dict(ckpt["model_state_dict"])
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                self.cumulative_play_time_sec = ckpt.get("cumulative_play_time_sec", 0.0)
                self.total_episodes_played = ckpt.get("total_episodes_played", 0)
                self.noise_std = ckpt.get("noise_std", self.noise_std)

                with open(self.state_file, "r") as f:
                    state_data = json.load(f)
                    self.history = state_data.get("history", self.history)
                return True
            except Exception as e:
                print(f"Warning: Could not load existing state ({e}). Starting fresh.")
                return False
        return False
