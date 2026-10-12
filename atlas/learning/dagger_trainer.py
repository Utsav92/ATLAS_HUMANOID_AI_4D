"""DAgger-Style (Dataset Aggregation) Interactive Corrective Training for Humanoid Policies.
Mitigates compounding error and distribution shift by collecting states visited during
MuJoCo physical simulation, labeling them with expert reference corrective actions,
and aggregating experience iteratively.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from atlas.learning.normalization import ObservationActionNormalizer
from atlas.simulation.manipulation_env import MuJoCoManipulationEnv


@dataclass
class DAggerIterationSummary:
    """Telemetry recorded per DAgger round."""
    iteration: int
    aggregated_dataset_size: int
    loss: float
    mean_drift_distance_m: float
    expert_corrective_magnitude: float
    simulation_success: bool


class DAggerTrainer:
    """Interactive Dataset Aggregation Trainer for Bounded-Drift Humanoid Motion."""

    def __init__(self,
                 policy: nn.Module,
                 normalizer: ObservationActionNormalizer,
                 reference_trajectory: np.ndarray, # (T, nq) reference expert trajectory
                 device: str = "cpu",
                 learning_rate: float = 5e-4):
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.normalizer = normalizer
        self.ref_traj = reference_trajectory
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.MSELoss()

        self.history_len = getattr(policy, "history_len", 1)
        self.history_buf = [np.zeros(self.normalizer.obs_dim, dtype=np.float32) for _ in range(self.history_len)]
        # Replay dataset
        self.dataset_states: List[np.ndarray] = []
        self.dataset_actions: List[np.ndarray] = []

    def seed_demonstrations(self, states: np.ndarray, actions: np.ndarray):
        """Seed replay buffer with initial clean demonstrations."""
        for s, a in zip(states, actions):
            norm_s = self.normalizer.normalize_observation(s)
            self.history_buf.pop(0)
            self.history_buf.append(norm_s)
            flat_hist = np.array(self.history_buf).flatten()
            self.dataset_states.append(flat_hist)
            self.dataset_actions.append(self.normalizer.normalize_action(a))

    def run_dagger_iteration(
        self,
        env: MuJoCoManipulationEnv,
        round_idx: int = 1,
        beta_expert: float = 0.5, # Probability of taking expert action during collection
        max_steps: int = 60,
        epochs_per_round: int = 5,
        batch_size: int = 32
    ) -> DAggerIterationSummary:
        """Execute one round of DAgger: Rollout -> Query Expert -> Aggregate -> Retrain."""
        env.reset()
        T_ref = len(self.ref_traj)

        collected_states = []
        collected_actions = []
        drift_distances = []

        curr_state = env.robot.get_state()
        state_vec = np.zeros(self.normalizer.obs_dim, dtype=np.float32)
        state_vec[:len(curr_state.qpos)] = curr_state.qpos[:self.normalizer.obs_dim]

        history_buf = [np.zeros(self.normalizer.obs_dim, dtype=np.float32) for _ in range(self.history_len)]

        for t in range(min(max_steps, T_ref)):
            norm_s = self.normalizer.normalize_observation(state_vec)
            history_buf.pop(0)
            history_buf.append(norm_s)
            flat_hist = np.array(history_buf).flatten()
            
            # Policy inference
            with torch.no_grad():
                s_t = torch.tensor(flat_hist, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_norm_a = self.policy(s_t).squeeze(0).cpu().numpy()

            pred_a = self.normalizer.denormalize_action(pred_norm_a)

            # Query Reference Expert Controller for corrective target at current state
            t_ref_idx = min(t, T_ref - 1)
            expert_target_q = self.ref_traj[t_ref_idx, 7:36] # 29 target joint angles
            
            # Proportional feedback correction towards reference
            current_q = curr_state.qpos[7:36]
            kp_correct = 1.2
            corrective_action = expert_target_q + kp_correct * (expert_target_q - current_q)
            norm_expert_a = self.normalizer.normalize_action(corrective_action)

            # Store (visited_state, expert_corrective_action)
            collected_states.append(flat_hist)
            collected_actions.append(norm_expert_a)

            # Execution action with beta mixing
            if np.random.rand() < beta_expert:
                exec_action = corrective_action
            else:
                exec_action = pred_a

            step_res = env.step(exec_action)
            curr_state = step_res.robot_state
            state_vec = np.zeros(self.normalizer.obs_dim, dtype=np.float32)
            state_vec[:len(curr_state.qpos)] = curr_state.qpos[:self.normalizer.obs_dim]

            # Measure drift distance
            drift = np.linalg.norm(curr_state.qpos[7:36] - expert_target_q)
            drift_distances.append(drift)

        # Aggregate new experience into replay buffer
        self.dataset_states.extend(collected_states)
        self.dataset_actions.extend(collected_actions)

        # Retrain policy on aggregated dataset
        all_s = torch.tensor(np.array(self.dataset_states), dtype=torch.float32)
        all_a = torch.tensor(np.array(self.dataset_actions), dtype=torch.float32)
        loader = DataLoader(TensorDataset(all_s, all_a), batch_size=batch_size, shuffle=True)

        final_loss = 0.0
        for _ in range(epochs_per_round):
            total_loss = 0.0
            count = 0
            for sb, ab in loader:
                sb, ab = sb.to(self.device), ab.to(self.device)
                self.optimizer.zero_grad()
                pred = self.policy(sb)
                loss = self.criterion(pred, ab)
                loss.backward()
                self.optimizer.step()
                total_loss += float(loss.item()) * len(sb)
                count += len(sb)
            final_loss = total_loss / max(1, count)

        return DAggerIterationSummary(
            iteration=round_idx,
            aggregated_dataset_size=len(self.dataset_states),
            loss=final_loss,
            mean_drift_distance_m=float(np.mean(drift_distances)),
            expert_corrective_magnitude=float(np.mean([np.linalg.norm(a) for a in collected_actions])),
            simulation_success=not env.robot.is_fallen()
        )
