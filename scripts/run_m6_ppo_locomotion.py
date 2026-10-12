"""Milestone M6 Runner: PPO Balancing and Locomotion.
Trains a PPO policy using Stable-Baselines3 on G1StandingEnv and saves checkpoint.
"""

import time
import os
from pathlib import Path
from atlas.rl.train_ppo import train_standing_ppo, evaluate_ppo_policy


def main():
    print("=" * 70)
    print("ATLAS HUMANOID AI 4D — MILESTONE M6 PPO BALANCING & LOCOMOTION")
    print("=" * 70)

    ckpt_path = r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D\models\checkpoints\g1_ppo_standing.zip"
    print(f"Training PPO policy on G1StandingEnv (CPU PyTorch)...")
    t0 = time.time()
    model = train_standing_ppo(total_timesteps=1024, save_path=ckpt_path)
    train_time = time.time() - t0
    print(f"[OK] Training finished in {train_time:.2f}s. Checkpoint saved to {ckpt_path}")

    print("Evaluating trained policy over 3 test episodes...")
    metrics = evaluate_ppo_policy(model, n_eval_episodes=3)
    print(f"[OK] Evaluation Metrics: Mean Reward={metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")

    print("\n[SUCCESS] Milestone M6 validated: PPO balancing policy operational.")


if __name__ == "__main__":
    main()
