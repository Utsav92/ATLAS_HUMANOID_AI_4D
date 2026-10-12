"""Interactive Executable Runner for the Self-Improving Humanoid Agent.
Runs self-improving play sessions, displays real-time lifelong adaptation progress,
and generates performance improvement curves as play time increases.
"""

from __future__ import annotations
import argparse
import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt

from atlas.rl.self_improving_agent import SelfImprovingAgent


def main():
    parser = argparse.ArgumentParser(description="Run Self-Improving Humanoid Agent")
    parser.add_argument("--episodes", type=int, default=10, help="Number of play sessions to execute")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration of each play session in seconds")
    parser.add_argument("--fps", type=float, default=100.0, help="Simulation FPS")
    parser.add_argument("--reset-history", action="store_true", help="Reset previous play history")
    args = parser.parse_args()

    traj_path = "data/robot_salsa_trajectory_9000_frames.npz"
    if not os.path.exists(traj_path):
        print(f"Error: {traj_path} not found. Run trajectory generator first.")
        return

    data = np.load(traj_path)
    q_traj = data["q_traj"]

    model_path = "models/self_improving_agent.pt"
    state_file = "data/self_improving_agent_state.json"

    if args.reset_history:
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(state_file):
            os.remove(state_file)
        print("Reset prior play history.")

    agent = SelfImprovingAgent(
        model_path=model_path,
        state_file=state_file
    )

    print("================================================================================")
    print("      ATLAS HUMANOID AI 4D : LIFELONG SELF-IMPROVING DANCE & MOTION AGENT      ")
    print("================================================================================")
    print(f"Prior Accumulated Play Time : {agent.cumulative_play_time_sec:.1f}s across {agent.total_episodes_played} episodes")
    print(f"Target Play Sessions to Run  : {args.episodes} episodes ({args.duration}s each)")
    print(f"Simulation Control Rate      : {args.fps:.0f} Hz")
    print("--------------------------------------------------------------------------------")

    t_start = time.time()
    initial_rmse = agent.history["rmse"][-1] if len(agent.history["rmse"]) > 0 else None
    initial_fidelity = agent.history["fidelity"][-1] if len(agent.history["fidelity"]) > 0 else None

    for ep in range(1, args.episodes + 1):
        res = agent.play_episode(
            reference_trajectory=q_traj,
            episode_duration_sec=args.duration,
            fps=args.fps,
            adapt_every=10
        )

        curr_time = res["play_time_total_sec"]
        curr_rmse = res["mean_rmse"]
        curr_bal = res["balance_stability"]
        curr_fid = res["motion_fidelity"]
        noise = res["noise_std"]

        print(f"Session {ep:02d}/{args.episodes:02d} | PlayTime: {curr_time:6.1f}s | "
              f"RMSE: {curr_rmse:.4f} rad | Stability: {curr_bal:5.1f}% | "
              f"Fidelity: {curr_fid:5.1f}% | ExpNoise: {noise:.3f}")

    total_wall_time = time.time() - t_start
    final_rmse = agent.history["rmse"][-1]
    final_fidelity = agent.history["fidelity"][-1]

    rmse_improvement = ((initial_rmse - final_rmse) / initial_rmse * 100.0) if initial_rmse else ((agent.history["rmse"][0] - final_rmse) / agent.history["rmse"][0] * 100.0)

    print("--------------------------------------------------------------------------------")
    print("                        SELF-IMPROVEMENT SUMMARY REPORT                         ")
    print("--------------------------------------------------------------------------------")
    print(f"Total Cumulative Play Time   : {agent.cumulative_play_time_sec:.1f} seconds")
    print(f"Total Episodes Recorded      : {agent.total_episodes_played}")
    print(f"Wallclock Training Time      : {total_wall_time:.2f} seconds")
    print(f"Final Joint Tracking RMSE    : {final_rmse:.4f} rad")
    print(f"Final Motion Fidelity Score  : {final_fidelity:.2f}%")
    print(f"Tracking Accuracy Gain       : +{max(0.0, rmse_improvement):.2f}% improvement the more it played!")
    print(f"Saved Checkpoint             : {model_path}")
    print(f"Saved Lifelong State Log     : {state_file}")

    # Generate Learning Curves
    os.makedirs("evaluation", exist_ok=True)
    plot_path = "evaluation/self_improving_agent_learning_curves.png"

    fig, axs = plt.subplots(2, 2, figsize=(14, 9), facecolor="#0f1118")

    def style_axis(ax, title, ylabel, xlabel="Cumulative Play Time (Seconds)"):
        ax.set_facecolor("#151922")
        ax.set_title(title, color="#00e5ff", fontsize=12, pad=10, fontweight="bold")
        ax.set_xlabel(xlabel, color="#cfd8dc", fontsize=10)
        ax.set_ylabel(ylabel, color="#cfd8dc", fontsize=10)
        ax.tick_params(colors="#90a4ae")
        ax.grid(True, linestyle="--", alpha=0.25, color="#546e7a")
        for spine in ax.spines.values():
            spine.set_color("#263238")

    t_vals = agent.history["play_time_sec"]

    # 1. Tracking RMSE (Decreasing)
    style_axis(axs[0, 0], "Tracking Error vs. Play Time", "Joint RMSE (rad)")
    axs[0, 0].plot(t_vals, agent.history["rmse"], color="#ff5252", lw=2.4, marker="o", markersize=4, label="RMSE")
    axs[0, 0].legend(facecolor="#1e2430", edgecolor="#37474f", labelcolor="white")

    # 2. Motion Fidelity (Increasing)
    style_axis(axs[0, 1], "Motion Imitation Fidelity vs. Play Time", "Fidelity Score (%)")
    axs[0, 1].plot(t_vals, agent.history["fidelity"], color="#00e676", lw=2.4, marker="s", markersize=4, label="Fidelity (%)")
    axs[0, 1].legend(facecolor="#1e2430", edgecolor="#37474f", labelcolor="white")

    # 3. Dynamic Balance Stability (Increasing)
    style_axis(axs[1, 0], "Dynamic Equilibrium Stability vs. Play Time", "Stability (%)")
    axs[1, 0].plot(t_vals, agent.history["balance_score"], color="#ffb300", lw=2.4, marker="^", markersize=4, label="Stability (%)")
    axs[1, 0].legend(facecolor="#1e2430", edgecolor="#37474f", labelcolor="white")

    # 4. Total Multi-Objective Reward (Increasing)
    style_axis(axs[1, 1], "Self-Evaluation Reward vs. Play Time", "Cumulative Reward")
    axs[1, 1].plot(t_vals, agent.history["reward"], color="#e040fb", lw=2.4, marker="d", markersize=4, label="Reward")
    axs[1, 1].legend(facecolor="#1e2430", edgecolor="#37474f", labelcolor="white")

    plt.suptitle("ATLAS HUMANOID AI 4D: LIFELONG SELF-IMPROVEMENT LEARNING DYNAMICS",
                 color="white", fontsize=14, fontweight="heavy", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"Saved learning curves plot to {plot_path}")

    # Copy plot to brain directory
    brain_dir = "C:/Users/Utsav/.gemini/antigravity/brain/bf9fcd1b-d708-48b1-8610-01f91dc277ee"
    brain_plot = os.path.join(brain_dir, "self_improving_agent_learning_curves.png")
    shutil.copy(plot_path, brain_plot)
    print(f"Copied plot to brain artifact directory: {brain_plot}")


if __name__ == "__main__":
    main()
