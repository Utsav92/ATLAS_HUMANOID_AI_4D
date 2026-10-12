"""Run Multi-Scenario Stress Testing and Benchmark Across Varied Conditions.
Evaluates:
1. End-effector tracking error: Target Below 5 mm
2. Object trajectory error: Target Below 8 mm
3. Joint jerk reduction: Target Maintain >= 92%
4. Behavior cloning loss: Target Reduced Held-Out Validation Loss
5. Manipulation success rate: Target >= 95% Across Varied Scenarios
Reports Mean, 95th-Percentile (P95), and Worst-Case Errors.
"""

from __future__ import annotations
import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt

from atlas.simulation.scenario_testing import ScenarioBenchmarkSuite
from atlas.simulation.manipulation_env import MuJoCoManipulationEnv
from atlas.learning.bc_ppo_pipeline import EnhancedBCPlusPPOPipeline


def main():
    print("================================================================================")
    print("      ATLAS HUMANOID AI 4D : MULTI-SCENARIO STRESS TESTING BENCHMARK            ")
    print("================================================================================")
    print("Running stress tests across:")
    print("  * Object Weights   : 0.2 kg to 5.0 kg")
    print("  * Object Locations : Nominal, Offset Left/Right, Elevated, Lowered, Extended")
    print("  * Motion Speeds    : 0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x")
    print("  * Perturbations    : Lateral Torso Push (15N, 30N, 50N), Actuator Noise (5%, 10%)")
    print("  * Unseen Demos     : Salsa Spin, Arm Wave, Squat Bend, Quick Step, Dual Lift")
    print("--------------------------------------------------------------------------------\n")

    suite = ScenarioBenchmarkSuite()
    env = MuJoCoManipulationEnv()
    pipeline = EnhancedBCPlusPPOPipeline(state_dim=70, action_dim=29, history_len=8)

    print("Executing benchmark suite...")
    results = suite.run_benchmark(pipeline=pipeline, env=env)
    print(f"Completed {results.total_scenarios_tested} stress scenarios in {results.duration_seconds:.2f} seconds.\n")

    # Tabular Results Report
    print("=========================================================================================================")
    print("                           MULTI-SCENARIO BENCHMARK RESULTS SUMMARY                                      ")
    print("=========================================================================================================")
    print(f"{'Metric':<32} | {'Target':<18} | {'Mean':<12} | {'95th-Percentile':<15} | {'Worst-Case':<12} | {'Status'}")
    print("---------------------------------------------------------------------------------------------------------")
    
    ee = results.ee_error
    print(f"{'End-Effector Tracking Error':<32} | {ee.target:<18} | {ee.mean:6.2f} mm    | {ee.p95:6.2f} mm         | {ee.worst_case:6.2f} mm    | {ee.status}")

    obj = results.object_error
    print(f"{'Object Trajectory Error':<32} | {obj.target:<18} | {obj.mean:6.2f} mm    | {obj.p95:6.2f} mm         | {obj.worst_case:6.2f} mm    | {obj.status}")

    jerk = results.jerk_reduction
    print(f"{'Joint Jerk Reduction':<32} | {jerk.target:<18} | {jerk.mean:6.2f} %     | {jerk.p95:6.2f} %          | {jerk.worst_case:6.2f} %     | {jerk.status}")

    loss = results.bc_validation_loss
    print(f"{'Behavior Cloning Val Loss':<32} | {loss.target:<18} | {loss.mean:6.4f}       | {loss.p95:6.4f}          | {loss.worst_case:6.4f}       | {loss.status}")

    succ = results.manipulation_success_rate
    print(f"{'Manipulation Success Rate':<32} | {succ.target:<18} | {succ.mean:6.2f} %     | {succ.p95:6.2f} %          | {succ.worst_case:6.2f} %     | {succ.status}")
    print("=========================================================================================================\n")

    # Detailed Per-Condition Breakdown
    print("---------------------------------------------------------------------------------------------------------")
    print(f"{'Scenario Name':<28} | {'Demo Tested':<20} | {'EE (mm)':<8} | {'Obj (mm)':<8} | {'Jerk (%)':<8} | {'Succ (%)':<8}")
    print("---------------------------------------------------------------------------------------------------------")
    for r in results.condition_records:
        print(f"{r['scenario_name']:<28} | {r['demo_tested']:<20} | {r['ee_error_mm']:6.2f}   | {r['object_error_mm']:6.2f}   | {r['jerk_reduction_pct']:6.1f}   | {r['manipulation_success_rate']:6.1f}")
    print("---------------------------------------------------------------------------------------------------------\n")

    # -------------------------------------------------------------------------
    # Generate Multi-Panel Telemetry Chart (6 Panels)
    # -------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), facecolor="#0f1118")

    # Panel 1: End-Effector Tracking Error Distribution
    axs[0, 0].set_facecolor("#161a24")
    ee_vals = [r["ee_error_mm"] for r in results.condition_records]
    sc_names = [r["scenario_name"] for r in results.condition_records]
    x_pos = np.arange(len(ee_vals))
    axs[0, 0].bar(x_pos, ee_vals, color="#00e5ff", alpha=0.85, label="Mean EE Error (mm)")
    axs[0, 0].axhline(5.0, color="#ff0055", ls="--", lw=2, label="Engineering Target (5.0 mm)")
    axs[0, 0].axhline(ee.mean, color="#00ff99", ls=":", lw=1.5, label=f"Mean: {ee.mean:.2f} mm")
    axs[0, 0].set_title("1. End-Effector Error (< 5 mm)", color="#00e5ff", fontweight="bold")
    axs[0, 0].set_ylabel("Error (mm)", color="white")
    axs[0, 0].tick_params(colors="white")
    axs[0, 0].legend(loc="upper right", facecolor="#1f2430", edgecolor="none", labelcolor="white", fontsize=8)
    axs[0, 0].grid(True, color="#2b3245", alpha=0.4)

    # Panel 2: Object Trajectory Error Distribution
    axs[0, 1].set_facecolor("#161a24")
    obj_vals = [r["object_error_mm"] for r in results.condition_records]
    axs[0, 1].bar(x_pos, obj_vals, color="#ffaa00", alpha=0.85, label="Object Error (mm)")
    axs[0, 1].axhline(8.0, color="#ff0055", ls="--", lw=2, label="Engineering Target (8.0 mm)")
    axs[0, 1].axhline(obj.mean, color="#00ff99", ls=":", lw=1.5, label=f"Mean: {obj.mean:.2f} mm")
    axs[0, 1].set_title("2. Object Trajectory Error (< 8 mm)", color="#00e5ff", fontweight="bold")
    axs[0, 1].set_ylabel("Error (mm)", color="white")
    axs[0, 1].tick_params(colors="white")
    axs[0, 1].legend(loc="upper right", facecolor="#1f2430", edgecolor="none", labelcolor="white", fontsize=8)
    axs[0, 1].grid(True, color="#2b3245", alpha=0.4)

    # Panel 3: Joint Jerk Reduction Distribution
    axs[0, 2].set_facecolor("#161a24")
    jerk_vals = [r["jerk_reduction_pct"] for r in results.condition_records]
    axs[0, 2].bar(x_pos, jerk_vals, color="#b026ff", alpha=0.85, label="Jerk Reduction (%)")
    axs[0, 2].axhline(92.0, color="#ff0055", ls="--", lw=2, label="Target Bound (>= 92.0%)")
    axs[0, 2].set_ylim(85.0, 100.0)
    axs[0, 2].set_title("3. Joint Jerk Reduction (>= 92%)", color="#00e5ff", fontweight="bold")
    axs[0, 2].set_ylabel("Reduction (%)", color="white")
    axs[0, 2].tick_params(colors="white")
    axs[0, 2].legend(loc="lower right", facecolor="#1f2430", edgecolor="none", labelcolor="white", fontsize=8)
    axs[0, 2].grid(True, color="#2b3245", alpha=0.4)

    # Panel 4: Manipulation Success Rate Across Scenarios
    axs[1, 0].set_facecolor("#161a24")
    succ_vals = [r["manipulation_success_rate"] for r in results.condition_records]
    axs[1, 0].bar(x_pos, succ_vals, color="#00ff99", alpha=0.85, label="Success Rate (%)")
    axs[1, 0].axhline(95.0, color="#ff0055", ls="--", lw=2, label="Target Bound (>= 95.0%)")
    axs[1, 0].set_ylim(88.0, 102.0)
    axs[1, 0].set_title("4. Manipulation Success Rate (>= 95%)", color="#00e5ff", fontweight="bold")
    axs[1, 0].set_ylabel("Success Rate (%)", color="white")
    axs[1, 0].tick_params(colors="white")
    axs[1, 0].legend(loc="lower right", facecolor="#1f2430", edgecolor="none", labelcolor="white", fontsize=8)
    axs[1, 0].grid(True, color="#2b3245", alpha=0.4)

    # Panel 5: Held-Out Validation Loss Reduction
    axs[1, 1].set_facecolor("#161a24")
    loss_vals = [r["bc_val_loss"] for r in results.condition_records]
    axs[1, 1].plot(x_pos, loss_vals, marker="o", color="#00ffff", lw=2, label="Val Loss per Condition")
    axs[1, 1].axhline(0.873, color="#888888", ls="--", label="Baseline (0.873)")
    axs[1, 1].axhline(0.044, color="#ffaa00", ls=":", label="Ambitious (0.044)")
    axs[1, 1].axhline(loss.mean, color="#00ff99", ls="-", lw=1.5, label=f"Achieved Mean: {loss.mean:.4f}")
    axs[1, 1].set_title("5. BC Validation Loss Progression", color="#00e5ff", fontweight="bold")
    axs[1, 1].set_ylabel("Held-Out Validation Loss", color="white")
    axs[1, 1].tick_params(colors="white")
    axs[1, 1].legend(loc="upper right", facecolor="#1f2430", edgecolor="none", labelcolor="white", fontsize=8)
    axs[1, 1].grid(True, color="#2b3245", alpha=0.4)

    # Panel 6: Error Percentiles Comparison (Mean vs P95 vs Worst-Case)
    axs[1, 2].set_facecolor("#161a24")
    metrics_comp = ["EE Error (mm)", "Obj Error (mm)", "Val Loss (*100)"]
    means = [ee.mean, obj.mean, loss.mean * 100.0]
    p95s = [ee.p95, obj.p95, loss.p95 * 100.0]
    worsts = [ee.worst_case, obj.worst_case, loss.worst_case * 100.0]
    targets = [5.0, 8.0, 4.4]

    w_bar = 0.2
    idx_arr = np.arange(len(metrics_comp))
    axs[1, 2].bar(idx_arr - w_bar, means, width=w_bar, color="#00ff99", label="Mean")
    axs[1, 2].bar(idx_arr, p95s, width=w_bar, color="#ffaa00", label="95th-Percentile")
    axs[1, 2].bar(idx_arr + w_bar, worsts, width=w_bar, color="#ff0055", label="Worst-Case")
    axs[1, 2].scatter(idx_arr, targets, marker="D", color="#ffffff", s=60, zorder=5, label="Target Bound")
    axs[1, 2].set_xticks(idx_arr)
    axs[1, 2].set_xticklabels(metrics_comp, color="white", fontsize=9)
    axs[1, 2].set_title("6. Mean vs P95 vs Worst-Case", color="#00e5ff", fontweight="bold")
    axs[1, 2].tick_params(colors="white")
    axs[1, 2].legend(loc="upper left", facecolor="#1f2430", edgecolor="none", labelcolor="white", fontsize=8)
    axs[1, 2].grid(True, color="#2b3245", alpha=0.4)

    plt.suptitle("Atlas Humanoid AI 4D: Multi-Scenario Stress Testing Telemetry\n"
                 "(Varied Weights, Locations, Speeds, Perturbations & Unseen Demonstrations)",
                 color="white", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = "evaluation/scenario_stress_testing_metrics.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved stress test telemetry plot to {plot_path}")

    # Copy to artifacts directory
    artifact_dir = r"C:\Users\Utsav\.gemini\antigravity\brain\bf9fcd1b-d708-48b1-8610-01f91dc277ee"
    if os.path.exists(artifact_dir):
        dest_plot = os.path.join(artifact_dir, "scenario_stress_testing_metrics.png")
        shutil.copyfile(plot_path, dest_plot)
        print(f"Copied telemetry plot to artifact directory: {dest_plot}")


if __name__ == "__main__":
    main()
