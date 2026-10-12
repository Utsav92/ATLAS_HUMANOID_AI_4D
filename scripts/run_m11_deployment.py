"""Milestone M11 Runner: Neural Network Deployment & Latency Profiling.
Exports trained PyTorch policies to ONNX and benchmarks batch-one CPU inference latencies.
"""

import time
import os
import numpy as np
from atlas.deep_learning.models import CompactMLP
from atlas.deployment.onnx_exporter import (
    export_policy_to_onnx, ONNXDeployablePolicy, benchmark_onnx_latency
)


def main():
    print("=" * 70)
    print("ATLAS HUMANOID AI 4D — MILESTONE M11 ONNX DEPLOYMENT & PROFILING")
    print("=" * 70)

    onnx_path = r"C:\Users\Utsav\.gemini\antigravity\scratch\ATLAS_HUMANOID_AI_4D\models\onnx\g1_policy.onnx"
    print("Exporting PyTorch CompactMLP policy to ONNX format...")
    mlp = CompactMLP(in_dim=50, out_dim=43)
    export_policy_to_onnx(mlp, onnx_path, input_dim=50)
    print(f"[OK] Model exported: {onnx_path} ({os.path.getsize(onnx_path):,} bytes)")

    print("Initializing ONNX Runtime CPU Engine (4 threads, optimized graph)...")
    policy = ONNXDeployablePolicy(onnx_path, input_dim=50, action_dim=43)

    print("Benchmarking 100 batch-one inference cycles on CPU...")
    bench = benchmark_onnx_latency(policy, n_runs=100)

    print("\n--- Latency Profiling Results ---")
    print(f"  p50 Latency:  {bench['p50_ms']:.3f} ms")
    print(f"  p95 Latency:  {bench['p95_ms']:.3f} ms")
    print(f"  p99 Latency:  {bench['p99_ms']:.3f} ms")
    print(f"  Mean Latency: {bench['mean_ms']:.3f} ms")
    print(f"  Min Latency:  {bench['min_ms']:.3f} ms")
    print(f"  Max Latency:  {bench['max_ms']:.3f} ms")

    assert bench["p50_ms"] < 5.0, "p50 latency exceeds 5ms threshold!"
    print("\n[SUCCESS] Milestone M11 validated: ONNX Runtime deployment ready.")


if __name__ == "__main__":
    main()
