"""Neural Network Deployment and ONNX Runtime CPU Inference for ATLAS HUMANOID AI 4D.
Handles ONNX model export, schema validation, batch-1 inference latency profiling, and safe fallback.
"""

from __future__ import annotations
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

from atlas.core.types import PolicyOutput, RobotState


def export_policy_to_onnx(
    model: nn.Module,
    onnx_path: str,
    input_dim: int = 50,
    dynamic_batch: bool = True
) -> str:
    """Export PyTorch policy model to ONNX format."""
    model.eval()
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    dynamic_axes = {"state": {0: "batch_size"}, "action": {0: "batch_size"}} if dynamic_batch else None

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["state"],
        output_names=["action"],
        dynamic_axes=dynamic_axes,
        dynamo=False
    )
    return onnx_path


class ONNXDeployablePolicy:
    """Production ONNX Runtime CPU inference engine with safety supervision fallback."""
    def __init__(self, onnx_path: str, input_dim: int = 50, action_dim: int = 43):
        self.onnx_path = onnx_path
        self.input_dim = input_dim
        self.action_dim = action_dim

        # Configure ONNX Runtime for CPU execution with 4 threads
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_action(
        self,
        state_vec: np.ndarray,
        nominal_safe_ctrl: np.ndarray
    ) -> PolicyOutput:
        """Run batch-one inference on CPU. Fall back to safe controller on anomaly or error."""
        try:
            if state_vec.ndim == 1:
                inp = state_vec.reshape(1, -1).astype(np.float32)
            else:
                inp = state_vec.astype(np.float32)

            # Input validation
            if inp.shape[1] != self.input_dim:
                raise ValueError(f"Expected input dim {self.input_dim}, got {inp.shape[1]}")

            ort_inputs = {self.input_name: inp}
            ort_outs = self.session.run([self.output_name], ort_inputs)
            action = ort_outs[0].squeeze(0)

            # Sanity check for NaN/Inf
            if not np.all(np.isfinite(action)):
                raise ValueError("Non-finite action detected from ONNX runtime")

            return PolicyOutput(
                action=action,
                policy_name="onnx_cpu_policy",
                safety_override=False
            )

        except Exception as e:
            return PolicyOutput(
                action=nominal_safe_ctrl.copy(),
                policy_name="onnx_fallback_safe",
                safety_override=True,
                details={"error": str(e)}
            )


def benchmark_onnx_latency(
    policy: ONNXDeployablePolicy,
    n_runs: int = 100
) -> Dict[str, float]:
    """Measure batch-one p50, p95, p99 inference latencies in milliseconds."""
    sample_state = np.random.randn(policy.input_dim).astype(np.float32)
    nominal = np.zeros(policy.action_dim, dtype=np.float32)

    # Warmup
    for _ in range(10):
        _ = policy.predict_action(sample_state, nominal)

    latencies_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = policy.predict_action(sample_state, nominal)
        dt = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt)

    return {
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "mean_ms": float(np.mean(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
        "min_ms": float(np.min(latencies_ms))
    }
