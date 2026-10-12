"""Unit tests for Milestone M11: Neural Network Deployment and ONNX Runtime CPU Inference."""

import pytest
import os
import numpy as np
import torch
from atlas.deep_learning.models import CompactMLP
from atlas.deployment.onnx_exporter import (
    export_policy_to_onnx, ONNXDeployablePolicy, benchmark_onnx_latency
)


def test_m11_onnx_export_and_inference(tmp_path):
    onnx_file = str(tmp_path / "test_policy.onnx")
    mlp = CompactMLP(in_dim=50, out_dim=43)
    
    # 1. Export
    export_policy_to_onnx(mlp, onnx_file, input_dim=50)
    assert os.path.exists(onnx_file)

    # 2. Deploy
    deployable = ONNXDeployablePolicy(onnx_file, input_dim=50, action_dim=43)
    test_state = np.random.randn(50).astype(np.float32)
    nominal = np.zeros(43, dtype=np.float32)

    out = deployable.predict_action(test_state, nominal)
    assert out.safety_override is False
    assert out.action.shape == (43,)

    # Compare with PyTorch output
    with torch.no_grad():
        pt_out = mlp(torch.tensor(test_state).unsqueeze(0)).squeeze(0).numpy()
    np.testing.assert_allclose(out.action, pt_out, atol=1e-4)


def test_m11_onnx_latency_and_fallback(tmp_path):
    onnx_file = str(tmp_path / "test_policy2.onnx")
    mlp = CompactMLP(in_dim=50, out_dim=43)
    export_policy_to_onnx(mlp, onnx_file, input_dim=50)

    deployable = ONNXDeployablePolicy(onnx_file, input_dim=50, action_dim=43)

    # Latency benchmarking
    bench = benchmark_onnx_latency(deployable, n_runs=50)
    assert bench["p50_ms"] < 5.0  # Fast CPU execution
    assert bench["p95_ms"] < 15.0

    # Test error fallback on invalid dimension
    bad_state = np.random.randn(20).astype(np.float32)  # Wrong dimension
    nominal = np.ones(43, dtype=np.float32) * 0.5
    out_fallback = deployable.predict_action(bad_state, nominal)

    assert out_fallback.safety_override is True
    assert np.allclose(out_fallback.action, nominal)
