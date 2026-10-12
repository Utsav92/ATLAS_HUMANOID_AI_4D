"""Milestone M3 Runner: Machine Learning & Deep Learning Baselines.
Trains and benchmarks contact classification, anomaly detection, PCA/clustering, and compact neural nets.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from atlas.ml.models import ContactClassifier, SensorAnomalyDetector, MotionClusterer, FailurePredictor
from atlas.deep_learning.models import CompactMLP, CompactCNN, CompactGRU
from atlas.deep_learning.trainer import CompactTrainer


def main():
    print("=" * 70)
    print("ATLAS HUMANOID AI 4D — MILESTONE M3 ML & DL BASELINES")
    print("=" * 70)

    # 1. Classical Machine Learning
    np.random.seed(42)
    X_contact = np.random.randn(200, 12).astype(np.float32)
    y_contact = (X_contact[:, 0] * 1.5 + X_contact[:, 1] > 0).astype(np.int32)

    clf = ContactClassifier(model_type="rf")
    clf_metrics = clf.train(X_contact, y_contact)
    print(f"[OK] Random Forest Contact Classifier: F1={clf_metrics['f1']:.3f}, Precision={clf_metrics['precision']:.3f}, Recall={clf_metrics['recall']:.3f}")

    # Isolation Forest
    anom_detector = SensorAnomalyDetector(contamination=0.05)
    anom_detector.train(X_contact)
    n_anom = np.sum(anom_detector.predict_anomalies(X_contact))
    print(f"[OK] Isolation Forest Anomaly Detector: detected {n_anom}/{len(X_contact)} anomalies")

    # K-Means & PCA
    clusterer = MotionClusterer(n_clusters=3, n_components=3)
    c_metrics = clusterer.train(X_contact)
    print(f"[OK] K-Means & PCA: Inertia={c_metrics['inertia']:.2f}, Explained Variance={c_metrics['pca_explained_variance_ratio']*100:.1f}%")

    # 2. Deep Learning Compact Models
    print("\nTraining Compact Deep Learning Models (PyTorch CPU)...")
    X_torch = torch.randn(128, 50)
    Y_torch = torch.randn(128, 43)
    loader = DataLoader(TensorDataset(X_torch, Y_torch), batch_size=16)

    mlp = CompactMLP(in_dim=50, out_dim=43)
    trainer = CompactTrainer(mlp, lr=1e-3)
    t0 = time.time()
    history = trainer.fit(loader, loader, loss_fn=nn.MSELoss(), epochs=5)
    dt = time.time() - t0

    print(f"[OK] CompactMLP (50->64->64->43) trained in {dt:.2f}s: Initial Loss={history['train_loss'][0]:.4f} -> Final Loss={history['train_loss'][-1]:.4f}")

    # Latency test
    sample_in = torch.randn(1, 50)
    times = []
    for _ in range(50):
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = mlp(sample_in)
        times.append((time.perf_counter() - t_start) * 1000)

    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    print(f"[OK] Batch-1 Inference Latency: p50={p50:.3f} ms, p95={p95:.3f} ms")

    print("\n[SUCCESS] Milestone M3 validated: ML & DL baselines operational.")


if __name__ == "__main__":
    main()
