"""Unit tests for Milestone M3: Machine Learning & Deep Learning Baselines."""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from atlas.ml.models import ContactClassifier, SensorAnomalyDetector, MotionClusterer, FailurePredictor
from atlas.deep_learning.models import (
    CompactMLP, CompactCNN, CompactGRU, TemporalConvNet, StateAutoencoder, CompactTransformerEncoder
)
from atlas.deep_learning.trainer import CompactTrainer


def test_m3_ml_contact_classifier():
    np.random.seed(42)
    # 100 samples, 10 features (torques, foot force)
    X = np.random.randn(100, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

    clf = ContactClassifier(model_type="rf")
    metrics = clf.train(X, y)
    assert metrics["f1"] > 0.8
    preds = clf.predict(X[:5])
    assert preds.shape == (5,)


def test_m3_ml_anomaly_and_clustering():
    np.random.seed(42)
    X_normal = np.random.randn(80, 8).astype(np.float32)
    X_anomaly = np.random.randn(10, 8).astype(np.float32) * 5.0  # huge spikes
    X_combined = np.vstack([X_normal, X_anomaly])

    detector = SensorAnomalyDetector(contamination=0.1)
    detector.train(X_combined)
    anom_preds = detector.predict_anomalies(X_anomaly)
    assert np.sum(anom_preds) > 0  # Should detect anomalous points

    clusterer = MotionClusterer(n_clusters=3, n_components=2)
    clust_metrics = clusterer.train(X_normal)
    assert clust_metrics["pca_explained_variance_ratio"] > 0.3
    proj = clusterer.project_pca(X_normal)
    assert proj.shape == (80, 2)


def test_m3_ml_failure_predictor():
    np.random.seed(42)
    X = np.random.randn(100, 6).astype(np.float32)
    y = (X[:, 2] > 0.5).astype(np.int32)

    predictor = FailurePredictor()
    metrics = predictor.train(X, y)
    assert metrics["auroc"] > 0.7


def test_m3_dl_forward_passes():
    batch_size = 4
    
    # MLP
    mlp = CompactMLP(in_dim=50, out_dim=43)
    x_mlp = torch.randn(batch_size, 50)
    out_mlp = mlp(x_mlp)
    assert out_mlp.shape == (batch_size, 43)

    # CNN
    cnn = CompactCNN(in_channels=3, out_classes=4)
    x_cnn = torch.randn(batch_size, 3, 64, 64)
    out_cnn = cnn(x_cnn)
    assert out_cnn.shape == (batch_size, 43 if False else 4)

    # GRU
    gru = CompactGRU(in_dim=43, out_dim=43)
    x_gru = torch.randn(batch_size, 10, 43)
    out_gru = gru(x_gru)
    assert out_gru.shape == (batch_size, 43)

    # TCN
    tcn = TemporalConvNet(in_channels=43, out_channels=43)
    x_tcn = torch.randn(batch_size, 43, 16)
    out_tcn = tcn(x_tcn)
    assert out_tcn.shape == (batch_size, 43, 16)

    # Autoencoder
    ae = StateAutoencoder(state_dim=50, latent_dim=8)
    recon, latent = ae(x_mlp)
    assert recon.shape == (batch_size, 50)
    assert latent.shape == (batch_size, 8)

    # Transformer
    trans = CompactTransformerEncoder(out_dim=43)
    out_trans = trans(x_gru)
    assert out_trans.shape == (batch_size, 43)


def test_m3_dl_trainer_fit():
    torch.manual_seed(42)
    X = torch.randn(64, 50)
    Y = torch.randn(64, 43)
    dataset = TensorDataset(X, Y)
    train_loader = DataLoader(dataset, batch_size=16)
    val_loader = DataLoader(dataset, batch_size=16)

    model = CompactMLP(in_dim=50, out_dim=43)
    trainer = CompactTrainer(model, lr=1e-3)
    history = trainer.fit(train_loader, val_loader, loss_fn=nn.MSELoss(), epochs=3)

    assert len(history["train_loss"]) == 3
    assert history["train_loss"][-1] < history["train_loss"][0]
