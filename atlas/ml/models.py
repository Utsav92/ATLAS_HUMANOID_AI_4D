"""Machine Learning Module for ATLAS HUMANOID AI 4D.
Implements scikit-learn models for contact classification, anomaly detection,
motion clustering, state dimensionality reduction, and failure prediction.
"""

from __future__ import annotations
import os
import pickle
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, silhouette_score


class ContactClassifier:
    """Classifies binary or multi-class contact states from joint torques and foot sensors."""
    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        if model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)
        elif model_type == "logistic":
            self.model = LogisticRegression(max_iter=200, random_state=42)
        elif model_type == "gb":
            self.model = GradientBoostingClassifier(n_estimators=30, max_depth=4, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self.model.fit(X, y)
        preds = self.model.predict(X)
        return {
            "f1": float(f1_score(y, preds, average="weighted", zero_division=0)),
            "precision": float(precision_score(y, preds, average="weighted", zero_division=0)),
            "recall": float(recall_score(y, preds, average="weighted", zero_division=0)),
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class SensorAnomalyDetector:
    """Detects sensor faults, torque spikes, and physical disturbances via Isolation Forest."""
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42, n_estimators=50)

    def train(self, X: np.ndarray):
        self.model.fit(X)

    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        """Returns True for anomalies (-1 in sklearn), False for normal (+1)."""
        preds = self.model.predict(X)
        return (preds == -1)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Normalized anomaly score: lower means more anomalous."""
        return self.model.score_samples(X)


class MotionClusterer:
    """Discovers gait regimes (standing, double-support, swing) via K-Means and projects via PCA."""
    def __init__(self, n_clusters: int = 3, n_components: int = 3):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.pca = PCA(n_components=n_components)

    def train(self, X: np.ndarray) -> Dict[str, float]:
        cluster_labels = self.kmeans.fit_predict(X)
        pca_proj = self.pca.fit_transform(X)
        sil = float(silhouette_score(X, cluster_labels)) if len(np.unique(cluster_labels)) > 1 else 0.0
        return {
            "inertia": float(self.kmeans.inertia_),
            "silhouette": sil,
            "pca_explained_variance_ratio": float(np.sum(self.pca.explained_variance_ratio_))
        }

    def predict_cluster(self, X: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(X)

    def project_pca(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)


class FailurePredictor:
    """Predicts impending robot falls or instability events within horizon tau."""
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=42)

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self.model.fit(X, y)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1] if len(np.unique(y)) > 1 else np.zeros(len(y))
        auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 1.0
        return {
            "f1": float(f1_score(y, preds, zero_division=0)),
            "auroc": auc,
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0))
        }

    def predict_fall_risk(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
