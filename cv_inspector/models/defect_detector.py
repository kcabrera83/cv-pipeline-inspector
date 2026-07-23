"""Anomaly detector using Isolation Forest for pipeline images."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os


class DefectDetector:
    """Isolation Forest based anomaly detector for pipeline inspection."""

    def __init__(self, contamination=0.3):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
            n_jobs=-1,
        )
        self.is_trained = False

    def train(self, X):
        """Train the anomaly detector on feature data."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True

        scores = self.model.score_samples(X_scaled)
        predictions = self.model.predict(X_scaled)
        n_anomalies = int(np.sum(predictions == -1))

        return {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "n_anomalies_detected": n_anomalies,
            "anomaly_ratio": float(n_anomalies / X.shape[0]),
            "mean_anomaly_score": float(np.mean(scores)),
            "std_anomaly_score": float(np.std(scores)),
        }

    def detect(self, X):
        """Detect anomalies. Returns -1 for anomaly, 1 for normal."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        X = np.array(X).reshape(1, -1) if X.ndim == 1 else np.array(X)
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        return predictions, scores

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.is_trained = True
