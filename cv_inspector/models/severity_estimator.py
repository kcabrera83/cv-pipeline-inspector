"""Defect severity estimator using GradientBoosting regression."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os


class SeverityEstimator:
    """Estimates defect severity on a 0-10 scale."""

    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )
        self.is_trained = False

    def train(self, X, y):
        """Train the severity estimator."""
        self.model.fit(X, y)
        self.is_trained = True

        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="r2")
        y_pred = self.model.predict(X)

        return {
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "train_mse": float(mean_squared_error(y, y_pred)),
            "train_mae": float(mean_absolute_error(y, y_pred)),
            "train_r2": float(r2_score(y, y_pred)),
        }

    def predict(self, X):
        """Predict severity score clamped to [0, 10]."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")
        X = np.array(X).reshape(1, -1) if X.ndim == 1 else np.array(X)
        preds = self.model.predict(X)
        return np.clip(preds, 0, 10)

    def evaluate(self, X, y):
        """Evaluate on test data."""
        y_pred = self.predict(X)
        return {
            "mse": float(mean_squared_error(y, y_pred)),
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred)),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
