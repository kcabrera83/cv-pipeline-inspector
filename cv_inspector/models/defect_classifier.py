"""Defect type classifier using RandomForest and GradientBoosting."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os


class DefectClassifier:
    """Ensemble classifier combining RandomForest and GradientBoosting."""

    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.random_forest = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        self.gradient_boosting = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self.is_trained = False

    def train(self, X, y):
        """Train both classifiers and return cross-validation scores."""
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)

        rf_cv = cross_val_score(self.random_forest, X, y_encoded, cv=5, scoring="accuracy")
        gb_cv = cross_val_score(self.gradient_boosting, X, y_encoded, cv=5, scoring="accuracy")

        self.random_forest.fit(X, y_encoded)
        self.gradient_boosting.fit(X, y_encoded)
        self.is_trained = True

        rf_train_acc = accuracy_score(y_encoded, self.random_forest.predict(X))
        gb_train_acc = accuracy_score(y_encoded, self.gradient_boosting.predict(X))

        return {
            "random_forest_cv_mean": float(rf_cv.mean()),
            "random_forest_cv_std": float(rf_cv.std()),
            "gradient_boosting_cv_mean": float(gb_cv.mean()),
            "gradient_boosting_cv_std": float(gb_cv.std()),
            "random_forest_train_accuracy": float(rf_train_acc),
            "gradient_boosting_train_accuracy": float(gb_train_acc),
        }

    def predict(self, X):
        """Predict defect type using ensemble voting."""
        if not self.is_trained:
            raise RuntimeError("Model not trained yet.")

        X = np.array(X).reshape(1, -1) if X.ndim == 1 else np.array(X)
        rf_probs = self.random_forest.predict_proba(X)
        gb_probs = self.gradient_boosting.predict_proba(X)

        ensemble_probs = (rf_probs + gb_probs) / 2.0
        predictions = np.argmax(ensemble_probs, axis=1)
        labels = self.label_encoder.inverse_transform(predictions)
        confidences = np.max(ensemble_probs, axis=1)

        return labels, confidences

    def evaluate(self, X, y):
        """Evaluate on test data."""
        y_encoded = self.label_encoder.transform(y)
        labels, _ = self.predict(X)
        pred_encoded = self.label_encoder.transform(labels)
        return classification_report(y_encoded, pred_encoded, target_names=self.label_encoder.classes_, output_dict=True)

    def save(self, path):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "rf": self.random_forest,
                "gb": self.gradient_boosting,
                "encoder": self.label_encoder,
            }, f)

    def load(self, path):
        """Load a trained model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.random_forest = data["rf"]
        self.gradient_boosting = data["gb"]
        self.label_encoder = data["encoder"]
        self.is_trained = True
