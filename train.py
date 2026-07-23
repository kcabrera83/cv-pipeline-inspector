"""Train all models and save them to outputs/models/."""

import os
import sys
import json
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cv_inspector.data_generator import generate_samples, DEFECT_TYPES
from cv_inspector.models.defect_classifier import DefectClassifier
from cv_inspector.models.severity_estimator import SeverityEstimator
from cv_inspector.models.defect_detector import DefectDetector

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")


def main():
    print("=" * 70)
    print("  CV Pipeline Inspector - Model Training Pipeline")
    print("=" * 70)

    # Generate data
    print("\n[1/5] Generating synthetic dataset...")
    X, y, severities = generate_samples(n_per_class=300, seed=42)
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Features per sample: {X.shape[1]}")
    print(f"  Defect classes: {', '.join(DEFECT_TYPES)}")

    # Split data
    print("\n[2/5] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test, sev_train, sev_test = train_test_split(
        X, y, severities, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # Train classifier
    print("\n[3/5] Training Defect Classifier (RandomForest + GradientBoosting)...")
    classifier = DefectClassifier()
    clf_metrics = classifier.train(X_train, y_train)
    print(f"  RF  CV Accuracy: {clf_metrics['random_forest_cv_mean']:.4f} (+/- {clf_metrics['random_forest_cv_std']:.4f})")
    print(f"  GB  CV Accuracy: {clf_metrics['gradient_boosting_cv_mean']:.4f} (+/- {clf_metrics['gradient_boosting_cv_std']:.4f})")
    print(f"  RF  Train Acc:   {clf_metrics['random_forest_train_accuracy']:.4f}")
    print(f"  GB  Train Acc:   {clf_metrics['gradient_boosting_train_accuracy']:.4f}")

    clf_report = classifier.evaluate(X_test, y_test)
    print(f"  Test Accuracy (ensemble): {clf_report['accuracy']:.4f}")
    classifier.save(os.path.join(OUTPUT_DIR, "defect_classifier.pkl"))

    # Train severity estimator
    print("\n[4/5] Training Severity Estimator (GradientBoosting)...")
    severity_est = SeverityEstimator()
    sev_metrics = severity_est.train(X_train, sev_train)
    print(f"  CV R2: {sev_metrics['cv_r2_mean']:.4f} (+/- {sev_metrics['cv_r2_std']:.4f})")
    print(f"  Train MAE: {sev_metrics['train_mae']:.4f}")
    print(f"  Train R2:  {sev_metrics['train_r2']:.4f}")

    sev_eval = severity_est.evaluate(X_test, sev_test)
    print(f"  Test MAE: {sev_eval['mae']:.4f}")
    print(f"  Test R2:  {sev_eval['r2']:.4f}")
    severity_est.save(os.path.join(OUTPUT_DIR, "severity_estimator.pkl"))

    # Train anomaly detector
    print("\n[5/5] Training Anomaly Detector (Isolation Forest)...")
    detector = DefectDetector(contamination=0.15)
    det_metrics = detector.train(X)
    print(f"  Samples: {det_metrics['n_samples']}")
    print(f"  Anomalies detected: {det_metrics['n_anomalies_detected']} ({det_metrics['anomaly_ratio']:.2%})")
    print(f"  Mean score: {det_metrics['mean_anomaly_score']:.4f}")
    detector.save(os.path.join(OUTPUT_DIR, "defect_detector.pkl"))

    # Summary
    print("\n" + "=" * 70)
    print("  Training Summary")
    print("=" * 70)
    summary = {
        "classifier": clf_metrics,
        "classifier_test_accuracy": clf_report["accuracy"],
        "severity_estimator": sev_metrics,
        "severity_test_mae": sev_eval["mae"],
        "anomaly_detector": det_metrics,
    }
    print(json.dumps(summary, indent=2))

    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print("Training complete.")


if __name__ == "__main__":
    main()
