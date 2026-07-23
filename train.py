"""Train all models and save them to outputs/models/."""

import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import albumentations as A

from cv_inspector.data_generator import generate_samples, DEFECT_TYPES, FEATURE_NAMES
from cv_inspector.models.defect_classifier import DefectClassifier
from cv_inspector.models.severity_estimator import SeverityEstimator

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")


def build_augmentation_pipeline():
    """Build albumentations augmentation pipeline"""
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.MotionBlur(blur_limit=7, p=0.2),
        A.Rotate(limit=15, p=0.3),
        A.Resize(640, 640)
    ])
    return transform


def extract_features_opencv(image):
    """Extract image features using OpenCV"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, 100, 200)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log2(hist + 1e-10))

    features = {
        'mean_intensity': float(np.mean(gray)),
        'std_intensity': float(np.std(gray)),
        'edge_density': float(np.sum(edges > 0) / edges.size),
        'texture_entropy': float(entropy),
    }
    return features


def main():
    print("=" * 70)
    print("  CV Pipeline Inspector - Model Training Pipeline")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Generate synthetic feature data
    print("\n[1/5] Generating synthetic dataset...")
    X, y, severities = generate_samples(n_per_class=300, seed=42)
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Features per sample: {X.shape[1]}")
    print(f"  Defect classes: {', '.join(DEFECT_TYPES)}")

    # Step 2: Build augmentation pipeline
    print("\n[2/5] Building albumentations augmentation pipeline...")
    transform = build_augmentation_pipeline()
    print("  Augmentations: RandomBrightnessContrast, GaussNoise, MotionBlur, Rotate, Resize")

    # Step 3: Train defect classifier
    print("\n[3/5] Training defect classifier (RandomForest + GradientBoosting)...")
    classifier = DefectClassifier()
    clf_metrics = classifier.train(X, y)
    print(f"  RandomForest CV accuracy: {clf_metrics['random_forest_cv_mean']:.4f} (+/- {clf_metrics['random_forest_cv_std']:.4f})")
    print(f"  GradientBoosting CV accuracy: {clf_metrics['gradient_boosting_cv_mean']:.4f} (+/- {clf_metrics['gradient_boosting_cv_std']:.4f})")
    print(f"  RandomForest train accuracy: {clf_metrics['random_forest_train_accuracy']:.4f}")
    print(f"  GradientBoosting train accuracy: {clf_metrics['gradient_boosting_train_accuracy']:.4f}")
    classifier_path = os.path.join(OUTPUT_DIR, "defect_classifier.pkl")
    classifier.save(classifier_path)
    print(f"  Classifier saved to {classifier_path}")

    # Step 4: Train severity estimator
    print("\n[4/5] Training severity estimator (GradientBoosting regression)...")
    severity_est = SeverityEstimator()
    sev_metrics = severity_est.train(X, severities)
    print(f"  CV R2: {sev_metrics['cv_r2_mean']:.4f} (+/- {sev_metrics['cv_r2_std']:.4f})")
    print(f"  Train MAE: {sev_metrics['train_mae']:.4f}")
    print(f"  Train R2: {sev_metrics['train_r2']:.4f}")
    severity_path = os.path.join(OUTPUT_DIR, "severity_estimator.pkl")
    severity_est.save(severity_path)
    print(f"  Severity estimator saved to {severity_path}")

    # Step 5: Feature extraction demo with OpenCV
    print("\n[5/5] Feature extraction demo with OpenCV...")
    sample_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    features = extract_features_opencv(sample_image)
    print(f"  Sample feature extraction:")
    for k, v in features.items():
        print(f"    {k}: {v:.4f}")

    # Save training summary
    summary = {
        "num_classes": len(DEFECT_TYPES),
        "defect_types": DEFECT_TYPES,
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "n_samples": X.shape[0],
        "augmentation": ["RandomBrightnessContrast", "GaussNoise", "MotionBlur", "Rotate", "Resize"],
        "opencv_features": list(features.keys()),
        "classifier_metrics": clf_metrics,
        "severity_metrics": sev_metrics,
    }
    summary_path = os.path.join(OUTPUT_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("  Training Summary")
    print("=" * 70)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("classifier_metrics", "severity_metrics")}, indent=2))
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print("Training complete.")


if __name__ == "__main__":
    main()
