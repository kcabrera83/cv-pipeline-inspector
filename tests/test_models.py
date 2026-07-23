import pytest
import os
import json
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_defect_classifier_loads():
    path = os.path.join(OUTPUT_DIR, "defect_classifier.pkl")
    if not os.path.exists(path):
        pytest.skip("defect_classifier.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_severity_estimator_loads():
    path = os.path.join(OUTPUT_DIR, "severity_estimator.pkl")
    if not os.path.exists(path):
        pytest.skip("severity_estimator.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_defect_detector_loads():
    path = os.path.join(OUTPUT_DIR, "defect_detector.pkl")
    if not os.path.exists(path):
        pytest.skip("defect_detector.pkl not found - run train.py first")
    assert os.path.getsize(path) > 0


def test_training_summary_exists():
    path = os.path.join(OUTPUT_DIR, "training_summary.json")
    if not os.path.exists(path):
        pytest.skip("training_summary.json not found")
    with open(path) as f:
        summary = json.load(f)
    assert "classifier" in summary
    assert "severity_estimator" in summary
    assert "anomaly_detector" in summary


def test_classifier_accuracy():
    path = os.path.join(OUTPUT_DIR, "training_summary.json")
    if not os.path.exists(path):
        pytest.skip("training_summary.json not found")
    with open(path) as f:
        summary = json.load(f)
    accuracy = summary.get("classifier_test_accuracy", 0)
    assert accuracy > 0.5, f"Classifier accuracy too low: {accuracy}"


def test_severity_estimator_metrics():
    path = os.path.join(OUTPUT_DIR, "training_summary.json")
    if not os.path.exists(path):
        pytest.skip("training_summary.json not found")
    with open(path) as f:
        summary = json.load(f)
    mae = summary.get("severity_test_mae", 999)
    assert mae < 5.0, f"Severity MAE too high: {mae}"


def test_anomaly_detector_metrics():
    path = os.path.join(OUTPUT_DIR, "training_summary.json")
    if not os.path.exists(path):
        pytest.skip("training_summary.json not found")
    with open(path) as f:
        summary = json.load(f)
    det = summary.get("anomaly_detector", {})
    assert det.get("n_samples", 0) > 0
    assert det.get("n_anomalies_detected", 0) > 0
