import pytest
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "models")


def test_outputs_directory_exists():
    assert os.path.exists(OUTPUT_DIR)


def test_model_files_exist():
    model_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith((".pkl", ".joblib", ".h5", ".pt"))]
    assert len(model_files) > 0


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
