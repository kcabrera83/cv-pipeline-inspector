import pytest
from conftest import client


SAMPLE_FEATURES = [[120.0, 35.0, 0.3, 4.5, 150.0, 90.0, 60.0, 0.55, 1800.0, 30.0, 110.0, 0.2, 45.0, 0.7]]


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert data["version"] == "1.0.0"


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "classifier" in data
    assert "severity_estimator" in data
    assert "anomaly_detector" in data
    assert "feature_names" in data
    assert "defect_types" in data


def test_api_docs(client):
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.json()
    assert data["openapi"] == "3.0.0"
    assert "/api/detect" in data["paths"]
    assert "/api/classify" in data["paths"]
    assert "/api/severity" in data["paths"]


def test_detect_valid(client):
    response = client.post("/api/detect", json={"features": SAMPLE_FEATURES})
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data
    assert data["total"] == 1
    detection = data["detections"][0]
    assert "is_anomaly" in detection
    assert "anomaly_score" in detection
    assert "label" in detection


def test_detect_missing_features(client):
    response = client.post("/api/detect", json={})
    assert response.status_code == 422


def test_classify_valid(client):
    response = client.post("/api/classify", json={"features": SAMPLE_FEATURES})
    assert response.status_code == 200
    data = response.json()
    assert "classifications" in data
    assert data["total"] == 1
    clf = data["classifications"][0]
    assert "defect_type" in clf
    assert "confidence" in clf


def test_classify_missing_features(client):
    response = client.post("/api/classify", json={})
    assert response.status_code == 422


def test_severity_valid(client):
    response = client.post("/api/severity", json={"features": SAMPLE_FEATURES})
    assert response.status_code == 200
    data = response.json()
    assert "severity" in data
    assert data["total"] == 1
    sev = data["severity"][0]
    assert "severity_score" in sev
    assert "severity_level" in sev
    assert sev["severity_level"] in ["low", "medium", "high", "critical"]


def test_severity_missing_features(client):
    response = client.post("/api/severity", json={})
    assert response.status_code == 422
