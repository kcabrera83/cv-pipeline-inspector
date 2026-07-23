import pytest


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert data["version"] == "1.0.0"


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "classifier" in data
    assert "severity_estimator" in data
    assert "anomaly_detector" in data
    assert "feature_names" in data
    assert "defect_types" in data


def test_api_docs(client):
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.get_json()
    assert data["openapi"] == "3.0.0"
    assert "/api/detect" in data["paths"]
    assert "/api/classify" in data["paths"]
    assert "/api/severity" in data["paths"]


def test_detect_valid(client, sample_features):
    response = client.post("/api/detect", json={"features": sample_features})
    assert response.status_code == 200
    data = response.get_json()
    assert "detections" in data
    assert data["total"] == 1
    detection = data["detections"][0]
    assert "is_anomaly" in detection
    assert "anomaly_score" in detection
    assert "label" in detection


def test_detect_batch(client, sample_features):
    batch = [sample_features, sample_features]
    response = client.post("/api/detect", json={"features": batch})
    assert response.status_code == 200
    data = response.get_json()
    assert data["total"] == 2


def test_detect_missing_features(client):
    response = client.post("/api/detect", json={})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_classify_valid(client, sample_features):
    response = client.post("/api/classify", json={"features": sample_features})
    assert response.status_code == 200
    data = response.get_json()
    assert "classifications" in data
    assert data["total"] == 1
    clf = data["classifications"][0]
    assert "defect_type" in clf
    assert "confidence" in clf


def test_classify_missing_features(client):
    response = client.post("/api/classify", json={})
    assert response.status_code == 400


def test_severity_valid(client, sample_features):
    response = client.post("/api/severity", json={"features": sample_features})
    assert response.status_code == 200
    data = response.get_json()
    assert "severity" in data
    assert data["total"] == 1
    sev = data["severity"][0]
    assert "severity_score" in sev
    assert "severity_level" in sev
    assert sev["severity_level"] in ["low", "medium", "high", "critical"]


def test_severity_missing_features(client):
    response = client.post("/api/severity", json={})
    assert response.status_code == 400
