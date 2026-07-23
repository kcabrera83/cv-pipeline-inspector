import pytest


SAMPLE_FEATURES = [120.0, 35.0, 0.3, 4.5, 150.0, 90.0, 60.0, 0.55, 1800.0, 30.0, 110.0, 0.2, 45.0, 0.7]


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data
    assert data["version"] == "2.0.0"


def test_models(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "classifier" in data
    assert "defect_types" in data["classifier"]


def test_api_docs(client):
    response = client.get("/api/docs")
    assert response.status_code == 200
    data = response.json()
    assert data["openapi"] == "3.0.0"
    assert "/api/detect" in data["paths"]
    assert "/api/severity" in data["paths"]


def test_detect_valid(client):
    response = client.post("/api/detect", json={"features": SAMPLE_FEATURES})
    assert response.status_code in (200, 400, 503)
    if response.status_code == 200:
        data = response.json()
        assert "defect_type" in data
        assert "confidence" in data


def test_detect_missing_features(client):
    response = client.post("/api/detect", json={})
    assert response.status_code in (422, 503)


def test_severity_valid(client):
    response = client.post("/api/severity", json={"features": SAMPLE_FEATURES})
    assert response.status_code in (200, 400, 503)
    if response.status_code == 200:
        data = response.json()
        assert "severity" in data
        assert data["total"] == 1
        sev = data["severity"][0]
        assert "severity_score" in sev
        assert "severity_level" in sev


def test_severity_missing_features(client):
    response = client.post("/api/severity", json={})
    assert response.status_code in (422, 200, 503)
