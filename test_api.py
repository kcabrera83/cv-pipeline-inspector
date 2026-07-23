"""API tests for CV Pipeline Inspector."""

import os
import sys
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from app import app
from cv_inspector.data_generator import generate_samples


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health(self):
        resp = self.client.get("/api/health")
        data = resp.json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("classifier", data["models_loaded"])

    def test_models(self):
        resp = self.client.get("/api/models")
        data = resp.json()
        self.assertEqual(resp.status_code, 200)
        self.assertIn("classifier", data)
        self.assertIn("feature_names", data)
        self.assertEqual(len(data["feature_names"]), 14)

    def test_detect_with_features(self):
        X, _, _ = generate_samples(n_per_class=1, seed=99)
        payload = {"features": X[0].tolist()}
        resp = self.client.post("/api/detect", json=payload)
        data = resp.json()
        self.assertEqual(resp.status_code, 200)
        self.assertIn("detections", data)
        self.assertEqual(len(data["detections"]), 1)
        self.assertIn("is_anomaly", data["detections"][0])

    def test_detect_missing_features(self):
        resp = self.client.post("/api/detect", json={"features": []})
        self.assertEqual(resp.status_code, 400)

    def test_classify_with_features(self):
        X, _, _ = generate_samples(n_per_class=1, seed=77)
        payload = {"features": X[0].tolist()}
        resp = self.client.post("/api/classify", json=payload)
        data = resp.json()
        self.assertEqual(resp.status_code, 200)
        self.assertIn("classifications", data)
        self.assertIn("defect_type", data["classifications"][0])
        self.assertIn("confidence", data["classifications"][0])

    def test_classify_missing_features(self):
        resp = self.client.post("/api/classify", json={"features": []})
        self.assertEqual(resp.status_code, 400)

    def test_severity_with_features(self):
        X, _, _ = generate_samples(n_per_class=1, seed=55)
        payload = {"features": X[0].tolist()}
        resp = self.client.post("/api/severity", json=payload)
        data = resp.json()
        self.assertEqual(resp.status_code, 200)
        self.assertIn("severity", data)
        self.assertIn("severity_score", data["severity"][0])
        self.assertIn("severity_level", data["severity"][0])

    def test_severity_missing_features(self):
        resp = self.client.post("/api/severity", json={"features": []})
        self.assertEqual(resp.status_code, 400)

    def test_batch_classification(self):
        X, _, _ = generate_samples(n_per_class=2, seed=33)
        payload = {"features": X[:4].tolist()}
        resp = self.client.post("/api/classify", json=payload)
        data = resp.json()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(data["total"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
