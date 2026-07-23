"""Flask API for CV Pipeline Inspector."""

import os
import sys
import json
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cv_inspector.data_generator import DEFECT_TYPES, FEATURE_NAMES
from cv_inspector.utils.image_processor import ImageFeatureExtractor
from cv_inspector.models.defect_classifier import DefectClassifier
from cv_inspector.models.severity_estimator import SeverityEstimator
from cv_inspector.models.defect_detector import DefectDetector

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")
extractor = ImageFeatureExtractor()
classifier = DefectClassifier()
severity_est = SeverityEstimator()
detector = DefectDetector()


def _load_models():
    clf_path = os.path.join(MODEL_DIR, "defect_classifier.pkl")
    sev_path = os.path.join(MODEL_DIR, "severity_estimator.pkl")
    det_path = os.path.join(MODEL_DIR, "defect_detector.pkl")
    if os.path.exists(clf_path):
        classifier.load(clf_path)
    if os.path.exists(sev_path):
        severity_est.load(sev_path)
    if os.path.exists(det_path):
        detector.load(det_path)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "classifier": classifier.is_trained,
            "severity_estimator": severity_est.is_trained,
            "anomaly_detector": detector.is_trained,
        },
        "version": "1.0.0",
        "defect_types": DEFECT_TYPES,
        "features": FEATURE_NAMES,
    })


@app.route("/api/models", methods=["GET"])
def models_info():
    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    return jsonify({
        "classifier": {
            "type": "RandomForest + GradientBoosting (ensemble)",
            "is_trained": classifier.is_trained,
            "training_metrics": summary.get("classifier", {}),
        },
        "severity_estimator": {
            "type": "GradientBoosting Regressor",
            "is_trained": severity_est.is_trained,
            "training_metrics": summary.get("severity_estimator", {}),
        },
        "anomaly_detector": {
            "type": "Isolation Forest",
            "is_trained": detector.is_trained,
            "training_metrics": summary.get("anomaly_detector", {}),
        },
        "feature_names": FEATURE_NAMES,
        "defect_types": DEFECT_TYPES,
    })


@app.route("/api/detect", methods=["POST"])
def detect():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400
    features = np.array(data["features"])
    if features.ndim == 1:
        features = features.reshape(1, -1)
    predictions, scores = detector.detect(features)
    results = []
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        results.append({
            "index": i,
            "is_anomaly": int(pred) == -1,
            "anomaly_score": float(score),
            "label": "anomalous" if int(pred) == -1 else "normal",
        })
    return jsonify({"detections": results, "total": len(results)})


@app.route("/api/classify", methods=["POST"])
def classify():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400
    features = np.array(data["features"])
    if features.ndim == 1:
        features = features.reshape(1, -1)
    labels, confidences = classifier.predict(features)
    results = []
    for label, conf in zip(labels, confidences):
        results.append({"defect_type": label, "confidence": float(conf)})
    return jsonify({"classifications": results, "total": len(results)})


@app.route("/api/severity", methods=["POST"])
def severity():
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' in request body"}), 400
    features = np.array(data["features"])
    if features.ndim == 1:
        features = features.reshape(1, -1)
    scores = severity_est.predict(features)
    results = []
    for score in scores:
        level = "low" if score < 3 else "medium" if score < 6 else "high" if score < 8 else "critical"
        results.append({"severity_score": float(score), "severity_level": level})
    return jsonify({"severity": results, "total": len(results)})


@app.route("/api/docs")
def api_docs():
    return jsonify({
        "openapi": "3.0.0",
        "info": {"title": "CV Pipeline Inspector", "version": "1.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/detect": {"post": {"summary": "Detect anomalies in image features"}},
            "/api/classify": {"post": {"summary": "Classify defect type"}},
            "/api/severity": {"post": {"summary": "Estimate defect severity"}},
        }
    })


_load_models()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5016, debug=False)
