"""FastAPI application for CV Pipeline Inspector."""

import os
import sys
import json
import pickle
import numpy as np
from typing import Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cv_inspector.data_generator import DEFECT_TYPES, FEATURE_NAMES
from cv_inspector.utils.image_processor import ImageFeatureExtractor
from cv_inspector.models.defect_classifier import DefectClassifier
from cv_inspector.models.severity_estimator import SeverityEstimator
from cv_inspector.models.defect_detector import DefectDetector

app = FastAPI(
    title="CV Pipeline Inspector",
    description="Computer Vision Defect Detection for Oil & Gas pipelines",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


class FeaturesRequest(BaseModel):
    features: List[List[float]]


class DetectionItem(BaseModel):
    index: int
    is_anomaly: bool
    anomaly_score: float
    label: str


class DetectResponse(BaseModel):
    detections: List[DetectionItem]
    total: int


class ClassificationItem(BaseModel):
    defect_type: str
    confidence: float


class ClassifyResponse(BaseModel):
    classifications: List[ClassificationItem]
    total: int


class SeverityItem(BaseModel):
    severity_score: float
    severity_level: str


class SeverityResponse(BaseModel):
    severity: List[SeverityItem]
    total: int


@app.on_event("startup")
async def startup_event():
    _load_models()


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "classifier": classifier.is_trained,
            "severity_estimator": severity_est.is_trained,
            "anomaly_detector": detector.is_trained,
        },
        "version": "1.0.0",
        "defect_types": DEFECT_TYPES,
        "features": FEATURE_NAMES,
    }


@app.get("/api/models")
async def models_info():
    summary_path = os.path.join(MODEL_DIR, "training_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    return {
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
    }


@app.post("/api/detect", response_model=DetectResponse)
async def detect(request: FeaturesRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="Missing 'features' in request body")
    features = np.array(request.features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    predictions, scores = detector.detect(features)
    results = []
    for i, (pred, score) in enumerate(zip(predictions, scores)):
        results.append(DetectionItem(
            index=i,
            is_anomaly=int(pred) == -1,
            anomaly_score=float(score),
            label="anomalous" if int(pred) == -1 else "normal",
        ))
    return DetectResponse(detections=results, total=len(results))


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(request: FeaturesRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="Missing 'features' in request body")
    features = np.array(request.features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    labels, confidences = classifier.predict(features)
    results = []
    for label, conf in zip(labels, confidences):
        results.append(ClassificationItem(defect_type=label, confidence=float(conf)))
    return ClassifyResponse(classifications=results, total=len(results))


@app.post("/api/severity", response_model=SeverityResponse)
async def severity(request: FeaturesRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="Missing 'features' in request body")
    features = np.array(request.features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    scores = severity_est.predict(features)
    results = []
    for score in scores:
        level = "low" if score < 3 else "medium" if score < 6 else "high" if score < 8 else "critical"
        results.append(SeverityItem(severity_score=float(score), severity_level=level))
    return SeverityResponse(severity=results, total=len(results))


@app.get("/api/docs")
async def api_docs():
    return {
        "openapi": "3.0.0",
        "info": {"title": "CV Pipeline Inspector", "version": "1.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/detect": {"post": {"summary": "Detect anomalies in image features"}},
            "/api/classify": {"post": {"summary": "Classify defect type"}},
            "/api/severity": {"post": {"summary": "Estimate defect severity"}},
        }
    }


if __name__ == "__main__":
    import uvicorn
    _load_models()
    uvicorn.run(app, host="0.0.0.0", port=5016)
