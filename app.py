"""FastAPI application for CV Pipeline Inspector."""

import os
import sys
import json
import base64
import cv2
import numpy as np
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

import albumentations as A

from cv_inspector.data_generator import DEFECT_TYPES, FEATURE_NAMES
from cv_inspector.models.defect_classifier import DefectClassifier
from cv_inspector.models.severity_estimator import SeverityEstimator

app = FastAPI(
    title="CV Pipeline Inspector",
    description="Computer Vision Defect Detection for Oil & Gas pipelines",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "models")

models = {}
augmentation = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.MotionBlur(blur_limit=7, p=0.2),
    A.Rotate(limit=15, p=0.3),
    A.Resize(640, 640)
])


def _load_models():
    classifier_path = os.path.join(MODEL_DIR, "defect_classifier.pkl")
    severity_path = os.path.join(MODEL_DIR, "severity_estimator.pkl")

    if os.path.exists(classifier_path):
        clf = DefectClassifier()
        clf.load(classifier_path)
        models["classifier"] = clf
        print("  Loaded DefectClassifier")
    else:
        print("  Warning: defect_classifier.pkl not found, run train.py first")

    if os.path.exists(severity_path):
        est = SeverityEstimator()
        est.load(severity_path)
        models["severity"] = est
        print("  Loaded SeverityEstimator")
    else:
        print("  Warning: severity_estimator.pkl not found, run train.py first")


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
    return {
        'mean_intensity': float(np.mean(gray)),
        'std_intensity': float(np.std(gray)),
        'edge_density': float(np.sum(edges > 0) / edges.size),
        'texture_entropy': float(entropy),
    }


class DetectRequest(BaseModel):
    features: List[float]


class DetectResponse(BaseModel):
    defect_type: str
    confidence: float
    all_probabilities: dict
    model: str


class FeatureItem(BaseModel):
    mean_intensity: float
    std_intensity: float
    edge_density: float
    texture_entropy: float


class FeaturesFromImageRequest(BaseModel):
    image_base64: str


class FeaturesFromFeaturesRequest(BaseModel):
    features: List[float]


class FeaturesResponse(BaseModel):
    features: dict
    source: str
    processing_time_ms: float


class SeverityRequest(BaseModel):
    features: List[float]


class SeverityItem(BaseModel):
    severity_score: float
    severity_level: str


class SeverityResponse(BaseModel):
    severity: List[SeverityItem]
    total: int


@app.on_event("startup")
async def startup_event():
    try:
        _load_models()
    except Exception as e:
        print(f"[WARN] Error loading models: {e}")


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": {
            "classifier": "classifier" in models,
            "severity_estimator": "severity" in models,
        },
        "version": "2.0.0",
        "defect_types": DEFECT_TYPES,
        "notes": "YOLO weights are COCO-pretrained (general objects); defect detection uses the trained classifier on extracted features.",
    }


@app.get("/api/models")
async def models_info():
    return {
        "classifier": {
            "type": "DefectClassifier (RandomForest + GradientBoosting ensemble)",
            "loaded": "classifier" in models,
            "defect_types": DEFECT_TYPES,
            "feature_names": FEATURE_NAMES,
            "input_dim": len(FEATURE_NAMES),
        },
        "severity_estimator": {
            "type": "SeverityEstimator (GradientBoosting regression)",
            "loaded": "severity" in models,
            "scale": "0-10",
        },
        "augmentation": {
            "type": "albumentations",
            "transforms": ["RandomBrightnessContrast", "GaussNoise", "MotionBlur", "Rotate", "Resize"],
        },
        "opencv_feature_extraction": {
            "type": "OpenCV",
            "features": ["mean_intensity", "std_intensity", "edge_density", "texture_entropy"],
            "note": "Used for image-level feature extraction from raw images",
        },
    }


@app.post("/api/detect", response_model=DetectResponse)
async def detect(request: DetectRequest):
    if "classifier" not in models:
        raise HTTPException(status_code=503, detail="Defect classifier not loaded. Run train.py first.")

    features = np.array(request.features).reshape(1, -1)
    if features.shape[1] != len(FEATURE_NAMES):
        raise HTTPException(status_code=400, detail=f"Expected {len(FEATURE_NAMES)} features, got {features.shape[1]}")

    labels, confidences = models["classifier"].predict(features)
    probs = {}
    rf_probs = models["classifier"].random_forest.predict_proba(features)[0]
    gb_probs = models["classifier"].gradient_boosting.predict_proba(features)[0]
    ensemble_probs = (rf_probs + gb_probs) / 2.0
    for i, cls in enumerate(models["classifier"].label_encoder.classes_):
        probs[cls] = round(float(ensemble_probs[i]), 4)

    return DetectResponse(
        defect_type=str(labels[0]),
        confidence=round(float(confidences[0]), 4),
        all_probabilities=probs,
        model="ensemble(classifier)",
    )


@app.post("/api/features", response_model=FeaturesResponse)
async def extract_features(request: FeaturesFromImageRequest):
    t0 = __import__('time').time()

    try:
        image_bytes = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
        features = extract_features_opencv(image)
        source = "opencv_from_image"
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    elapsed = __import__('time').time() - t0
    return FeaturesResponse(
        features=features,
        source=source,
        processing_time_ms=round(elapsed * 1000, 2),
    )


@app.post("/api/severity", response_model=SeverityResponse)
async def severity(request: SeverityRequest):
    if "severity" not in models:
        raise HTTPException(status_code=503, detail="Severity estimator not loaded. Run train.py first.")

    features = np.array(request.features).reshape(1, -1)
    if features.shape[1] != len(FEATURE_NAMES):
        raise HTTPException(status_code=400, detail=f"Expected {len(FEATURE_NAMES)} features, got {features.shape[1]}")

    score = float(models["severity"].predict(features)[0])
    level = "low" if score < 3 else "medium" if score < 6 else "high" if score < 8 else "critical"

    return SeverityResponse(
        severity=[SeverityItem(severity_score=round(score, 2), severity_level=level)],
        total=1,
    )


@app.get("/api/docs")
async def api_docs():
    return {
        "openapi": "3.0.0",
        "info": {"title": "CV Pipeline Inspector", "version": "2.0.0"},
        "paths": {
            "/api/health": {"get": {"summary": "Health check"}},
            "/api/models": {"get": {"summary": "Model info"}},
            "/api/detect": {"post": {"summary": "Classify defect type from feature vector"}},
            "/api/features": {"post": {"summary": "Extract OpenCV features from base64 image"}},
            "/api/severity": {"post": {"summary": "Estimate defect severity from feature vector"}},
        }
    }


if __name__ == "__main__":
    import uvicorn
    _load_models()
    uvicorn.run(app, host="0.0.0.0", port=5016)
