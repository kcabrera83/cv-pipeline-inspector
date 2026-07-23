"""FastAPI application for CV Pipeline Inspector."""

import os
import sys
import json
import cv2
import numpy as np
from typing import Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from ultralytics import YOLO
import albumentations as A

from cv_inspector.data_generator import DEFECT_TYPES

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
    try:
        models["yolo"] = YOLO('yolov8n.pt')
        print("  Loaded YOLOv8n model")
    except Exception as e:
        print(f"  Warning: Could not load YOLO model: {e}")
        models["yolo"] = None


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


class ImageRequest(BaseModel):
    width: int = 640
    height: int = 640


class DetectionItem(BaseModel):
    index: int
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]


class DetectResponse(BaseModel):
    detections: List[DetectionItem]
    total: int
    model: str


class FeatureItem(BaseModel):
    mean_intensity: float
    std_intensity: float
    edge_density: float
    texture_entropy: float


class FeaturesResponse(BaseModel):
    features: FeatureItem
    processing_time_ms: float


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
            "yolo": models.get("yolo") is not None,
        },
        "version": "2.0.0",
        "defect_types": DEFECT_TYPES,
    }


@app.get("/api/models")
async def models_info():
    return {
        "yolo": {
            "type": "YOLOv8n (Ultralytics)",
            "loaded": models.get("yolo") is not None,
        },
        "augmentation": {
            "type": "albumentations",
            "transforms": ["RandomBrightnessContrast", "GaussNoise", "MotionBlur", "Rotate", "Resize"],
        },
        "feature_extraction": {
            "type": "OpenCV",
            "features": ["mean_intensity", "std_intensity", "edge_density", "texture_entropy"],
        },
        "defect_types": DEFECT_TYPES,
    }


@app.post("/api/detect", response_model=DetectResponse)
async def detect(request: ImageRequest):
    if not models.get("yolo"):
        raise HTTPException(status_code=503, detail="YOLO model not loaded")

    dummy_image = np.random.randint(0, 255, (request.height, request.width, 3), dtype=np.uint8)
    results = models["yolo"](dummy_image, verbose=False)

    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                detections.append(DetectionItem(
                    index=i,
                    class_id=cls_id,
                    class_name=r.names.get(cls_id, "unknown"),
                    confidence=conf,
                    bbox=xyxy,
                ))

    return DetectResponse(detections=detections, total=len(detections), model="yolov8n")


@app.post("/api/features", response_model=FeaturesResponse)
async def extract_features(request: ImageRequest):
    t0 = __import__('time').time()
    dummy_image = np.random.randint(0, 255, (request.height, request.width, 3), dtype=np.uint8)
    features = extract_features_opencv(dummy_image)
    elapsed = __import__('time').time() - t0

    return FeaturesResponse(
        features=FeatureItem(**features),
        processing_time_ms=round(elapsed * 1000, 2),
    )


@app.post("/api/severity", response_model=SeverityResponse)
async def severity(request: ImageRequest):
    dummy_image = np.random.randint(0, 255, (request.height, request.width, 3), dtype=np.uint8)
    features = extract_features_opencv(dummy_image)

    score = min(10.0, features["edge_density"] * 10 + features["std_intensity"] / 25.5)
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
            "/api/detect": {"post": {"summary": "Detect defects with YOLOv8"}},
            "/api/features": {"post": {"summary": "Extract OpenCV features"}},
            "/api/severity": {"post": {"summary": "Estimate defect severity"}},
        }
    }


if __name__ == "__main__":
    import uvicorn
    _load_models()
    uvicorn.run(app, host="0.0.0.0", port=5016)
