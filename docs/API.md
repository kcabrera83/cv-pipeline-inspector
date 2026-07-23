# API Documentation - CV Pipeline Inspector

## Base URL

```
http://localhost:5016
```

## Endpoints

### GET /

Serves the web interface (HTML dashboard).

**Response:** HTML page

---

### GET /api/health

Health check endpoint with model status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "classifier": true,
    "severity_estimator": true,
    "anomaly_detector": true
  },
  "version": "1.0.0",
  "defect_types": ["corrosion", "cracks", "dents", "leaks", "weld_defects", "healthy"],
  "features": [
    "pixel_mean", "pixel_std", "edge_density", "texture_entropy",
    "red_intensity", "green_intensity", "blue_intensity", "hog_energy",
    "laplacian_variance", "gradient_magnitude", "color_intensity",
    "edge_texture_ratio", "color_variance", "contrast_ratio"
  ]
}
```

---

### GET /api/models

Returns model information and training metrics.

**Response:**
```json
{
  "classifier": {
    "type": "RandomForest + GradientBoosting (ensemble)",
    "is_trained": true,
    "training_metrics": {
      "random_forest_cv_mean": 0.9450,
      "gradient_boosting_cv_mean": 0.9520,
      "classifier_test_accuracy": 0.9600
    }
  },
  "severity_estimator": {
    "type": "GradientBoosting Regressor",
    "is_trained": true,
    "training_metrics": {
      "cv_r2_mean": 0.9200,
      "train_mae": 0.3500,
      "test_mae": 0.4100
    }
  },
  "anomaly_detector": {
    "type": "Isolation Forest",
    "is_trained": true,
    "training_metrics": {
      "n_samples": 1800,
      "anomaly_ratio": 0.15
    }
  },
  "feature_names": [...],
  "defect_types": [...]
}
```

---

### POST /api/classify

Classify defect type from image features.

**Request:**
```json
{
  "features": [120.5, 30.2, 0.25, 4.1, 140.0, 120.0, 100.0, 0.55, 1200.0, 30.0, 120.0, 0.1, 18.0, 0.25]
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| features | array of float | Yes | 14-element feature vector |

**Response:**
```json
{
  "classifications": [
    {
      "defect_type": "corrosion",
      "confidence": 0.9234
    }
  ],
  "total": 1
}
```

**Error Response:**
```json
{"error": "Missing 'features' in request body"}
```
Status: `400 Bad Request`

---

### POST /api/severity

Estimate defect severity on a 0-10 scale.

**Request:**
```json
{
  "features": [120.5, 30.2, 0.25, 4.1, 140.0, 120.0, 100.0, 0.55, 1200.0, 30.0, 120.0, 0.1, 18.0, 0.25]
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| features | array of float | Yes | 14-element feature vector |

**Response:**
```json
{
  "severity": [
    {
      "severity_score": 6.5,
      "severity_level": "high"
    }
  ],
  "total": 1
}
```

Severity levels:
| Score Range | Level |
|-------------|-------|
| 0 - 2.99 | low |
| 3 - 5.99 | medium |
| 6 - 7.99 | high |
| 8 - 10.0 | critical |

**Error Response:**
```json
{"error": "Missing 'features' in request body"}
```
Status: `400 Bad Request`

---

### POST /api/detect

Detect anomalies in image feature vectors using Isolation Forest.

**Request:**
```json
{
  "features": [120.5, 30.2, 0.25, 4.1, 140.0, 120.0, 100.0, 0.55, 1200.0, 30.0, 120.0, 0.1, 18.0, 0.25]
}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| features | array of float | Yes | 14-element feature vector (or 2D array for batch) |

**Response:**
```json
{
  "detections": [
    {
      "index": 0,
      "is_anomaly": false,
      "anomaly_score": -0.2345,
      "label": "normal"
    }
  ],
  "total": 1
}
```

**Error Response:**
```json
{"error": "Missing 'features' in request body"}
```
Status: `400 Bad Request`

---

### GET /api/docs

Returns OpenAPI 3.0.0 specification.

**Response:**
```json
{
  "openapi": "3.0.0",
  "info": {"title": "CV Pipeline Inspector", "version": "1.0.0"},
  "paths": {
    "/api/health": {"get": {"summary": "Health check"}},
    "/api/models": {"get": {"summary": "Model info"}},
    "/api/detect": {"post": {"summary": "Detect anomalies in image features"}},
    "/api/classify": {"post": {"summary": "Classify defect type"}},
    "/api/severity": {"post": {"summary": "Estimate defect severity"}}
  }
}
```

## Feature Vector Format

The 14-element feature vector represents extracted image features:

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | pixel_mean | Mean pixel intensity |
| 1 | pixel_std | Standard deviation of pixel intensity |
| 2 | edge_density | Proportion of edge pixels |
| 3 | texture_entropy | Shannon entropy of texture |
| 4 | red_intensity | Red channel mean intensity |
| 5 | green_intensity | Green channel mean intensity |
| 6 | blue_intensity | Blue channel mean intensity |
| 7 | hog_energy | HOG descriptor energy |
| 8 | laplacian_variance | Laplacian filter variance |
| 9 | gradient_magnitude | Mean gradient magnitude |
| 10 | color_intensity | Overall color intensity |
| 11 | edge_texture_ratio | Edge-to-texture ratio |
| 12 | color_variance | Color channel variance |
| 13 | contrast_ratio | Image contrast ratio |

## Error Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request - missing or invalid parameters |
| 500 | Internal server error |
