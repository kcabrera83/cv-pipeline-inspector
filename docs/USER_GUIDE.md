# User Guide - CV Pipeline Inspector

## Overview

CV Pipeline Inspector is a machine learning system for detecting, classifying, and estimating the severity of defects in pipeline infrastructure. It analyzes 14 image features per sample using ensemble models.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
cd cv-pipeline-inspector
pip install -r requirements.txt
pip install -e .
```

### Train Models

```bash
python train.py
```

Generates synthetic training data and trains all three models (classifier, severity estimator, anomaly detector). Models are saved to `outputs/models/`.

### Start the API Server

```bash
python app.py
```

The server starts on `http://localhost:5016`.

### Open the Dashboard

Navigate to `http://localhost:5016` in your browser.

### Run Tests

```bash
python test_api.py
```

## Dashboard Features

- **Defect Classification**: Upload feature vectors to classify defect types
- **Severity Estimation**: Get severity scores (0-10 scale) with severity levels
- **Anomaly Detection**: Run unsupervised anomaly detection on feature data
- **Model Statistics**: View training metrics for all models

## Defect Types

| Type | Description |
|------|-------------|
| corrosion | Surface degradation and rust formation |
| cracks | Structural fractures in pipeline material |
| dents | Physical deformations from impact |
| leaks | Fluid escape points in pipeline |
| weld_defects | Imperfections in welded joints |
| healthy | No defects detected |

## API Usage

### Using curl

**Classify a defect:**
```bash
curl -X POST http://localhost:5016/api/classify \
  -H "Content-Type: application/json" \
  -d '{"features": [120.5, 30.2, 0.25, 4.1, 140.0, 120.0, 100.0, 0.55, 1200.0, 30.0, 120.0, 0.1, 18.0, 0.25]}'
```

**Estimate severity:**
```bash
curl -X POST http://localhost:5016/api/severity \
  -H "Content-Type: application/json" \
  -d '{"features": [120.5, 30.2, 0.25, 4.1, 140.0, 120.0, 100.0, 0.55, 1200.0, 30.0, 120.0, 0.1, 18.0, 0.25]}'
```

**Detect anomalies:**
```bash
curl -X POST http://localhost:5016/api/detect \
  -H "Content-Type: application/json" \
  -d '{"features": [120.5, 30.2, 0.25, 4.1, 140.0, 120.0, 100.0, 0.55, 1200.0, 30.0, 120.0, 0.1, 18.0, 0.25]}'
```

### Using Python

```python
import requests

features = [120.5, 30.2, 0.25, 4.1, 140.0, 120.0, 100.0, 0.55, 1200.0, 30.0, 120.0, 0.1, 18.0, 0.25]

# Classify defect type
response = requests.post("http://localhost:5016/api/classify", json={"features": features})
result = response.json()
print(f"Defect: {result['classifications'][0]['defect_type']}")
print(f"Confidence: {result['classifications'][0]['confidence']:.2%}")

# Estimate severity
response = requests.post("http://localhost:5016/api/severity", json={"features": features})
result = response.json()
sev = result['severity'][0]
print(f"Severity: {sev['severity_score']:.1f}/10 ({sev['severity_level']})")

# Detect anomalies
response = requests.post("http://localhost:5016/api/detect", json={"features": features})
result = response.json()
det = result['detections'][0]
print(f"Anomaly: {det['label']} (score: {det['anomaly_score']:.4f})")
```

### Batch Processing

```python
import requests
import numpy as np

# Generate multiple feature vectors
batch_features = np.random.rand(10, 14).tolist()

response = requests.post("http://localhost:5016/api/classify", json={"features": batch_features})
for cls in response.json()["classifications"]:
    print(f"Defect: {cls['defect_type']}, Confidence: {cls['confidence']:.2%}")
```
