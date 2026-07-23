# Architecture - CV Pipeline Inspector

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Web Dashboard (HTML)                   │
│                    Port 5016 /                           │
├─────────────────────────────────────────────────────────┤
│                    Flask API Layer                       │
│      /api/classify  /api/severity  /api/detect          │
├──────────┬──────────────────┬───────────────────────────┤
│ Defect   │ Severity         │ Anomaly                   │
│ Classif. │ Estimator        │ Detector                  │
│ (RF + GB)│ (GradientBoost)  │ (Isolation Forest)        │
├──────────┴──────────────────┴───────────────────────────┤
│           Image Feature Extractor (14 features)          │
│              Synthetic Data Generator                    │
└─────────────────────────────────────────────────────────┘
```

## Components

### Data Layer

- **Data Generator**: Creates synthetic pipeline defect samples with 14 image features
- **Defect Types**: corrosion, cracks, dents, leaks, weld_defects, healthy
- **Training Data**: 300 samples per class, 80/20 train/test split
- **Feature Vector**: 14-dimensional (pixel stats, edge, texture, color, gradients)

### Model Layer

#### Defect Classifier (`DefectClassifier`)
- **Algorithm**: Ensemble of RandomForest + GradientBoosting
- **RandomForest**: 100 estimators, max_depth=15, min_samples_split=5
- **GradientBoosting**: 100 estimators, max_depth=5, learning_rate=0.1
- **Voting**: Soft voting (average predicted probabilities)
- **Evaluation**: Cross-validation accuracy + test set accuracy
- **Persistence**: Pickle (.pkl)

#### Severity Estimator (`SeverityEstimator`)
- **Algorithm**: GradientBoosting Regressor
- **Parameters**: 100 estimators, max_depth=4, learning_rate=0.1
- **Output**: Continuous score 0-10
- **Levels**: low (<3), medium (3-6), high (6-8), critical (>8)
- **Evaluation**: MAE, R2 score, cross-validation R2
- **Persistence**: Pickle (.pkl)

#### Anomaly Detector (`DefectDetector`)
- **Algorithm**: Isolation Forest (unsupervised)
- **Parameters**: contamination=0.15, 100 estimators
- **Input**: Normal data only (unsupervised training)
- **Output**: Binary label (normal/anomaly) + anomaly score
- **Persistence**: Pickle (.pkl)

#### Image Feature Extractor (`ImageFeatureExtractor`)
- **Approach**: Simulated feature extraction from synthetic image data
- **Features**: 14 handcrafted features (intensity, edge, texture, color, gradient)

### API Layer

- **Framework**: Flask (Python)
- **Serialization**: JSON request/response
- **Model Loading**: Pickle deserialization at startup
- **Port**: 5016

### Dashboard Layer

- **Frontend**: HTML/CSS/JavaScript (single page)
- **Visualization**: Chart.js for metrics and results
- **Style**: Dark-themed responsive UI

## Data Flow

### Classification Pipeline

```
1. Input Feature Vector (14 dimensions)
   ↓
2. DefectClassifier
   ├── RandomForest Prediction
   └── GradientBoosting Prediction
   ↓
3. Soft Voting (Probability Averaging)
   ↓
4. Defect Type + Confidence
   ↓
5. Dashboard Display
```

### Severity Estimation Pipeline

```
1. Input Feature Vector (14 dimensions)
   ↓
2. SeverityEstimator (GradientBoosting)
   ↓
3. Continuous Score (0-10)
   ↓
4. Severity Level Mapping
   ↓
5. Score + Level Output
```

### Anomaly Detection Pipeline

```
1. Input Feature Vector(s)
   ↓
2. Isolation Forest
   ↓
3. Binary Prediction (-1=anomaly, 1=normal)
   ↓
4. Anomaly Score (lower = more anomalous)
   ↓
5. Label + Score Output
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| Web Framework | Flask |
| ML Library | scikit-learn |
| Numerical | NumPy |
| Model Persistence | Pickle |
| Frontend | HTML/CSS/JS + Chart.js |

## Model Artifacts

| File | Description |
|------|-------------|
| `outputs/models/defect_classifier.pkl` | Ensemble classifier (RF + GB) |
| `outputs/models/severity_estimator.pkl` | GradientBoosting severity regressor |
| `outputs/models/defect_detector.pkl` | Isolation Forest anomaly detector |
| `outputs/models/training_summary.json` | Training metrics and results |
