# CV Pipeline Inspector

Computer Vision system for pipeline defect detection using image analysis and machine learning.

## Overview

CV Pipeline Inspector is a machine learning-based system designed to detect, classify, and estimate the severity of defects in pipeline infrastructure. The system uses simulated image feature extraction and ensemble machine learning models to analyze pipeline conditions.

### Defect Types

- **Corrosion** - Surface degradation and rust formation
- **Cracks** - Structural fractures in pipeline material
- **Dents** - Physical deformations from impact
- **Leaks** - Fluid escape points in pipeline
- **Weld Defects** - Imperfections in welded joints
- **Healthy** - No defects detected

### Features Extracted

The system analyzes 14 image features per sample:

1. Pixel mean intensity
2. Pixel standard deviation
3. Edge density
4. Texture entropy (Shannon entropy)
5. Red channel intensity
6. Green channel intensity
7. Blue channel intensity
8. HOG energy
9. Laplacian variance
10. Gradient magnitude
11. Color intensity
12. Edge-to-texture ratio
13. Color variance
14. Contrast ratio

### Machine Learning Models

- **Defect Classifier**: Ensemble of RandomForest and GradientBoosting classifiers for defect type identification
- **Severity Estimator**: GradientBoosting regressor for defect severity on a 0-10 scale
- **Anomaly Detector**: Isolation Forest for unsupervised anomaly detection in pipeline images

## Installation

```bash
git clone https://github.com/username/cv-pipeline-inspector.git
cd cv-pipeline-inspector
pip install -r requirements.txt
pip install -e .
```

## Training

```bash
python train.py
```

This generates synthetic training data and trains all three models. Trained models are saved to `outputs/models/`.

## Running the API

```bash
python app.py
```

The API runs on port 5016.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web interface |
| `/api/health` | GET | Health check and model status |
| `/api/models` | GET | Model information and training metrics |
| `/api/detect` | POST | Anomaly detection on feature vectors |
| `/api/classify` | POST | Defect type classification |
| `/api/severity` | POST | Severity estimation |

### Request Format

```json
POST /api/classify
{
  "features": [120.5, 30.2, 0.25, 4.1, 140.0, 120.0, 100.0, 0.55, 1200.0, 30.0, 120.0, 0.1, 18.0, 0.25]
}
```

## Running Tests

```bash
python test_api.py
```

## Project Structure

```
cv-pipeline-inspector/
    cv_inspector/
        __init__.py
        data_generator.py
        models/
            __init__.py
            defect_classifier.py
            severity_estimator.py
            defect_detector.py
        utils/
            __init__.py
            image_processor.py
    outputs/
        models/
    templates/
        index.html
    .github/
        workflows/
            ci.yml
    app.py
    train.py
    test_api.py
    requirements.txt
    setup.py
    .gitignore
    README.md
```

## License

MIT License

## Author

Elaborado por Ing. Kelvin Cabrera
