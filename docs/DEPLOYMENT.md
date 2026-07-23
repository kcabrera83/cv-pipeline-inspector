# Deployment Guide - CV Pipeline Inspector

## Docker Deployment

### Build the Image

```bash
cd cv-pipeline-inspector
docker build -t cv-pipeline-inspector .
```

### Run the Container

```bash
docker run -p 5016:5016 cv-pipeline-inspector
```

### With Model Training

```bash
docker run -p 5016:5016 cv-pipeline-inspector bash -c "python train.py && python app.py"
```

## Docker Compose

```yaml
version: '3.8'
services:
  cv-inspector:
    build: .
    ports:
      - "5016:5016"
    volumes:
      - ./outputs:/app/outputs
    environment:
      - FLASK_ENV=production
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| FLASK_ENV | Flask environment mode | development |
| PYTHONUNBUFFERED | Disable Python output buffering | 1 |
| PORT | Server port (hardcoded in app.py) | 5016 |

## Manual Deployment

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Train Models

```bash
python train.py
```

### Run with Gunicorn (Production)

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5016 app:app
```

### Run with Flask Development Server

```bash
python app.py
```

## Production Considerations

- Use **gunicorn** with multiple workers for production deployments
- Ensure `debug=False` in `app.py` (already set by default)
- Configure proper logging for request/error tracking
- Place behind a reverse proxy (nginx/Apache) for SSL termination
- Models load from `outputs/models/` - ensure this directory exists
- Pre-train models with `train.py` before starting the server

## Health Check

```bash
curl http://localhost:5016/api/health
```

Expected response:
```json
{"status": "healthy", "models_loaded": {"classifier": true, ...}, "version": "1.0.0"}
```

## Ports

| Service | Port |
|---------|------|
| CV Pipeline Inspector | 5016 |
