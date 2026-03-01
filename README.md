# CIFAR-10 Image classification API
`PyTorch` · `FastAPI` · `Docker`

This mini-project demmonstrates model integration and containerized deployment of a trained PyTorch CNN model using FastAPI.

## Architecture Overview
The application is structured as two services:
- **API Service (FastAPI)**
    Handles model loading, preprocessing and inference.
- **UI Service (Streamlit)**
    Provides a web interface for uploading images and displaying predictions.

Both services run in Docker containers and communicate internally via Docker networking.

## System Diagram
```
[ User ]
    ↓
[ Streamlit UI ]  →  [ FastAPI API ]  →  [ PyTorch Model ]
```

## Project Structure
```
tree
```

## Model Details
- Architecture defined in `app/make_model.py`
- Weights stored in `artifacts/best_model.pt`
- Lazy-loaded via `_get_model()`
- Preprocessing:
    - `Resize(32, 32)`
    - `ToTensor()`
    - `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))`
- Softmax used to compute confidence

## API Endpoints

### `GET /health`
Health check endpoint.

Expected response:
```JSON
{"status":"ok"}
```

### `POST /predict`
Accepts an image file (multipart upload) and returns:
```JSON
{
    "predicted_index": 3,
    "predicted_label": cat,
    "confidence": 0.87
}
```

## Streamlit Frontend

[...]

## Screenshots

[...]

## Running the System
### Build and start all services
```bash
docker compose up --build
```

### Access the Services
**API:**
<a link>http://localhost:8000</a>
**Streamlit UI**
<a link> .. </a>

## Verification
### Health Check

### Manual API Test

### UI Test

## Dockerized Deployment

## Pull requests (with review)

## Conclusion