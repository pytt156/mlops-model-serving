from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.model import predict

app = FastAPI(title="CIFAR-10 Prediction API")


class PredictResponse(BaseModel):
    predicted_index: int
    predicted_label: str
    confidence: float


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check for Docker/orchestration."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(file: UploadFile = File(..., description="Image file (.jpg or .png)")):
    """Upload one image; returns predicted class index, label, and confidence."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid or missing image file")
    allowed = ("image/jpeg", "image/png", "image/jpg")
    if file.content_type not in allowed and not (file.filename and file.filename.lower().endswith((".jpg", ".jpeg", ".png"))):
        raise HTTPException(status_code=400, detail="File must be .jpg or .png")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    result = predict(contents)
    return PredictResponse(**result)
