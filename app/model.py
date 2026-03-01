"""
Model inference for CIFAR-10.

Contract:
- predict(image_bytes: bytes) -> dict with keys:
    - predicted_index: int
    - predicted_label: str
    - confidence: float
"""

from app.config import CIFAR10_CLASSES
from app.make_model import make_model

import torch
import os

from torchvision import transforms
from PIL import Image
from io import BytesIO

MODEL_WEIGHTS_PATH = "artifacts/best_model.pt"

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_preprocess = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ]
)

_model = None


def _get_model():
    global _model

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS_PATH}")

    if _model is None:
        model = make_model(num_classes=len(CIFAR10_CLASSES))
        state = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _model = model

    return _model


def _bytes_to_tensor(image_bytes: bytes) -> torch.Tensor:
    """Decode image bytes and return tensor of shape (1,3,32,32)"""

    if not image_bytes:
        raise ValueError("Empty image file")

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Could not decode image bytes")

    input_tensor = _preprocess(img)
    batch_tensor = input_tensor.unsqueeze(0)
    return batch_tensor


def predict(image_bytes: bytes) -> dict:
    """Run inference and return contract-compliant dict"""

    model = _get_model()
    x = _bytes_to_tensor(image_bytes)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    predicted_index = int(idx.item())
    return {
        "predicted_index": predicted_index,
        "predicted_label": CIFAR10_CLASSES[predicted_index],
        "confidence": float(conf.item()),
    }
