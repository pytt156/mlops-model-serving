"""
Model inference for CIFAR-10.

Contract:
- predict(image_bytes: bytes) -> dict with keys:
    - predicted_index: int
    - predicted_label: str
    - confidence: float
"""

from app.config import CIFAR10_CLASSES

import torch
import os

from torchvision import transforms
from PIL import Image
from io import BytesIO
from typing import Callable, Any

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)

_preprocess: Callable[[Image.Image], torch.Tensor] = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ]
)

MODEL_TORCHSCRIPT_PATH = "artifacts/model.torchscript.pt"

_model: Any | None = None


def _get_model() -> Any:
    global _model

    if not os.path.exists(MODEL_TORCHSCRIPT_PATH):
        raise FileNotFoundError(
            f"TorchScript model not found: {MODEL_TORCHSCRIPT_PATH}"
        )

    if _model is None:
        model = torch.jit.load(MODEL_TORCHSCRIPT_PATH, map_location="cpu")
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
