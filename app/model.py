"""Model inference. predict() will be wired to Person 2's implementation."""
from app.config import CIFAR10_CLASSES


def predict(image_bytes: bytes) -> dict:
    """
    Run model on image bytes. Returns dict with predicted_index, predicted_label, confidence.
    Stub until Person 2's model loader is integrated.
    """
    # Stub: return first class with zero confidence until real model is integrated
    return {
        "predicted_index": 0,
        "predicted_label": CIFAR10_CLASSES[0],
        "confidence": 0.0,
    }
