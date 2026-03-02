import os
from typing import cast
import torch

from app.config import CIFAR10_CLASSES
from app.make_model import make_model

WEIGHTS_PATH = "artifacts/best_model.pt"
OUT_PATH = "artifacts/model.torchscript.pt"


def main() -> None:
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Missing weights: {WEIGHTS_PATH}")

    model = make_model(num_classes=len(CIFAR10_CLASSES))
    state = torch.load(WEIGHTS_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    example = torch.randn(1, 3, 32, 32)
    ts = cast(torch.jit.ScriptModule, torch.jit.trace(model, example))
    ts.save(OUT_PATH)

    print(f"Saved TorchScript model to: {OUT_PATH}")


if __name__ == "__main__":
    main()
