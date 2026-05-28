from __future__ import annotations

import os
import importlib.util
from typing import Any


def create_torchvision_resnet(*, model_name: str, num_classes: int) -> Any:
    """
    Create a torchvision ResNet without any pretrained weights.
    Intended to be used from TorchStateDictProvider factory.
    """
    try:
        from torchvision import models  # type: ignore
    except Exception as e:
        raise RuntimeError(f"torchvision is not installed: {e}") from e

    name = str(model_name).strip().lower()
    if not hasattr(models, name):
        raise RuntimeError(f"torchvision.models has no '{name}'")
    constructor = getattr(models, name)

    # Newer torchvision: supports weights=None and num_classes
    try:
        return constructor(weights=None, num_classes=int(num_classes))
    except TypeError:
        m = constructor(weights=None)
        # ensure classifier head is adjusted (ResNet has .fc)
        if not hasattr(m, "fc"):
            raise RuntimeError(f"Model '{name}' does not expose an 'fc' attribute")
        import torch  # type: ignore

        in_features = int(m.fc.in_features)
        m.fc = torch.nn.Linear(in_features, int(num_classes))
        return m


def create_timm_model(*, model_name: str, num_classes: int) -> Any:
    """
    Create a timm model without pretrained weights.
    Intended to be used from TorchStateDictProvider factory.
    """
    try:
        import timm  # type: ignore
    except Exception as e:
        raise RuntimeError(f"timm is not installed: {e}") from e
    return timm.create_model(str(model_name), pretrained=False, num_classes=int(num_classes))


def create_emonet(*, n_expression: int = 8) -> Any:
    """
    Create EmoNet architecture (no weights) for TorchStateDictProvider.

    Notes:
    - The architecture implementation is vendored under dp_models/emonet/.
    - We load it by file path to avoid relying on package import layout.
    """
    # dp_models/factories/ -> dp_models/ -> dp_models/emonet/emonet/models/emonet.py
    dp_models_dir = os.path.dirname(__file__)
    emonet_py = os.path.join(
        dp_models_dir,
        "..",
        "emonet",
        "emonet",
        "models",
        "emonet.py",
    )
    emonet_py = os.path.abspath(emonet_py)
    
    if not os.path.isfile(emonet_py):
        raise RuntimeError(f"EmoNet source file not found: {emonet_py}")

    spec = importlib.util.spec_from_file_location("_dp_vendor_emonet", emonet_py)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create import spec for EmoNet")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    EmoNet = getattr(mod, "EmoNet", None)
    if EmoNet is None:
        raise RuntimeError("EmoNet class not found in vendored emonet.py")
    return EmoNet(n_expression=int(n_expression))


