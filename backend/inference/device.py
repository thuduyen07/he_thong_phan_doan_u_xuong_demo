from __future__ import annotations

import os

import torch


def resolve_inference_device(configured_device: str | None) -> torch.device:
    requested_device = os.getenv("DEMO_INFERENCE_DEVICE", configured_device or "auto").strip().lower()

    if requested_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if requested_device == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    return torch.device("cpu")
