"""Deterministic seeding and device selection."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything", "resolve_device", "device_name"]


def seed_everything(seed: int = 1337, deterministic: bool = True) -> None:
    """Seed python, numpy and torch (CPU + CUDA) RNGs."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # MPS has no deterministic flag; document that in the README instead.


def resolve_device(spec: str = "auto") -> torch.device:
    """Turn ``auto|cpu|cuda|mps`` into a ``torch.device``.

    ``auto`` prefers CUDA, then Apple MPS, then CPU.  Never raises: if the
    requested accelerator is unavailable we *record* the fallback rather than
    pretending it worked, by returning CPU.
    """
    spec = (spec or "auto").lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    if spec == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    return torch.device(spec)


def device_name(device: torch.device) -> str:
    """Human-readable device description, recorded verbatim in artifacts."""
    if device.type == "cuda":
        try:
            return f"cuda:{torch.cuda.get_device_name(device.index or 0)}"
        except Exception:  # pragma: no cover - driver quirk
            return "cuda"
    if device.type == "mps":
        return "apple-mps (Metal Performance Shaders)"
    return "cpu"
