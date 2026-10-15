"""Checkpoint save/load helpers shared by ``train``, ``eval``, ``export`` and ``ablate``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from aerial_vit.config import Config
from aerial_vit.models import build_model
from aerial_vit.utils.logging import get_logger

log = get_logger("aerialvit.ckpt")

__all__ = ["load_checkpoint", "load_checkpoint_model", "save_checkpoint"]


def save_checkpoint(
    path: str | Path,
    model,
    cfg: Config,
    epoch: int,
    metrics: dict[str, Any],
    mode: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "epoch": int(epoch),
        "metrics": metrics,
        "mode": mode,
        "param_summary": model.parameter_summary(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, p)
    return p


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_checkpoint_model(
    path: str | Path,
    cfg: Config | None = None,
    device: str | torch.device = "cpu",
    mode: str | None = None,
    pretrained: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Rebuild the exact model that produced ``path`` and load its weights.

    Returns ``(model, adapt_info, checkpoint)``.  ``pretrained=False`` by default
    because the backbone weights are already inside the checkpoint, which keeps
    evaluation and export fully offline.
    """
    ckpt = load_checkpoint(path, map_location="cpu")
    if cfg is None:
        from aerial_vit.config import Config as _Config

        raw = ckpt.get("config", {})
        cfg = _Config()
        from aerial_vit.config import _TYPE_MAP, _build  # local import: internal helper

        for key, val in raw.items():
            if key in _TYPE_MAP and isinstance(val, dict):
                setattr(cfg, key, _build(_TYPE_MAP[key], val))
    mode = mode or ckpt.get("mode", "linear")
    model, info = build_model(
        cfg, mode=mode, pretrained=pretrained, device=torch.device("cpu"), lora_impl="native"
    )
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        log.warning("unexpected keys in checkpoint: %d (e.g. %s)", len(unexpected), unexpected[:3])
    if missing:
        log.warning("missing keys when loading checkpoint: %d (e.g. %s)", len(missing), missing[:3])
    model.to(device)
    model.eval()
    return model, info, ckpt
