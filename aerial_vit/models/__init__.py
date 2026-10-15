"""Models: frozen DINOv2 backbone, perception heads, LoRA adapters, open-vocab VLMs."""

from __future__ import annotations

from typing import Any

import torch

from aerial_vit.config import Config
from aerial_vit.models.box_ops import box_iou, giou_loss, nms
from aerial_vit.models.dinov2_backbone import (
    DINOv2Backbone,
    build_backbone,
    small_backbone_cfg,
    tiny_backbone_cfg,
)
from aerial_vit.models.heads import (
    AerialPerceptionModel,
    BoxHead,
    ConvSegHead,
    LinearSegHead,
    build_seg_head,
    build_targets,
    detection_loss,
    dice_loss,
    focal_ce_loss,
    sigmoid_focal_loss,
)
from aerial_vit.models.lora import (
    DEFAULT_TARGETS as DEFAULT_LORA_TARGETS,
)
from aerial_vit.models.lora import (
    LoRALinear,
    apply_lora,
    inject_lora,
    lora_parameter_count,
    lora_parameter_summary,
    merge_lora,
)

__all__ = [
    "AerialPerceptionModel",
    "BoxHead",
    "ConvSegHead",
    "DINOv2Backbone",
    "LinearSegHead",
    "LoRALinear",
    "apply_lora",
    "box_iou",
    "build_backbone",
    "build_model",
    "build_seg_head",
    "build_targets",
    "configure_adaptation",
    "detection_loss",
    "dice_loss",
    "focal_ce_loss",
    "giou_loss",
    "inject_lora",
    "lora_parameter_count",
    "lora_parameter_summary",
    "merge_lora",
    "nms",
    "sigmoid_focal_loss",
    "small_backbone_cfg",
    "tiny_backbone_cfg",
]


def configure_adaptation(
    model: AerialPerceptionModel,
    mode: str,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    targets: list[str] | None = None,
    lora_impl: str = "native",
) -> tuple[AerialPerceptionModel, dict[str, Any]]:
    """Set up ``linear`` / ``lora`` / ``full`` adaptation on an assembled model.

    Returns the (possibly rewrapped, for the PEFT backend) model and an info dict.
    """
    mode = (mode or "linear").lower()
    info: dict[str, Any] = {"mode": mode, "lora_impl": None, "lora_modules": []}

    if mode == "linear":
        model.set_backbone_trainable(False)
    elif mode in ("lora", "lora_r4", "lora_r8", "lora_r16", "lora_r32"):
        if mode.startswith("lora_r"):
            rank = int(mode.split("_r")[1])
        model.set_backbone_trainable(False)
        model, linfo = apply_lora(
            model,
            rank=rank,
            alpha=alpha or 2 * rank,
            dropout=dropout,
            targets=targets or list(DEFAULT_LORA_TARGETS),
            impl=lora_impl,
        )
        info.update(lora_impl=linfo["impl"], lora_modules=list(linfo["replaced"]), lora_rank=rank)
    elif mode == "full":
        model.set_backbone_trainable(True)
    else:
        raise ValueError(f"unknown adaptation mode {mode!r}")

    # heads are always trainable
    for name, p in model.named_parameters():
        if name.startswith("seg.") or name.startswith("box_head."):
            p.requires_grad_(True)

    summary = getattr(model, "parameter_summary", None)
    info.update(summary() if callable(summary) else _peft_summary(model))
    if hasattr(model, "named_parameters"):
        info["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        info["total_params"] = sum(p.numel() for p in model.parameters())
        info["trainable_pct"] = round(
            100.0 * info["trainable_params"] / max(1, info["total_params"]), 4
        )
    return model, info


def _peft_summary(model) -> dict[str, Any]:
    from aerial_vit.models.lora import lora_parameter_summary

    return lora_parameter_summary(model)


def build_model(
    cfg: Config,
    mode: str | None = None,
    pretrained: bool = True,
    device: str | torch.device = "cpu",
    lora_impl: str = "native",
) -> tuple[AerialPerceptionModel, dict[str, Any]]:
    """Build the full perception model and configure the requested adaptation."""
    backbone = build_backbone(cfg.backbone, pretrained=pretrained)
    model = AerialPerceptionModel(
        backbone,
        num_classes=cfg.data.num_classes,
        seg_head=cfg.model.seg_head,
        head_dim=cfg.model.head_dim,
        conv_layers=cfg.model.conv_layers,
        use_box_head=cfg.model.box_head,
        box_head_dim=cfg.model.box_head_dim,
        dropout=cfg.model.dropout,
        box_upsample=tuple(cfg.model.box_strides),
    )
    mode = mode or cfg.train.mode
    model, info = configure_adaptation(
        model,
        mode,
        rank=cfg.train.lora_rank,
        alpha=cfg.train.lora_alpha,
        dropout=cfg.train.lora_dropout,
        targets=list(cfg.train.lora_targets),
        lora_impl=lora_impl or cfg.train.lora_impl,
    )
    model.to(device)
    return model, info
