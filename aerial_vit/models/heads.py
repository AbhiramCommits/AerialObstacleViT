"""Perception heads + the assembled ``AerialPerceptionModel``.

Three heads, deliberately at very different capacities, so the linear-probe vs
LoRA vs full-finetune ablation in Step 4 is a fair comparison of *backbone*
adaptation rather than of head size:

``LinearSegHead``  a single 1x1 conv on the frozen patch tokens (the classic
linear probe), bilinearly upsampled to the input resolution.

``ConvSegHead``    a small 2-3 layer conv decoder (GroupNorm + GELU).

``BoxHead``        an anchor-free FCOS-lite head on the token grid, with an
optional 2x upsampled second level so that 6-15 px objects (persons,
vehicles) are not invisible to the detector.

Losses: cross-entropy + Dice for segmentation; sigmoid-focal on the dense class
logits, GIoU on the positive regressions (centerness-weighted) and sigmoid-focal
on the centerness branch for detection.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from aerial_vit.models.box_ops import fcos_decode, fcos_encode, generate_locations, giou_loss

__all__ = [
    "LinearSegHead",
    "ConvSegHead",
    "BoxHead",
    "AerialPerceptionModel",
    "build_seg_head",
    "dice_loss",
    "focal_ce_loss",
    "sigmoid_focal_loss",
    "build_targets",
    "detection_loss",
]


# --------------------------------------------------------------------------- #
# losses
# --------------------------------------------------------------------------- #


def dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int, eps: float = 1.0
) -> torch.Tensor:
    """Soft multi-class Dice over the batch."""
    probs = logits.softmax(dim=1)
    tgt = F.one_hot(targets.clamp(0, num_classes - 1), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * tgt).sum(dims)
    card = probs.sum(dims) + tgt.sum(dims)
    return (1 - (2 * inter + eps) / (card + eps)).mean()


def focal_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy with focal modulation for the segmentation branch."""
    ce = F.cross_entropy(
        logits, targets, weight=weight, reduction="none", label_smoothing=label_smoothing
    )
    pt = torch.exp(-ce)
    return (((1 - pt) ** gamma) * ce).mean()


def sigmoid_focal_loss(
    logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """FCOS-style sigmoid focal loss, normalised by the number of positives."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce
    n_pos = targets.sum().clamp(min=1.0)
    return loss.sum() / n_pos


# --------------------------------------------------------------------------- #
# segmentation heads
# --------------------------------------------------------------------------- #


class LinearSegHead(nn.Module):
    """1x1 conv on patch tokens -> bilinear upsample. The linear probe."""

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.drop = nn.Dropout2d(dropout)
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, tokens: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        x = self.conv(self.drop(tokens))
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


class ConvSegHead(nn.Module):
    """Small conv decoder (GroupNorm + GELU) then bilinear upsample."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden: int = 256,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        c = in_channels
        for _ in range(max(1, layers)):
            blocks += [
                nn.Conv2d(c, hidden, 3, padding=1),
                nn.GroupNorm(_gn_groups(hidden), hidden),
                nn.GELU(),
                nn.Dropout2d(dropout),
            ]
            c = hidden
        self.trunk = nn.Sequential(*blocks)
        self.out = nn.Conv2d(c, num_classes, 1)

    def forward(self, tokens: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        x = self.out(self.trunk(tokens))
        return F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)


def _gn_groups(channels: int, target: int = 8) -> int:
    for g in range(min(target, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def build_seg_head(
    kind: str, in_channels: int, num_classes: int, hidden: int, layers: int, dropout: float
):
    kind = (kind or "linear").lower()
    if kind in ("linear", "probe"):
        return LinearSegHead(in_channels, num_classes, dropout=min(dropout, 0.2))
    if kind in ("conv", "decoder"):
        return ConvSegHead(in_channels, num_classes, hidden=hidden, layers=layers, dropout=dropout)
    raise ValueError(f"unknown seg head {kind!r}")


# --------------------------------------------------------------------------- #
# detection head
# --------------------------------------------------------------------------- #


class BoxHead(nn.Module):
    """Anchor-free FCOS-lite head over the ViT token grid.

    Each level ``u`` in ``upsample_factors`` interpolates the token map by ``u``
    and predicts class logits, ``ltrb`` distances (in stride units) and an
    objectness/centerness logit.  The stride for level ``u`` is
    ``patch_size / u`` pixels.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden: int = 128,
        upsample_factors: tuple[int, ...] = (1, 2),
        dropout: float = 0.1,
        patch_size: int = 14,
        max_reg: float = 12.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.upsample_factors = tuple(int(u) for u in upsample_factors)
        self.patch_size = int(patch_size)
        self.max_reg = float(max_reg)
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.GroupNorm(_gn_groups(hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(_gn_groups(hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.cls = nn.Conv2d(hidden, num_classes, 3, padding=1)
        self.reg = nn.Conv2d(hidden, 4, 3, padding=1)
        self.obj = nn.Conv2d(hidden, 1, 3, padding=1)

        # Bias the regression branch towards a small *positive* initial box.
        # A zero-initialised ltrb head predicts degenerate boxes whose IoU (and
        # whose IoU-gradient) are both exactly zero, which freezes training at
        # loss == 1.0 forever.  Starting from a box of ~1 stride keeps the
        # gradient alive from step 0.
        nn.init.constant_(self.reg.bias, 0.5)
        nn.init.normal_(self.reg.weight, std=0.01)
        # Focal-style prior for the objectness branch (~1% positive).
        nn.init.constant_(self.obj.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, tokens: torch.Tensor) -> list[dict[str, Any]]:
        b = tokens.shape[0]
        levels: list[dict[str, Any]] = []
        for u in self.upsample_factors:
            f = (
                F.interpolate(tokens, scale_factor=u, mode="bilinear", align_corners=False)
                if u != 1
                else tokens
            )
            f = self.shared(f)
            cls = self.cls(f)
            reg = self.reg(f).clamp(min=-self.max_reg, max=self.max_reg)
            obj = self.obj(f)
            h, w = f.shape[-2:]
            stride = self.patch_size / u
            loc = generate_locations(h, w, stride, tokens.device, tokens.dtype)
            levels.append(
                {
                    "cls": cls.reshape(b, self.num_classes, -1).permute(0, 2, 1).contiguous(),
                    "reg": reg.reshape(b, 4, -1).permute(0, 2, 1).contiguous(),
                    "obj": obj.reshape(b, 1, -1).permute(0, 2, 1).contiguous(),
                    "locations": loc,
                    "stride": stride,
                    "shape": (h, w),
                }
            )
        return levels


# --------------------------------------------------------------------------- #
# targets + detection loss
# --------------------------------------------------------------------------- #


def build_targets(
    levels: list[dict[str, Any]],
    boxes: list[torch.Tensor],
    labels: list[torch.Tensor],
    num_classes: int,
) -> list[dict[str, torch.Tensor]]:
    """Encode GT boxes onto every level's location grid."""
    out: list[dict[str, torch.Tensor]] = []
    for lvl in levels:
        cls_t, reg_t, cen_t, pos = [], [], [], []
        for b in range(len(boxes)):
            t = fcos_encode(
                lvl["locations"], lvl["stride"], boxes[b], labels[b].long(), num_classes
            )
            cls_t.append(t["cls"])
            reg_t.append(t["reg"])
            cen_t.append(t["centerness"])
            pos.append(t["pos_mask"])
        out.append(
            {
                "cls": torch.stack(cls_t),
                "reg": torch.stack(reg_t),
                "centerness": torch.stack(cen_t),
                "pos_mask": torch.stack(pos),
            }
        )
    return out


def _boxes_from_ltrb(ltrb: torch.Tensor, locations: torch.Tensor) -> torch.Tensor:
    """``(B, N, 4)`` distances in pixels + ``(B, N, 2)`` centres -> xyxy."""
    loc = locations.unsqueeze(0).expand_as(ltrb[..., :2]) if locations.dim() == 2 else locations
    return torch.stack(
        [
            loc[..., 0] - ltrb[..., 0],
            loc[..., 1] - ltrb[..., 1],
            loc[..., 0] + ltrb[..., 2],
            loc[..., 1] + ltrb[..., 3],
        ],
        dim=-1,
    )


def detection_loss(
    levels: list[dict[str, Any]],
    targets: list[dict[str, torch.Tensor]],
    cfg,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Focal (class) + focal (objectness) + L1 + GIoU box regression, averaged over levels.

    The L1 term is not decoration: with an ``ltrb`` parameterisation the
    regression target is always positive, and L1 gives a constant-magnitude
    gradient even when the predicted box is degenerate -- which is exactly the
    regime a randomly initialised head starts in.
    """
    total = levels[0]["cls"].sum() * 0.0
    parts = {"focal": 0.0, "giou": 0.0, "obj": 0.0, "bbox_l1": 0.0}
    for lvl, tgt in zip(levels, targets, strict=False):
        focal = sigmoid_focal_loss(lvl["cls"], tgt["cls"], cfg.focal_alpha, cfg.focal_gamma)
        obj = sigmoid_focal_loss(lvl["obj"][..., 0], tgt["centerness"], 0.25, cfg.focal_gamma)

        pos = tgt["pos_mask"]
        bbox_l1 = lvl["reg"].sum() * 0.0
        giou = lvl["reg"].sum() * 0.0
        if bool(pos.any()):
            stride = lvl["stride"]
            bbox_l1 = F.l1_loss(lvl["reg"][pos], tgt["reg"][pos], reduction="mean")
            pred = _boxes_from_ltrb(lvl["reg"] * stride, lvl["locations"])
            truth = _boxes_from_ltrb(tgt["reg"] * stride, lvl["locations"])
            gl = giou_loss(pred[pos], truth[pos])
            w = tgt["centerness"][pos]
            giou = (gl * w).sum() / w.sum().clamp(min=1e-6)

        total = (
            total
            + cfg.focal_weight * focal
            + cfg.obj_weight * obj
            + cfg.bbox_weight * bbox_l1
            + cfg.giou_weight * giou
        )
        parts["focal"] += float(focal.detach())
        parts["obj"] += float(obj.detach())
        parts["giou"] += float(giou.detach())
        parts["bbox_l1"] += float(bbox_l1.detach())

    n = max(1, len(levels))
    return total / n, {k: v / n for k, v in parts.items()}


# --------------------------------------------------------------------------- #
# assembled model
# --------------------------------------------------------------------------- #


class AerialPerceptionModel(nn.Module):
    """Frozen DINOv2 backbone + segmentation head + optional FCOS-lite box head."""

    def __init__(
        self,
        backbone,
        num_classes: int,
        seg_head: str = "linear",
        head_dim: int = 256,
        conv_layers: int = 2,
        use_box_head: bool = True,
        box_head_dim: int = 128,
        dropout: float = 0.1,
        box_upsample: tuple[int, ...] = (1, 2),
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.seg = build_seg_head(
            seg_head, backbone.hidden_size, num_classes, head_dim, conv_layers, dropout
        )
        self.box_head = (
            BoxHead(
                backbone.hidden_size,
                num_classes,
                hidden=box_head_dim,
                upsample_factors=box_upsample,
                dropout=dropout,
                patch_size=backbone.patch_size,
            )
            if use_box_head
            else None
        )

    # -- forward ----------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        return_detections: bool = False,
        score_thresh: float = 0.05,
        nms_iou: float = 0.5,
        max_dets: int = 100,
    ) -> dict[str, Any]:
        h, w = x.shape[-2:]
        feats = self.backbone.forward_features(x)
        out: dict[str, Any] = {
            "seg_logits": self.seg(feats, (h, w)),
            "features": feats,
            "image_size": (h, w),
        }
        if self.box_head is not None:
            out["box"] = self.box_head(feats)
        if return_detections:
            out["detections"] = self.decode(out, score_thresh, nms_iou, max_dets)
        return out

    # -- decoding ---------------------------------------------------------- #
    @torch.no_grad()
    def decode(
        self,
        out: dict[str, Any],
        score_thresh: float = 0.05,
        nms_iou: float = 0.5,
        max_dets: int = 100,
    ) -> list[dict[str, torch.Tensor]]:
        if self.box_head is None or "box" not in out:
            return [
                {
                    "boxes": torch.zeros((0, 4)),
                    "scores": torch.zeros((0,)),
                    "labels": torch.zeros((0,), dtype=torch.long),
                }
            ]
        h, w = out["image_size"]
        batch = out["box"][0]["cls"].shape[0]
        results: list[dict[str, torch.Tensor]] = []
        for b in range(batch):
            boxes, scores, labels = [], [], []
            for lvl in out["box"]:
                bx, sc, lb = fcos_decode(
                    lvl["locations"], lvl["stride"], lvl["cls"][b], lvl["reg"][b], lvl["obj"][b]
                )
                keep = sc > score_thresh
                boxes.append(bx[keep])
                scores.append(sc[keep])
                labels.append(lb[keep])
            bx = torch.cat(boxes) if boxes else torch.zeros((0, 4))
            sc = torch.cat(scores) if scores else torch.zeros((0,))
            lb = torch.cat(labels) if labels else torch.zeros((0,), dtype=torch.long)
            if bx.numel():
                bx = bx.clone()
                bx[:, [0, 2]] = bx[:, [0, 2]].clamp(0, w)
                bx[:, [1, 3]] = bx[:, [1, 3]].clamp(0, h)
                keep = (bx[:, 2] - bx[:, 0] > 1) & (bx[:, 3] - bx[:, 1] > 1)
                bx, sc, lb = bx[keep], sc[keep], lb[keep]
            if bx.numel():
                from aerial_vit.models.box_ops import nms

                idx = nms(bx, sc, nms_iou)[:max_dets]
                bx, sc, lb = bx[idx], sc[idx], lb[idx]
            results.append({"boxes": bx, "scores": sc, "labels": lb})
        return results

    # -- parameter accounting --------------------------------------------- #
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_summary(self) -> dict[str, Any]:
        total = self.total_parameter_count()
        trainable = self.trainable_parameter_count()
        heads = self.head_parameters()
        return {
            "total_params": int(total),
            "trainable_params": int(trainable),
            "frozen_params": int(total - trainable),
            "trainable_pct": round(100.0 * trainable / max(1, total), 4),
            "head_params": int(heads),
            "backbone_params": int(total - heads),
        }

    def head_parameters(self) -> int:
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("backbone."))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(trainable)
        self.backbone.cfg.freeze = not trainable
