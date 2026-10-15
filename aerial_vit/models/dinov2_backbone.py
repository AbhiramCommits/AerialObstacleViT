"""Frozen DINOv2 ViT backbone with a slim, head-friendly interface.

``facebook/dinov2-small`` (22.1 M params, 384-d, patch 14) is the default.  The
backbone exposes exactly three things the rest of the codebase needs:

* ``forward_features(x)`` -> patch tokens as ``(B, C, H/14, W/14)``
* ``cls_token(x)``        -> ``(B, C)`` global embedding (for kNN/ODD scores)
* ``freeze()`` / ``unfreeze()`` / ``trainable_parameter_count()``

A **tiny randomly-initialised DINOv2** (`name: tiny`) is available so tests and
CI can exercise the real code path without downloading 85 MB of weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aerial_vit.config import BackboneCfg

# Presets that need no download. `tiny` keeps tests fast; `dinov2-small-random`
# reproduces the *real* dinov2-small architecture (384-d, 12 layers) with random
# weights, so parameter-accounting tests reflect the true trainable percentages
# without CI ever touching the network.
PRESETS: dict[str, dict[str, int]] = {
    "tiny": {
        "hidden_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 3,
        "intermediate_size": 96,
        "image_size": 224,
        "patch_size": 14,
    },
    "dinov2-small-random": {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "intermediate_size": 1536,
        "image_size": 224,
        "patch_size": 14,
    },
}
TINY_PRESET = PRESETS["tiny"]


class DINOv2Backbone(nn.Module):
    """Thin wrapper around a HuggingFace DINOv2 encoder."""

    def __init__(self, cfg: BackboneCfg, pretrained: bool = True):
        super().__init__()
        self.cfg = cfg
        self.name = cfg.name
        preset = PRESETS.get(cfg.name.lower())
        self.preset = cfg.name.lower() if preset else None

        from transformers import AutoModel, Dinov2Config, Dinov2Model

        if preset is not None:
            self.model = Dinov2Model(Dinov2Config(**preset, num_channels=3))
            self.pretrained = False
        elif pretrained:
            self.model = AutoModel.from_pretrained(cfg.name)
            self.pretrained = True
        else:
            conf = Dinov2Config.from_pretrained(cfg.name)
            self.model = Dinov2Model(conf)
            self.pretrained = False

        self.patch_size = int(getattr(self.model.config, "patch_size", cfg.patch_size))
        self.hidden_size = int(self.model.config.hidden_size)
        if cfg.freeze:
            self.freeze()

    # -- feature extraction ------------------------------------------------ #
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Patch tokens reshaped to ``(B, C, H/patch, W/patch)``."""
        out = self.model(pixel_values=x)
        tokens = out.last_hidden_state  # (B, 1+N, C), token 0 is the CLS token
        patch = tokens[:, 1:, :]
        b, n, c = patch.shape
        h = x.shape[2] // self.patch_size
        w = x.shape[3] // self.patch_size
        if h * w != n:  # tolerate non-square / interpolated inputs
            h = w = int(round(n**0.5))
        return patch.transpose(1, 2).reshape(b, c, h, w).contiguous()

    def cls_token(self, x: torch.Tensor) -> torch.Tensor:
        """CLS embedding ``(B, C)``."""
        out = self.model(pixel_values=x)
        return out.last_hidden_state[:, 0, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)

    # -- freezing ---------------------------------------------------------- #
    def freeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    def unfreeze(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(True)

    def train(self, mode: bool = True):  # noqa: D102 - inherit docstring
        super().train(mode)
        # A frozen backbone must stay in eval mode: dropout / stochastic depth
        # would otherwise make "frozen features" non-deterministic.
        if self.cfg.freeze:
            self.model.eval()
        return self

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def num_blocks(self) -> int:
        return int(len(self.model.encoder.layer))

    def named_blocks(self):
        return [(f"blocks.{i}", blk) for i, blk in enumerate(self.model.encoder.layer)]


def build_backbone(cfg: BackboneCfg, pretrained: bool = True) -> DINOv2Backbone:
    return DINOv2Backbone(cfg, pretrained=pretrained)


def tiny_backbone_cfg(**overrides) -> BackboneCfg:
    base = {"name": "tiny", "family": "dinov2", "patch_size": 14, "freeze": True}
    base.update(overrides)
    return BackboneCfg(**base)


def small_backbone_cfg(**overrides) -> BackboneCfg:
    """Real dinov2-small architecture, random weights, no download."""
    base = {"name": "dinov2-small-random", "family": "dinov2", "patch_size": 14, "freeze": True}
    base.update(overrides)
    return BackboneCfg(**base)
