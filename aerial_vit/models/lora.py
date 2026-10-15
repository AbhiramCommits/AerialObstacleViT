"""LoRA adapters for the frozen DINOv2 attention/MLP projections.

Two interchangeable backends:

``native`` (default)
    :class:`LoRALinear` replaces the matching ``nn.Linear`` modules in place.
    Zero third-party version risk, fully inspectable, and it works on MPS/CPU
    without any PEFT-specific device handling.

``peft``
    ``peft.get_peft_model`` with a :class:`peft.LoraConfig`.  Kept available
    because PEFT is the ecosystem standard, but not the default path.

Both are reported with the same accounting helpers, so the Step 4 ablation can
compare ``lora_r4`` / ``lora_r16`` against ``linear`` / ``full`` on equal terms.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "LoRALinear",
    "inject_lora",
    "apply_lora",
    "lora_parameter_count",
    "lora_parameter_summary",
    "merge_lora",
    "count_lora_modules",
]

DEFAULT_TARGETS = (
    # transformers v5 DINOv2 naming
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    # legacy / timm-ish naming, kept so the config is portable
    "query",
    "key",
    "value",
    "dense",
)


class LoRALinear(nn.Module):
    """``y = W0 x + (alpha / r) * B A x`` with ``W0`` frozen and ``B`` zero-init.

    Zero-initialising ``B`` means the adapted model is *exactly* the frozen model
    at step 0, which is what makes LoRA a clean control against the linear probe.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = self.lora_dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + delta * self.scaling

    @torch.no_grad()
    def merged_weight(self) -> torch.Tensor:
        """``W0 + scaling * B A`` -- used by :func:`merge_lora`."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}"


# --------------------------------------------------------------------------- #
# injection
# --------------------------------------------------------------------------- #


def _leaf(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _set_submodule(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parent_path, _, leaf = dotted.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, new)


def inject_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: Iterable[str] = DEFAULT_TARGETS,
) -> list[str]:
    """Replace matching ``nn.Linear`` layers with :class:`LoRALinear` in place.

    Returns the list of replaced module paths, which is what the tests assert on.
    """
    target_set = {t.lower() for t in targets}
    to_replace = [
        name
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and _leaf(name).lower() in target_set
    ]
    for name in to_replace:
        base = model.get_submodule(name)
        _set_submodule(model, name, LoRALinear(base, rank=rank, alpha=alpha, dropout=dropout))
    # everything that is not a LoRA parameter stays frozen
    for name, p in model.named_parameters():
        if ".lora_A" not in name and ".lora_B" not in name:
            p.requires_grad_(False)
    return to_replace


def apply_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    targets: Iterable[str] = DEFAULT_TARGETS,
    impl: str = "native",
    **_: Any,
):
    """Inject LoRA with either the native or the PEFT backend."""
    impl = (impl or "native").lower()
    if impl == "native":
        replaced = inject_lora(model, rank, alpha, dropout, targets)
        return model, {"impl": "native", "replaced": replaced, "rank": rank, "alpha": alpha}
    if impl == "peft":
        from peft import LoraConfig, get_peft_model

        conf = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=[t for t in targets if t != "dense"],
            bias="none",
        )
        wrapped = get_peft_model(model, conf)
        return wrapped, {
            "impl": "peft",
            "replaced": list(conf.target_modules),
            "rank": rank,
            "alpha": alpha,
        }
    raise ValueError(f"unknown LoRA impl {impl!r}")


# --------------------------------------------------------------------------- #
# accounting
# --------------------------------------------------------------------------- #


def count_lora_modules(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, LoRALinear))


def lora_parameter_count(model: nn.Module) -> int:
    """Number of trainable parameters introduced *by LoRA* (A and B matrices)."""
    n = 0
    for name, p in model.named_parameters():
        if name.endswith("lora_A") or name.endswith("lora_B"):
            n += p.numel()
    return n


def lora_parameter_summary(model: nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = lora_parameter_count(model)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "lora_params": int(lora),
        "non_lora_trainable_params": int(trainable - lora),
        "lora_pct_of_total": round(100.0 * lora / max(1, total), 6),
        "trainable_pct_of_total": round(100.0 * trainable / max(1, total), 6),
        "n_lora_modules": count_lora_modules(model),
    }


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Fold every ``LoRALinear`` back into a plain ``nn.Linear``. Returns count."""
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name in names:
        layer: LoRALinear = model.get_submodule(name)
        merged = nn.Linear(
            layer.base.in_features, layer.base.out_features, bias=layer.base.bias is not None
        )
        merged.weight.copy_(layer.merged_weight())
        if layer.base.bias is not None:
            merged.bias.copy_(layer.base.bias)
        merged = merged.to(layer.base.weight.device, layer.base.weight.dtype)
        _set_submodule(model, name, merged)
    return len(names)
