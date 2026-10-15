"""Export helpers: a tensor-only wrapper, TorchScript, and ONNX (opset 17).

The deployed artefact is a *tensor-in / tensor-out* module.  ``AerialPerceptionModel``
returns a dict of nested structures (nice in Python, awkward in ONNX), so
:class:`ExportWrapper` flattens it into a fixed tuple:

``seg_logits``            ``(B, C, H, W)``
``L{i}_cls``              ``(B, C, h_i, w_i)``  per detection level
``L{i}_reg``              ``(B, 4, h_i, w_i)``
``L{i}_obj``              ``(B, 1, h_i, w_i)``

Dynamic batch is declared on every input/output, so one graph serves batch 1 and
batch 8 (which is exactly what the latency benchmark needs).
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from aerial_vit.utils.logging import get_logger

log = get_logger("aerialvit.export")

__all__ = [
    "ExportWrapper",
    "export_torchscript",
    "export_onnx",
    "onnx_output_names",
    "verify_torchscript",
    "verify_onnx_parity",
    "DEFAULT_OPSET",
]

DEFAULT_OPSET = 17


class ExportWrapper(nn.Module):
    """Flatten the perception model's dict output into a tuple of tensors."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self.model(images)
        parts: list[torch.Tensor] = [out["seg_logits"]]
        for lvl in out.get("box", []) or []:
            b, n, c = lvl["cls"].shape
            h, w = lvl["shape"]
            parts.append(lvl["cls"].transpose(1, 2).reshape(b, c, h, w).contiguous())
            parts.append(lvl["reg"].transpose(1, 2).reshape(b, 4, h, w).contiguous())
            parts.append(lvl["obj"].transpose(1, 2).reshape(b, 1, h, w).contiguous())
        return tuple(parts)


def onnx_output_names(model: nn.Module) -> list[str]:
    """Output names that match :class:`ExportWrapper`'s tuple ordering."""
    names = ["seg_logits"]
    n_levels = len(model.box_head.upsample_factors) if getattr(model, "box_head", None) else 0
    for i in range(n_levels):
        names += [f"L{i}_cls", f"L{i}_reg", f"L{i}_obj"]
    return names


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #


def export_torchscript(model: nn.Module, path: str | Path, example: torch.Tensor) -> Path:
    """Trace the wrapper to TorchScript and round-trip it to verify it loads."""
    wrapper = ExportWrapper(model).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(p))
    loaded = torch.jit.load(str(p))  # fails loudly if the artefact is broken
    loaded.eval()
    return p


def export_onnx(
    model: nn.Module,
    path: str | Path,
    example: torch.Tensor,
    opset: int = DEFAULT_OPSET,
    prefer_dynamo: bool = True,
) -> dict[str, Any]:
    """Export to ONNX with a dynamic batch dimension.

    Tries the modern ``torch.export``-based exporter first and falls back to the
    legacy TorchScript exporter, recording which one produced the artefact.  The
    file is validated with :mod:`onnx` and round-tripped through onnxruntime.
    """
    import onnx
    import onnxruntime as ort

    wrapper = ExportWrapper(model).eval()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    names = onnx_output_names(model)
    dynamic_axes = {"images": {0: "batch"}}
    for nm in names:
        dynamic_axes[nm] = {0: "batch"}

    used = None
    errors: list[str] = []
    attempts = [True, False] if prefer_dynamo else [False]
    for dynamo in attempts:
        try:
            kwargs: dict[str, Any] = {
                "input_names": ["images"],
                "output_names": names,
                "opset_version": opset,
                "dynamic_axes": dynamic_axes,
                "do_constant_folding": True,
            }
            if dynamo:
                if not _has_onnxscript():
                    raise RuntimeError("onnxscript not installed")
                torch.onnx.export(wrapper, (example,), str(p), dynamo=True, **kwargs)
            else:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    torch.onnx.export(wrapper, (example,), str(p), dynamo=False, **kwargs)
            used = "dynamo" if dynamo else "torchscript-legacy"
            break
        except Exception as exc:  # noqa: BLE001 - try the other exporter
            errors.append(f"{'dynamo' if dynamo else 'legacy'}: {type(exc).__name__}: {exc}")
            log.warning("ONNX export via %s failed: %s", "dynamo" if dynamo else "legacy", exc)

    if used is None:
        raise RuntimeError("all ONNX export attempts failed: " + " | ".join(errors))

    onnx_model = onnx.load(str(p))
    onnx.checker.check_model(onnx_model)
    with contextlib.suppress(Exception):
        onnx.shape_inference.infer_shapes(onnx_model)

    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": example.numpy()})
    # Large initialisers are written to a sidecar `<name>.onnx.data` file, so the
    # graph file alone is a misleading size. Report both.
    ext = Path(str(p) + ".data")
    ext_bytes = ext.stat().st_size if ext.exists() else 0
    return {
        "path": str(p),
        "exporter": used,
        "opset": opset,
        "dynamic_batch": True,
        "size_bytes": p.stat().st_size,
        "external_data_bytes": ext_bytes,
        "total_bytes": p.stat().st_size + ext_bytes,
        "external_data_file": str(ext) if ext_bytes else None,
        "n_outputs": len(out),
        "output_names": names,
        "onnx_ir_version": onnx_model.ir_version,
        "onnxruntime_providers": ort.get_available_providers(),
        "export_errors": errors,
    }


def _has_onnxscript() -> bool:
    try:
        import onnxscript  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# parity
# --------------------------------------------------------------------------- #


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


@torch.no_grad()
def verify_torchscript(
    model: nn.Module, ts_path: str | Path, example: torch.Tensor
) -> dict[str, Any]:
    """TorchScript vs eager PyTorch parity on a fixed input."""
    wrapper = ExportWrapper(model).eval()
    ref = wrapper(example)
    loaded = torch.jit.load(str(ts_path)).eval()
    got = loaded(example)
    diffs = [_max_abs_diff(r.numpy(), g.numpy()) for r, g in zip(ref, got, strict=False)]
    miou = _argmax_miou(ref[0], got[0])
    return {
        "max_abs_diff": round(max(diffs) if diffs else 0.0, 8),
        "per_output_max_abs_diff": [round(d, 8) for d in diffs],
        "argmax_agreement": round(_argmax_agreement(ref[0], got[0]), 6),
        "miou": miou,
        "tolerance": 1e-4,
        "passed": bool(diffs and max(diffs) <= 1e-4),
    }


@torch.no_grad()
def verify_onnx_parity(
    model: nn.Module,
    onnx_path: str | Path,
    example: torch.Tensor,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """ONNX Runtime vs eager PyTorch parity on a fixed input."""
    import onnxruntime as ort

    wrapper = ExportWrapper(model).eval()
    ref = [t.numpy() for t in wrapper(example)]
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"images": example.numpy()})
    diffs = [_max_abs_diff(r, g) for r, g in zip(ref, got, strict=False)]
    miou = _argmax_miou(ref[0], got[0])
    return {
        "max_abs_diff": round(max(diffs) if diffs else 0.0, 8),
        "per_output_max_abs_diff": [round(d, 8) for d in diffs],
        "argmax_agreement": round(_argmax_agreement(ref[0], got[0]), 6),
        "miou": miou,
        "tolerance": tolerance,
        "passed": bool(diffs and max(diffs) <= tolerance),
    }


def _to_np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _argmax_agreement(a, b) -> float:
    a, b = _to_np(a), _to_np(b)
    return float((a.argmax(axis=1) == b.argmax(axis=1)).mean())


def _argmax_miou(a, b) -> float:
    """Treat the PyTorch argmax map as ground truth and score the exported map."""
    from aerial_vit.eval.metrics import ConfusionMatrix, seg_metrics_from_confusion

    a, b = _to_np(a), _to_np(b)
    gt = a.argmax(axis=1).reshape(-1)
    pred = b.argmax(axis=1).reshape(-1)
    k = int(max(gt.max(), pred.max())) + 1
    cm = ConfusionMatrix(k)
    cm.update(pred, gt)
    return round(seg_metrics_from_confusion(cm)["miou"], 6)


def load_export_meta(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}
