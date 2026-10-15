"""Latency benchmarking for eager PyTorch, TorchScript and ONNX Runtime.

Design choices that make the numbers trustworthy:

* **Warmup before timing.**  The first forward on any accelerator pays kernel
  compilation / graph optimisation; timing it would report a 10x-slow model.
* **``>= 50`` timed iterations** per configuration, with per-iteration
  ``perf_counter`` samples, so p50/p95 are real percentiles and not a mean that
  hides a long tail.
* **Explicit device per backend.**  ONNX Runtime runs on CPU here, so we also
  measure eager and TorchScript *on CPU* to give a like-for-like comparison
  instead of silently comparing MPS against CPU.
* **Stated requirement.**  The "onboard feasibility" verdict compares batch-1
  throughput against a named frame rate (default 10 Hz) rather than an arbitrary
  threshold.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aerial_vit.utils.logging import get_logger

log = get_logger("aerialvit.bench")

__all__ = [
    "BenchConfig",
    "benchmark_callable",
    "summarise_latencies",
    "onboard_verdict",
    "plot_latency",
]


class BenchConfig:
    def __init__(
        self, warmup: int = 10, iterations: int = 50, resolution: int = 224, batch_size: int = 1
    ):
        self.warmup = int(warmup)
        self.iterations = int(iterations)
        self.resolution = int(resolution)
        self.batch_size = int(batch_size)


def benchmark_callable(
    fn: Callable[[torch.Tensor], Any],
    cfg: BenchConfig,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Time ``fn`` on random inputs; returns mean/p50/p95 latency and FPS."""
    device = torch.device(device)
    x = torch.randn(cfg.batch_size, 3, cfg.resolution, cfg.resolution, device=device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    with torch.no_grad():
        for _ in range(max(0, cfg.warmup)):
            fn(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times: list[float] = []
    for _ in range(max(1, cfg.iterations)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times, np.float64)
    mean = float(arr.mean())
    return {
        "iterations": len(times),
        "warmup": cfg.warmup,
        "batch_size": cfg.batch_size,
        "resolution": f"{cfg.resolution}x{cfg.resolution}",
        "resolution_px": cfg.resolution,
        "device": str(device),
        "mean_ms": round(mean, 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
        "std_ms": round(float(arr.std()), 4),
        "fps": round(1000.0 * cfg.batch_size / mean, 4) if mean > 0 else None,
        "fps_per_stream": round(1000.0 / mean, 4) if mean > 0 else None,
        "all_ms": [round(t, 4) for t in times],
    }


def summarise_latencies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate rows by (backend, resolution, batch) for compact reporting."""
    out: dict[str, Any] = {}
    for r in rows:
        key = f"{r['backend']}|{r['resolution']}|b{r['batch_size']}"
        out[key] = {k: r.get(k) for k in ("mean_ms", "p50_ms", "p95_ms", "fps", "device")}
    return out


def onboard_verdict(
    rows: list[dict[str, Any]], requirement_hz: float = 10.0, prefer: str = "onnxruntime"
) -> dict[str, Any]:
    """Decide whether the fastest *batch-1* configuration meets a frame-rate target.

    Batch 1 is the operationally relevant case for a streaming delivery drone:
    frames arrive one at a time, so batch-8 throughput is not achievable.
    """
    batch1 = [r for r in rows if r.get("batch_size") == 1 and r.get("fps") is not None]
    if not batch1:
        return {"requirement_hz": requirement_hz, "best_fps": None, "verdict": "no measurements"}

    best_any = max(batch1, key=lambda r: r["fps"])
    per_backend: dict[str, Any] = {}
    for r in batch1:
        cur = per_backend.get(r["backend"])
        if cur is None or r["fps"] > cur["fps"]:
            per_backend[r["backend"]] = {
                "fps": r["fps"],
                "p50_ms": r["p50_ms"],
                "resolution": r["resolution"],
                "device": r["device"],
            }
    verdict = "feasible" if best_any["fps"] >= requirement_hz else "not feasible"
    return {
        "requirement_hz": requirement_hz,
        "best_backend": best_any["backend"],
        "best_resolution": best_any["resolution"],
        "best_fps": best_any["fps"],
        "best_p50_ms": best_any["p50_ms"],
        "best_device": best_any["device"],
        "per_backend_batch1": per_backend,
        "preferred_backend": prefer,
        "preferred_backend_batch1_fps": (per_backend.get(prefer) or {}).get("fps"),
        "headroom_ratio": round(best_any["fps"] / requirement_hz, 3) if requirement_hz else None,
        "verdict": verdict,
        "note": (
            "verdict uses the fastest batch-1 configuration across all backends, because a "
            "streaming drone processes one frame at a time; the CPU-only ONNX Runtime figure "
            "is reported separately under per_backend_batch1"
        ),
    }


def _dev_short(device: str) -> str:
    d = (device or "").lower()
    if "mps" in d:
        return "mps"
    if "cuda" in d:
        return "cuda"
    return "cpu"


def plot_latency(
    rows: list[dict[str, Any]], path: str | Path, requirement_hz: float = 10.0
) -> Path:
    """Grouped bar chart of p50 latency; a second panel shows batch-1 FPS vs target.

    Each bar is a *(backend, device)* pair -- eager and TorchScript are measured
    on both the accelerator and CPU, so collapsing them into one series would
    hide which device the number came from.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _key(r):
        return f"{r['backend']} [{_dev_short(r.get('device', 'cpu'))}]"

    series = sorted({_key(r) for r in rows})
    combos = sorted({(r["resolution"], r["batch_size"]) for r in rows}, key=lambda t: (t[0], t[1]))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=140)
    x = np.arange(len(combos))
    width = 0.8 / max(1, len(series))
    for i, s in enumerate(series):
        vals = []
        for res, bs in combos:
            match = [
                r for r in rows if _key(r) == s and r["resolution"] == res and r["batch_size"] == bs
            ]
            vals.append(match[0]["p50_ms"] if match else 0.0)
        axes[0].bar(x + i * width - 0.4 + width / 2, vals, width, label=s)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{res}\nb{bs}" for res, bs in combos], fontsize=8)
    axes[0].set_ylabel("p50 latency (ms)")
    axes[0].set_yscale("log")
    axes[0].set_title("p50 latency by backend / device / resolution / batch")
    axes[0].legend(fontsize=6.5, ncol=2)
    axes[0].grid(axis="y", alpha=0.25)

    batch1 = [r for r in rows if r["batch_size"] == 1]
    labels = [
        f"{r['backend']}\n{r['resolution']}\n[{_dev_short(r.get('device', 'cpu'))}]" for r in batch1
    ]
    fps = [r["fps"] for r in batch1]
    axes[1].bar(range(len(batch1)), fps, color="#1f77b4")
    axes[1].axhline(
        requirement_hz, color="red", ls="--", lw=1.4, label=f"{requirement_hz:g} Hz requirement"
    )
    axes[1].set_xticks(range(len(batch1)))
    axes[1].set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    axes[1].set_ylabel("FPS (batch 1)")
    axes[1].set_title("batch-1 throughput vs drone requirement")
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


def median_or_none(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(float(statistics.median(values)), 4) if values else None
