"""Label-scarcity ablation: grid construction, aggregation, and figures.

The sweep answers the question the benchmark exists for: *how does the frozen
foundation-feature advantage change as labels disappear?*  Three adaptation
modes at very different trainable-parameter budgets are crossed with five label
fractions, and every cell records the segmentation, landing-zone and detection
metrics plus wall-clock and memory.

This module deliberately contains **no training code** -- it is pure bookkeeping
and plotting, which is what makes it unit-testable on synthetic CSVs.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from aerial_vit.config import FIGURES_DIR, RESULTS_DIR, Config

__all__ = [
    "build_grid",
    "cell_key",
    "aggregate_rows",
    "write_csv",
    "write_summary",
    "plot_miou_vs_labels",
    "plot_params_vs_miou",
    "ABLATION_FIELDS",
]

ABLATION_FIELDS = [
    "mode",
    "label_fraction",
    "seed",
    "n_train_images",
    "trainable_params",
    "total_params",
    "trainable_pct",
    "miou",
    "pixel_accuracy",
    "landing_zone_iou",
    "map_50",
    "map_50_95",
    "rare_map_50",
    "train_seconds",
    "epochs_run",
    "peak_memory_mb",
    "device",
    "run_name",
]

METRIC_FIELDS = ["miou", "landing_zone_iou", "map_50", "map_50_95", "rare_map_50"]


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #


def cell_key(mode: str, fraction: float) -> str:
    return f"{mode}@{float(fraction):.3f}"


def build_grid(
    cfg: Config,
    modes: Sequence[str] | None = None,
    fractions: Sequence[float] | None = None,
    seeds: Sequence[int] | None = None,
    multiseed_max_fraction: float | None = None,
) -> list[dict[str, Any]]:
    """Cartesian product of mode x label fraction x seed.

    Cells at or below ``multiseed_max_fraction`` get every seed (they are small
    and noisy); the large cells get the first seed only, to keep the sweep
    affordable.  Because the runner iterates this list in order, the grid is
    itself the plan that ``ablation.csv`` must match cell-for-cell.
    """
    a = cfg.ablation
    modes = list(modes if modes is not None else a.modes)
    fractions = [float(f) for f in (fractions if fractions is not None else a.label_fractions)]
    seeds = [int(s) for s in (seeds if seeds is not None else a.seeds)]
    cut = (
        a.multiseed_max_fraction
        if multiseed_max_fraction is None
        else float(multiseed_max_fraction)
    )

    grid: list[dict[str, Any]] = []
    for mode in modes:
        for frac in sorted(fractions):
            use_seeds = seeds if frac <= cut + 1e-9 else seeds[:1]
            for seed in use_seeds:
                grid.append(
                    {
                        "mode": mode,
                        "label_fraction": float(frac),
                        "seed": int(seed),
                        "cell": cell_key(mode, frac),
                        "run_name": f"{mode}_f{int(round(frac * 100)):03d}_s{seed}",
                    }
                )
    return grid


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def _f(row: dict[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Mean/std across seeds for each (mode, label fraction) cell.

    Also computes the *few-label gain*: the mIoU gap between every mode and the
    linear probe at the smallest label budget, which is the headline number of
    the ablation.
    """
    rows = list(rows)
    cells: dict[str, dict[str, list[float]]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for r in rows:
        key = cell_key(r["mode"], float(r["label_fraction"]))
        bucket = cells.setdefault(key, {m: [] for m in METRIC_FIELDS})
        for m in METRIC_FIELDS:
            val = _f(r, m)
            if val is not None:
                bucket[m].append(val)
        meta.setdefault(
            key,
            {
                "mode": r["mode"],
                "label_fraction": float(r["label_fraction"]),
                "n_seeds": 0,
                "trainable_params": _f(r, "trainable_params"),
                "total_params": _f(r, "total_params"),
                "trainable_pct": _f(r, "trainable_pct"),
                "train_seconds_mean": None,
                "peak_memory_mb_mean": None,
            },
        )
        meta[key]["n_seeds"] += 1

    summary: dict[str, Any] = {}
    for key, bucket in cells.items():
        entry: dict[str, Any] = dict(meta[key])
        for m in METRIC_FIELDS:
            vals = bucket[m]
            entry[f"{m}_mean"] = round(float(np.mean(vals)), 6) if vals else None
            entry[f"{m}_std"] = round(float(np.std(vals, ddof=0)), 6) if len(vals) > 1 else 0.0
            entry[f"{m}_n"] = len(vals)
        ts = [
            _f(r, "train_seconds")
            for r in rows
            if cell_key(r["mode"], float(r["label_fraction"])) == key
        ]
        ts = [t for t in ts if t is not None]
        mem = [
            _f(r, "peak_memory_mb")
            for r in rows
            if cell_key(r["mode"], float(r["label_fraction"])) == key
        ]
        mem = [m for m in mem if m is not None]
        entry["train_seconds_mean"] = round(float(np.mean(ts)), 3) if ts else None
        entry["peak_memory_mb_mean"] = round(float(np.mean(mem)), 2) if mem else None
        summary[key] = entry

    # few-label gain vs the linear probe at the smallest budget
    if summary:
        min_frac = min(e["label_fraction"] for e in summary.values())
        ref_key = cell_key("linear", min_frac)
        ref = summary.get(ref_key, {}).get("miou_mean")
        gains = {}
        for key, e in summary.items():
            if abs(e["label_fraction"] - min_frac) < 1e-9 and ref is not None:
                gains[key] = round(float(e["miou_mean"] - ref), 6)
        summary["_few_label_gain_vs_linear"] = gains
        summary["_few_label_fraction"] = min_frac
    return summary


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #


def write_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ABLATION_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in ABLATION_FIELDS})
    return p


def read_csv(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open() as fh:
        return list(csv.DictReader(fh))


def write_summary(
    summary: dict[str, Any],
    path: str | Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = dict(summary)
    if extra:
        payload.update(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    return p


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

_MODE_COLOURS = {
    "linear": "#1f77b4",
    "lora_r4": "#2ca02c",
    "lora_r8": "#98df8a",
    "lora_r16": "#ff7f0e",
    "lora_r32": "#ffbb78",
    "full": "#d62728",
}


def _summary_entries(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Only real grid cells -- a summary file also carries a `config` block."""
    return [
        v
        for k, v in summary.items()
        if not k.startswith("_") and isinstance(v, dict) and "mode" in v and "label_fraction" in v
    ]


def _mode_colour(mode: str, i: int = 0) -> str:
    if mode in _MODE_COLOURS:
        return _MODE_COLOURS[mode]
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
    return palette[i % len(palette)]


def plot_miou_vs_labels(
    summary: dict[str, Any],
    path: str | Path,
    metric: str = "miou",
    title: str | None = None,
) -> Path:
    """mIoU vs label fraction, one line per adaptation mode, with std bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = _summary_entries(summary)
    modes = sorted({e["mode"] for e in entries}, key=lambda m: (m != "linear", m))
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=140)
    for i, mode in enumerate(modes):
        pts = sorted([e for e in entries if e["mode"] == mode], key=lambda e: e["label_fraction"])
        x = [max(1e-4, p["label_fraction"] * 100.0) for p in pts]
        y = [p.get(f"{metric}_mean") or 0.0 for p in pts]
        yerr = [p.get(f"{metric}_std") or 0.0 for p in pts]
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            lw=1.8,
            color=_mode_colour(mode, i),
            label=mode,
        )
    ax.set_xscale("log")
    ax.set_xlabel("label budget (% of training labels)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or f"{metric} vs label scarcity ({', '.join(modes)})")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_params_vs_miou(summary: dict[str, Any], path: str | Path, metric: str = "miou") -> Path:
    """Trainable parameters (log x) against mIoU, every grid cell a point."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = _summary_entries(summary)
    fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=140)
    modes = sorted({e["mode"] for e in entries}, key=lambda m: (m != "linear", m))
    for i, mode in enumerate(modes):
        pts = [e for e in entries if e["mode"] == mode]
        x = [max(1.0, p.get("trainable_params") or 1.0) for p in pts]
        y = [p.get(f"{metric}_mean") or 0.0 for p in pts]
        ax.scatter(
            x,
            y,
            s=42,
            color=_mode_colour(mode, i),
            label=mode,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        for p in pts:
            ax.annotate(
                f"{int(round(p['label_fraction'] * 100))}%",
                (max(1.0, p.get("trainable_params") or 1.0), p.get(f"{metric}_mean") or 0.0),
                fontsize=6,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric} vs trainable-parameter budget")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    return p


def default_paths(cfg: Config) -> dict[str, Path]:
    return {
        "csv": RESULTS_DIR / "ablation.csv",
        "summary": RESULTS_DIR / "ablation_summary.json",
        "fig_miou": FIGURES_DIR / "ablation_miou_vs_labels.png",
        "fig_params": FIGURES_DIR / "params_vs_miou.png",
    }


def mean_of(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    vals = [_f(r, key) for r in rows]
    vals = [v for v in vals if v is not None]
    return float(statistics.fmean(vals)) if vals else None
