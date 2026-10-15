"""``ablate`` -- sweep adaptation mode x label fraction and record everything.

    aerialvit ablate --config configs/ablate.yaml

Writes ``artifacts/results/ablation.csv`` (one row per run),
``artifacts/results/ablation_summary.json`` (mean/std per cell) and the two
figures under ``reports/figures/``.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import typer

from aerial_vit.ablation import (
    aggregate_rows,
    build_grid,
    default_paths,
    plot_miou_vs_labels,
    plot_params_vs_miou,
    write_csv,
    write_summary,
)
from aerial_vit.commands._common import ConfigOpt, SetOpt, load_cli_config
from aerial_vit.commands.eval import run_eval
from aerial_vit.commands.train import run_train
from aerial_vit.config import Config
from aerial_vit.utils.logging import get_logger
from aerial_vit.utils.profiling import MemorySampler
from aerial_vit.utils.seed import device_name, resolve_device

app = typer.Typer(add_completion=False, help="Label-scarcity ablation sweep.")
log = get_logger("aerialvit.ablate")


def run_cell(cfg: Config, cell: dict, run_root: Path, fast: bool = False) -> dict:
    """Train one (mode, fraction, seed) cell and evaluate it on the test split."""
    c = copy.deepcopy(cfg)
    c.run.name = cell["run_name"]
    run_dir = run_root / cell["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(c.run.device)
    t0 = time.perf_counter()
    # Sample live memory across training *and* evaluation so the recorded peak
    # is a per-cell measurement, not the process' cumulative high-water mark.
    with MemorySampler(device) as mem:
        summary = run_train(
            c,
            mode=cell["mode"],
            label_fraction=cell["label_fraction"],
            seed=cell["seed"],
            run_name=cell["run_name"],
            run_dir=run_dir,
        )
        train_seconds = time.perf_counter() - t0

        # evaluate the trained checkpoint on the held-out test split
        ev = run_eval(
            c,
            checkpoint=str(run_dir / "best.pt"),
            split="test",
            out_path=run_dir / "cell_eval.json",
            dump_dir=run_dir / "coco",
        )

    row = {
        "mode": cell["mode"],
        "label_fraction": float(cell["label_fraction"]),
        "seed": int(cell["seed"]),
        "n_train_images": summary["n_train_images"],
        "trainable_params": summary["adaptation"]["trainable_params"],
        "total_params": summary["adaptation"]["total_params"],
        "trainable_pct": summary["adaptation"]["trainable_pct"],
        "miou": ev["miou"],
        "pixel_accuracy": ev["pixel_accuracy"],
        "landing_zone_iou": ev["landing_zone_iou"],
        "map_50": ev.get("det_map_50"),
        "map_50_95": ev.get("det_map"),
        "rare_map_50": ev.get("rare_map_50"),
        "train_seconds": round(train_seconds, 3),
        "epochs_run": summary["epochs_run"],
        "peak_memory_mb": round(mem.peak_mb, 2),
        "device": device_name(device),
        "run_name": cell["run_name"],
    }
    (run_dir / "ablation_cell.json").write_text(json.dumps(row, indent=1, default=float))
    log.info(
        "cell %-22s frac=%.2f seed=%d  mIoU=%.4f  LZ=%.4f  mAP50=%s  trainable=%s (%.2f%%)  %.0fs",
        cell["mode"],
        cell["label_fraction"],
        cell["seed"],
        row["miou"],
        row["landing_zone_iou"],
        row["map_50"],
        f"{row['trainable_params']:,}",
        row["trainable_pct"],
        train_seconds,
    )
    return row


def run_sweep(
    cfg: Config,
    modes: list[str] | None = None,
    fractions: list[float] | None = None,
    seeds: list[int] | None = None,
    max_steps: int | None = None,
) -> dict:
    grid = build_grid(cfg, modes, fractions, seeds)
    log.info(
        "ablation grid: %d runs (%s)", len(grid), ", ".join(c["cell"] for c in grid[:3]) + " ..."
    )

    run_root = Path(cfg.run.out_dir) / "runs" / "ablation"
    rows: list[dict] = []
    paths = default_paths(cfg)
    t_all = time.perf_counter()

    for i, cell in enumerate(grid):
        c = copy.deepcopy(cfg)
        if max_steps is not None:
            c.train.max_steps = int(max_steps)
        log.info("[%d/%d] %s", i + 1, len(grid), cell["run_name"])
        rows.append(run_cell(c, cell, run_root))
        # checkpoint the partial sweep so a long run is never lost
        write_csv(rows, paths["csv"])

    summary = aggregate_rows(rows)
    extra = {
        "config": {
            "modes": [c["mode"] for c in grid][:0] or sorted({c["mode"] for c in grid}),
            "label_fractions": sorted({c["label_fraction"] for c in grid}),
            "seeds": sorted({c["seed"] for c in grid}),
            "epochs": cfg.train.epochs,
            "batch_size": cfg.train.batch_size,
            "image_size": cfg.data.image_size,
            "n_train_full": cfg.data.n_train,
            "backbone": cfg.backbone.name,
            "seg_head": cfg.model.seg_head,
            "max_steps": max_steps if max_steps is not None else cfg.train.max_steps,
            "data_source": json.loads((Path(cfg.run.out_dir) / "data_manifest.json").read_text())[
                "source"
            ],
        },
        "n_runs": len(rows),
        "n_cells": len({c["cell"] for c in grid}),
        "device": device_name(resolve_device(cfg.run.device)),
        "total_seconds": round(time.perf_counter() - t_all, 3),
    }
    write_summary(summary, paths["summary"], extra)
    plot_miou_vs_labels(summary, paths["fig_miou"], "miou")
    plot_params_vs_miou(summary, paths["fig_params"], "miou")
    log.info("figures -> %s , %s", paths["fig_miou"], paths["fig_params"])
    return {"rows": rows, "summary": summary, "paths": {k: str(v) for k, v in paths.items()}}


@app.command()
def main(
    config: str = ConfigOpt,
    modes: str = typer.Option(None, "--modes", help="Comma-separated adaptation modes."),
    fractions: str = typer.Option(None, "--fractions", help="Comma-separated label fractions."),
    seeds: str = typer.Option(None, "--seeds", help="Comma-separated seeds."),
    max_steps: int = typer.Option(None, "--max-steps", help="Cap steps per epoch."),
    set_: list[str] = SetOpt,
) -> None:
    """Run the full adaptation x label-budget grid."""
    cfg = load_cli_config(config, set_)
    res = run_sweep(
        cfg,
        modes=modes.split(",") if modes else None,
        fractions=[float(f) for f in fractions.split(",")] if fractions else None,
        seeds=[int(s) for s in seeds.split(",")] if seeds else None,
        max_steps=max_steps,
    )
    typer.echo(
        "\n".join(
            [
                f"runs           : {len(res['rows'])}",
                f"cells          : {len([k for k in res['summary'] if not k.startswith('_')])}",
                f"ablation.csv   : {res['paths']['csv']}",
                f"summary        : {res['paths']['summary']}",
                f"figures        : {res['paths']['fig_miou']}",
                f"                 {res['paths']['fig_params']}",
            ]
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
