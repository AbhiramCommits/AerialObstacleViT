"""``eval`` -- evaluate a trained checkpoint on the test split.

Writes ``artifacts/results/<run_name>_eval.json`` plus COCO-format
``*_gt.json`` / ``*_pred.json`` dumps so the mAP numbers are auditable.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from aerial_vit.checkpoint import load_checkpoint_model
from aerial_vit.commands._common import ConfigOpt, SetOpt, load_cli_config
from aerial_vit.config import RESULTS_DIR, Config
from aerial_vit.data.semantic_drone import build_datasets
from aerial_vit.eval.metrics import evaluate_predictions
from aerial_vit.train_engine import make_loader, predict_dataset, warmup
from aerial_vit.utils.logging import get_logger
from aerial_vit.utils.seed import device_name, resolve_device, seed_everything

app = typer.Typer(add_completion=False, help="Evaluate a trained checkpoint.")
log = get_logger("aerialvit.eval")


def resolve_checkpoint(cfg: Config, checkpoint: str | None) -> Path:
    if checkpoint:
        p = Path(checkpoint)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    cand = cfg.run_dir / "best.pt"
    if cand.exists():
        return cand
    runs = sorted(
        (Path(cfg.run.out_dir) / "runs").glob("*/best.pt"), key=lambda p: p.stat().st_mtime
    )
    if runs:
        return runs[-1]
    raise FileNotFoundError("no checkpoint found; train a model first")


def run_eval(cfg: Config, checkpoint: str | None = None, split: str = "test") -> dict:
    seed_everything(cfg.run.seed, cfg.run.deterministic)
    device = resolve_device(cfg.run.device)
    ckpt_path = resolve_checkpoint(cfg, checkpoint)
    model, info, ckpt = load_checkpoint_model(ckpt_path, cfg=cfg, device=device)
    log.info("loaded %s (mode=%s, device=%s)", ckpt_path, ckpt.get("mode"), device_name(device))

    train_ds, val_ds, test_ds = build_datasets(cfg, label_fraction=1.0, seed=cfg.run.seed)
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[split]
    loader = make_loader(ds, cfg.eval.batch_size, False, cfg.data.num_workers, cfg.run.seed)

    dump_dir = RESULTS_DIR / "coco"
    warmup(model, loader, device)
    t0 = time.perf_counter()
    preds = predict_dataset(model, loader, device, cfg)
    wall = time.perf_counter() - t0

    manifest = json.loads(Path(cfg.run.out_dir, "data_manifest.json").read_text())
    rare = manifest.get("rare_classes", [])

    report = evaluate_predictions(
        preds["seg_preds"],
        preds["seg_gts"],
        preds["det_preds"],
        preds["det_gts"],
        class_names=cfg.data.classes,
        landable_indices=cfg.data.landable_indices,
        num_classes=cfg.data.num_classes,
        rare_classes=rare,
        dump_dir=dump_dir,
        prefix=f"{cfg.run.name}_{split}",
        max_dets=cfg.eval.max_dets,
    )
    report.pop("_raw", None)
    report.update(
        {
            "run_name": cfg.run.name,
            "checkpoint": str(ckpt_path),
            "split": split,
            "split_size": len(preds["seg_preds"]),
            "mode": ckpt.get("mode"),
            "adaptation": {k: v for k, v in info.items() if k != "lora_modules"},
            "device": device_name(device),
            "inference_seconds": round(wall, 3),
            "ms_per_image": round(1000.0 * wall / max(1, len(ds)), 4),
            "fps": round(len(ds) / max(1e-9, wall), 3),
            "image_size": [cfg.data.image_size, cfg.data.image_size],
            "classes": list(cfg.data.classes),
            "config_path": str(cfg.taxonomy_path),
        }
    )

    out = RESULTS_DIR / f"{cfg.run.name}_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    log.info("wrote %s", out)
    return report


@app.command()
def main(
    config: str = ConfigOpt,
    checkpoint: str = typer.Option(None, "--checkpoint", help="Explicit checkpoint path."),
    split: str = typer.Option("test", "--split", help="train | val | test"),
    run_name: str = typer.Option(None, "--run-name", help="Override the output run name."),
    set_: list[str] = SetOpt,
) -> None:
    """Evaluate a checkpoint and write artifacts/results/<run>_eval.json."""
    cfg = load_cli_config(config, set_)
    if run_name:
        cfg.run.name = run_name
    r = run_eval(cfg, checkpoint, split)
    typer.echo(
        "\n".join(
            [
                f"run            : {r['run_name']}  (split={r['split']}, n={r['split_size']})",
                f"mode / device  : {r['mode']} / {r['device']}",
                f"mIoU           : {r['miou']}",
                f"pixel acc      : {r['pixel_accuracy']}",
                f"landing-zone   : IoU={r['landing_zone_iou']}  F1={r['landing_zone_f1']}",
                f"det mAP@0.5    : {r.get('det_map_50')}   mAP@[.5:.95]={r.get('det_map')}",
                f"rare mAP@0.5   : {r.get('rare_map_50')}  classes={r.get('rare_classes')}",
                f"ms/image       : {r['ms_per_image']}  ({r['fps']} FPS)",
                f"trainable      : {r['adaptation']['trainable_params']:,} "
                f"({r['adaptation']['trainable_pct']:.4f}%)",
            ]
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
