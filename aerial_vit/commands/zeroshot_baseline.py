"""``zeroshot`` -- run the open-vocabulary baselines and compare with the task-trained model.

    aerialvit zeroshot --config configs/full.yaml --method owlvit
    aerialvit zeroshot --config configs/full.yaml --method clip
    aerialvit zeroshot --config configs/full.yaml --method both --task-run full_linear

Writes ``artifacts/results/zeroshot_owlvit.json``, ``zeroshot_clip.json`` and the
combined ``artifacts/results/foundation_vs_tasktrained.csv``.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import typer

from aerial_vit.commands._common import ConfigOpt, SetOpt, load_cli_config
from aerial_vit.config import RESULTS_DIR, Config
from aerial_vit.data.semantic_drone import build_datasets, load_manifest
from aerial_vit.data.transforms import denormalize
from aerial_vit.eval.metrics import evaluate_predictions
from aerial_vit.models.openvocab import (
    CLIPZeroShot,
    OWLViTZeroShot,
    boxes_to_segmentation,
    load_prompts,
)
from aerial_vit.train_engine import make_loader
from aerial_vit.utils.logging import get_logger
from aerial_vit.utils.seed import device_name, resolve_device, seed_everything

app = typer.Typer(add_completion=False, help="Open-vocabulary OWL-ViT / CLIP zero-shot baseline.")
log = get_logger("aerialvit.zeroshot")

METHODS = ("owlvit", "clip")


def build_method(
    method: str,
    cfg: Config,
    device,
    prompts,
    score_thresh: float | None = None,
    window: int = 64,
    stride: int = 32,
):
    if method == "owlvit":
        # Zero-shot detectors emit *uncalibrated* scores that sit far below the
        # 0.5 a supervised detector would use; we keep the top-100 proposals per
        # image so mAP measures ranking quality rather than threshold choice.
        return OWLViTZeroShot(
            cfg.data.classes,
            prompts,
            device=device,
            score_thresh=0.001 if score_thresh is None else float(score_thresh),
            max_dets=cfg.eval.max_dets,
        )
    if method == "clip":
        return CLIPZeroShot(cfg.data.classes, prompts, device=device, window=window, stride=stride)
    raise ValueError(f"unknown method {method!r}")


def run_zeroshot(
    cfg: Config,
    method: str,
    split: str = "test",
    max_images: int | None = None,
    score_thresh: float | None = None,
    window: int = 64,
    stride: int = 32,
) -> dict:
    seed_everything(cfg.run.seed, cfg.run.deterministic)
    device = resolve_device(cfg.run.device)
    prompts = load_prompts()
    manifest = load_manifest(cfg=cfg)
    rare = manifest.get("rare_classes", [])

    _, val_ds, test_ds = build_datasets(cfg, label_fraction=1.0, seed=cfg.run.seed)
    ds = {"val": val_ds, "test": test_ds}.get(split, test_ds)
    if max_images:
        ds.names = ds.names[:max_images]

    model = build_method(method, cfg, device, prompts, score_thresh, window, stride)
    n_params = sum(p.numel() for p in model.model.parameters())
    log.info("%s: %s params, device=%s", method, f"{n_params:,}", device_name(device))

    loader = make_loader(ds, 1, False, 0, cfg.run.seed)
    # warm up the VLM so the first image is not billed for compilation
    for images, _, _, _ in loader:
        model.predict(denormalize(images[0]).astype(np.uint8))
        break

    seg_preds, seg_gts, det_preds, det_gts = [], [], [], []
    names, per_image_ms = [], []
    t_all = time.perf_counter()
    for img_id, (images, masks, boxes, metas) in enumerate(loader):
        rgb = denormalize(images[0]).astype(np.uint8)
        t0 = time.perf_counter()
        out = model.predict(rgb)
        per_image_ms.append(1000.0 * (time.perf_counter() - t0))

        if method == "clip":
            seg = out["seg"]
            dboxes, dscores, dlabels = out["boxes"], out["scores"], out["labels"]
        else:
            seg = boxes_to_segmentation(
                out["boxes"], out["labels"], out["scores"], rgb.shape[:2], cfg.data.num_classes, 0.0
            )
            dboxes, dscores, dlabels = out["boxes"], out["scores"], out["labels"]

        seg_preds.append(seg.astype(np.uint8))
        seg_gts.append(masks[0].numpy().astype(np.uint8))
        gt = boxes[0]
        det_gts.append(
            {
                "image_id": img_id,
                "width": rgb.shape[1],
                "height": rgb.shape[0],
                "boxes": gt[:, :4].numpy() if gt.numel() else np.zeros((0, 4)),
                "labels": gt[:, 4].numpy().astype(int) if gt.numel() else np.zeros((0,), int),
            }
        )
        det_preds.append(
            {"image_id": img_id, "boxes": dboxes, "scores": dscores, "labels": dlabels}
        )
        names.append(metas[0].get("name", str(img_id)))
    wall = time.perf_counter() - t_all

    report = evaluate_predictions(
        seg_preds,
        seg_gts,
        det_preds,
        det_gts,
        class_names=cfg.data.classes,
        landable_indices=cfg.data.landable_indices,
        num_classes=cfg.data.num_classes,
        rare_classes=rare,
        dump_dir=RESULTS_DIR / "coco",
        prefix=f"zeroshot_{method}_{split}",
        max_dets=cfg.eval.max_dets,
    )
    report.update(
        {
            "method": method,
            "model_id": model.model_id,
            "trainable_params": 0,
            "total_params": int(n_params),
            "trainable_pct": 0.0,
            "zero_shot": True,
            "device": device_name(device),
            "split": split,
            "split_size": len(seg_preds),
            "ms_per_image": round(float(np.mean(per_image_ms)), 3),
            "ms_per_image_median": round(float(np.median(per_image_ms)), 3),
            "total_seconds": round(wall, 3),
            "image_size": [cfg.data.image_size, cfg.data.image_size],
            "prompt_file": "configs/prompts.yaml",
        }
    )
    out = RESULTS_DIR / f"zeroshot_{method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    log.info("wrote %s", out)
    return report


def write_comparison(task_run: str | None, cfg: Config) -> Path:
    """Combine the task-trained eval JSON and the zero-shot JSONs into one CSV."""
    rows: list[dict[str, object]] = []

    task_json = None
    if task_run:
        cand = RESULTS_DIR / f"{task_run}_eval.json"
        if cand.exists():
            task_json = json.loads(cand.read_text())
    if task_json is None:
        # newest *_eval.json that is not a zero-shot file
        cands = sorted(RESULTS_DIR.glob("*_eval.json"), key=lambda p: p.stat().st_mtime)
        if cands:
            task_json = json.loads(cands[-1].read_text())

    if task_json:
        rows.append(
            {
                "method": f"DINOv2 (frozen) + {task_json.get('mode')} head",
                "trainable_params": task_json["adaptation"]["trainable_params"],
                "total_params": task_json["adaptation"]["total_params"],
                "trainable_pct": task_json["adaptation"]["trainable_pct"],
                "miou": task_json["miou"],
                "landing_zone_iou": task_json["landing_zone_iou"],
                "map_50": task_json.get("det_map_50"),
                "map_50_95": task_json.get("det_map"),
                "rare_map_50": task_json.get("rare_map_50"),
                "rare_classes": ",".join(task_json.get("rare_classes", [])),
                "ms_per_image": task_json.get("ms_per_image"),
                "device": task_json.get("device"),
                "n_test": task_json.get("split_size"),
                "zero_shot": False,
                "source_artifact": f"artifacts/results/{task_run or ''}" if task_run else "auto",
            }
        )

    for method in METHODS:
        p = RESULTS_DIR / f"zeroshot_{method}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        rows.append(
            {
                "method": f"{method} (zero-shot)",
                "trainable_params": r["trainable_params"],
                "total_params": r["total_params"],
                "trainable_pct": r["trainable_pct"],
                "miou": r["miou"],
                "landing_zone_iou": r["landing_zone_iou"],
                "map_50": r.get("det_map_50"),
                "map_50_95": r.get("det_map"),
                "rare_map_50": r.get("rare_map_50"),
                "rare_classes": ",".join(r.get("rare_classes", [])),
                "ms_per_image": r.get("ms_per_image"),
                "device": r.get("device"),
                "n_test": r.get("split_size"),
                "zero_shot": True,
                "source_artifact": f"artifacts/results/zeroshot_{method}.json",
            }
        )

    out = RESULTS_DIR / "foundation_vs_tasktrained.csv"
    fields = [
        "method",
        "trainable_params",
        "total_params",
        "trainable_pct",
        "miou",
        "landing_zone_iou",
        "map_50",
        "map_50_95",
        "rare_map_50",
        "rare_classes",
        "ms_per_image",
        "device",
        "n_test",
        "zero_shot",
        "source_artifact",
    ]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("wrote %s (%d rows)", out, len(rows))
    return out


@app.command()
def main(
    config: str = ConfigOpt,
    method: str = typer.Option("both", "--method", help="owlvit | clip | both"),
    split: str = typer.Option("test", "--split"),
    task_run: str = typer.Option(None, "--task-run", help="Task-trained run to compare against."),
    max_images: int = typer.Option(None, "--max-images", help="Cap the number of images."),
    score_thresh: float = typer.Option(None, "--score-thresh", help="OWL-ViT score threshold."),
    window: int = typer.Option(64, "--window", help="CLIP sliding-window size."),
    stride: int = typer.Option(32, "--stride", help="CLIP sliding-window stride."),
    set_: list[str] = SetOpt,
) -> None:
    """Run the zero-shot baselines and build the comparison table."""
    cfg = load_cli_config(config, set_)
    methods = METHODS if method == "both" else (method,)
    for m in methods:
        if m not in METHODS:
            raise typer.BadParameter(f"method must be one of {METHODS} or 'both'")
        r = run_zeroshot(cfg, m, split, max_images, score_thresh, window, stride)
        typer.echo(
            f"{m:8s} mIoU={r['miou']:.4f}  LZ-IoU={r['landing_zone_iou']:.4f}  "
            f"mAP@0.5={r.get('det_map_50')}  rare-mAP@0.5={r.get('rare_map_50')}  "
            f"{r['ms_per_image']:.1f} ms/img"
        )
    out = write_comparison(task_run, cfg)
    typer.echo(f"\ncomparison table -> {out}")


if __name__ == "__main__":  # pragma: no cover
    app()
