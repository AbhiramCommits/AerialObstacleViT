"""Shared training / evaluation engine used by ``train``, ``eval`` and ``ablate``.

Keeping the loop in one place is what makes the ablation numbers comparable:
every cell runs byte-identical code, differing only in the adaptation mode and
the label budget.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from aerial_vit.config import Config
from aerial_vit.eval.metrics import (
    ConfusionMatrix,
    evaluate_predictions,
    landing_zone_counts,
    seg_metrics_from_confusion,
)
from aerial_vit.models.heads import build_targets, detection_loss, dice_loss, focal_ce_loss
from aerial_vit.utils.logging import append_jsonl, get_logger
from aerial_vit.utils.seed import device_name

log = get_logger("aerialvit.engine")

__all__ = [
    "collate_fn",
    "make_loader",
    "run_epoch",
    "evaluate_model",
    "predict_dataset",
    "compute_class_weights",
    "train_model",
]


def collate_fn(batch):
    """Collate ``(image, mask, boxes, meta)`` tuples, keeping GT as lists."""
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    boxes = [b[2] for b in batch]
    metas = [b[3] for b in batch]
    return images, masks, boxes, metas


def make_loader(
    ds, batch_size: int, shuffle: bool, num_workers: int = 0, seed: int = 0
) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=False,
        generator=gen if shuffle else None,
    )


@torch.no_grad()
def warmup(model, loader, device: torch.device, batches: int = 1) -> None:
    """Run a few forwards so device kernel compilation is not billed to timing."""
    model.eval()
    for i, (images, _, _, _) in enumerate(loader):
        model(images.to(device))
        if i + 1 >= batches:
            break


def compute_class_weights(cfg: Config) -> torch.Tensor | None:
    """Inverse-frequency class weights from the data manifest (segmentation)."""
    try:
        from aerial_vit.data.semantic_drone import load_manifest

        manifest = load_manifest(cfg=cfg)
        counts = np.array(
            [manifest["per_class_pixels"].get(c, 1) for c in cfg.data.classes], dtype=np.float64
        )
        counts = np.maximum(counts, 1.0)
        w = counts.sum() / (len(counts) * counts)
        w = np.clip(w, 0.2, 20.0)
        w = w / w.mean()
        log.info("class weights: %s", np.round(w, 3).tolist())
        return torch.tensor(w, dtype=torch.float32)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not compute class weights (%s); using uniform", exc)
        return None


# --------------------------------------------------------------------------- #
# one epoch
# --------------------------------------------------------------------------- #


def run_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    cfg: Config,
    scaler=None,
    class_weights: torch.Tensor | None = None,
    max_steps: int = 0,
    step_offset: int = 0,
) -> dict[str, float]:
    """One training epoch. Returns averaged loss components."""
    model.train()
    totals = {
        "loss": 0.0,
        "seg": 0.0,
        "dice": 0.0,
        "det": 0.0,
        "focal": 0.0,
        "giou": 0.0,
        "obj": 0.0,
        "bbox_l1": 0.0,
    }
    n = 0
    w = class_weights.to(device) if class_weights is not None else None

    for step, (images, masks, boxes, _) in enumerate(loader):
        if max_steps and (step_offset + step) >= max_steps:
            break
        images = images.to(device)
        masks = masks.to(device)
        boxes_d = [b.to(device) for b in boxes]
        labels_d = [b[:, 4].long() for b in boxes_d]
        boxes_xyxy = [b[:, :4] for b in boxes_d]

        use_amp = scaler is not None and device.type == "cuda"
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(images)
            seg = focal_ce_loss(
                out["seg_logits"],
                masks,
                weight=w,
                gamma=cfg.train.focal_gamma if cfg.train.focal_weight > 0 else 0.0,
                label_smoothing=cfg.train.label_smoothing,
            )
            dsc = dice_loss(out["seg_logits"], masks, cfg.data.num_classes)
            loss = seg + cfg.train.dice_weight * dsc
            det_parts = {"focal": 0.0, "giou": 0.0, "obj": 0.0, "bbox_l1": 0.0}
            if model.box_head is not None and "box" in out:
                targets = build_targets(out["box"], boxes_xyxy, labels_d, cfg.data.num_classes)
                det, det_parts = detection_loss(out["box"], targets, cfg.train)
                loss = loss + det

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            scaler.scale(loss).backward()
            if cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.train.grad_clip:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.train.grad_clip
                )
            optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["seg"] += float(seg.detach())
        totals["dice"] += float(dsc.detach())
        totals["det"] += float(sum(det_parts.values()) / max(1, len(det_parts)))
        totals["focal"] += det_parts["focal"]
        totals["giou"] += det_parts["giou"]
        totals["obj"] += det_parts["obj"]
        totals["bbox_l1"] += det_parts["bbox_l1"]
        n += 1

    return {k: v / max(1, n) for k, v in totals.items()}


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


@torch.no_grad()
def predict_dataset(
    model,
    loader,
    device: torch.device,
    cfg: Config,
    return_features: bool = False,
    return_probs: bool = False,
) -> dict[str, Any]:
    """Run the model over a loader; collect seg preds, detections and metadata."""
    model.eval()
    seg_preds, seg_gts = [], []
    det_preds, det_gts = [], []
    names, meta_list = [], []
    probs_all, feats_all, masks_all = [], [], []
    img_id = 0

    for images, masks, boxes, metas in loader:
        images_d = images.to(device)
        out = model(
            images_d,
            return_detections=model.box_head is not None,
            score_thresh=cfg.eval.score_thresh,
            nms_iou=cfg.eval.nms_iou,
            max_dets=cfg.eval.max_dets,
        )
        seg = out["seg_logits"].argmax(dim=1).cpu()
        for i in range(images.shape[0]):
            h, w = images.shape[-2:]
            seg_preds.append(seg[i].numpy())
            seg_gts.append(masks[i].numpy())
            names.append(metas[i].get("name", str(img_id)))
            meta_list.append({k: v for k, v in metas[i].items() if k != "landing_zone"})
            gt = boxes[i]
            det_gts.append(
                {
                    "image_id": img_id,
                    "width": int(w),
                    "height": int(h),
                    "boxes": gt[:, :4].numpy() if gt.numel() else np.zeros((0, 4)),
                    "labels": gt[:, 4].numpy().astype(int) if gt.numel() else np.zeros((0,), int),
                }
            )
            det = (
                out["detections"][i]
                if "detections" in out
                else {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "labels": np.zeros(0, int)}
            )
            det_preds.append(
                {
                    "image_id": img_id,
                    "boxes": det["boxes"].cpu().numpy()
                    if torch.is_tensor(det["boxes"])
                    else np.asarray(det["boxes"]),
                    "scores": det["scores"].cpu().numpy()
                    if torch.is_tensor(det["scores"])
                    else np.asarray(det["scores"]),
                    "labels": det["labels"].cpu().numpy().astype(int)
                    if torch.is_tensor(det["labels"])
                    else np.asarray(det["labels"]).astype(int),
                }
            )
            if return_probs:
                probs_all.append(out["seg_logits"][i].softmax(0).float().cpu())
            if return_features:
                feats_all.append(out["features"][i].mean(dim=(1, 2)).float().cpu())
            masks_all.append(masks[i].clone())
            img_id += 1

    result: dict[str, Any] = {
        "seg_preds": seg_preds,
        "seg_gts": seg_gts,
        "det_preds": det_preds,
        "det_gts": det_gts,
        "names": names,
        "metas": meta_list,
    }
    if return_probs:
        result["probs"] = probs_all
    if return_features:
        result["cls_features"] = feats_all
    result["masks"] = masks_all
    return result


def evaluate_model(
    model,
    loader,
    device: torch.device,
    cfg: Config,
    dump_dir: str | Path | None = None,
    prefix: str = "det",
    return_features: bool = False,
) -> dict[str, Any]:
    """Evaluate a model end to end and return a JSON-ready report."""
    p = predict_dataset(model, loader, device, cfg, return_features=return_features)
    report = evaluate_predictions(
        p["seg_preds"],
        p["seg_gts"],
        p["det_preds"],
        p["det_gts"],
        class_names=cfg.data.classes,
        landable_indices=cfg.data.landable_indices,
        num_classes=cfg.data.num_classes,
        rare_classes=None,
        dump_dir=dump_dir,
        prefix=prefix,
        max_dets=cfg.eval.max_dets,
    )
    report["split_size"] = len(p["seg_preds"])
    report["frame_names"] = p["names"]
    report["_raw"] = p
    return report


# --------------------------------------------------------------------------- #
# training driver
# --------------------------------------------------------------------------- #


def build_optimizer(model, cfg: Config, mode: str):
    """Parameter groups: separate LR for a trainable backbone in ``full`` mode."""
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith("backbone.") else head_params).append(p)
    groups = [{"params": head_params, "lr": cfg.train.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.train.backbone_lr})
    opt = torch.optim.AdamW(groups, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    return opt


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int, epochs: int):
    total = max(1, steps_per_epoch * epochs)
    warmup = max(1, int(total * cfg.train.warmup_frac))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(
    model,
    train_loader,
    val_loader,
    cfg: Config,
    device: torch.device,
    run_dir: str | Path,
    mode: str,
    class_weights: torch.Tensor | None = None,
    max_steps: int = 0,
) -> dict[str, Any]:
    """Full training run with AMP, cosine LR, early stopping and JSONL logging."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    opt = build_optimizer(model, cfg, mode)
    sched = build_scheduler(opt, cfg, len(train_loader), cfg.train.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp) if device.type == "cuda" else None

    best_metric = -1.0
    best_epoch = -1
    patience = 0
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    peak_mem = 0.0

    for epoch in range(cfg.train.epochs):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        ep_t0 = time.perf_counter()
        losses = run_epoch(
            model, train_loader, opt, device, cfg, scaler, class_weights, max_steps=max_steps
        )
        sched.step()
        if device.type == "cuda":
            peak_mem = max(peak_mem, torch.cuda.max_memory_allocated(device) / 1e6)

        val = quick_val(model, val_loader, device, cfg)
        record = {
            "epoch": epoch,
            "lr": float(opt.param_groups[0]["lr"]),
            "epoch_seconds": round(time.perf_counter() - ep_t0, 3),
            **{f"train_{k}": round(v, 6) for k, v in losses.items()},
            **{f"val_{k}": round(v, 6) for k, v in val.items()},
        }
        append_jsonl(metrics_path, record)
        history.append(record)
        log.info(
            "epoch %d/%d loss=%.4f val_miou=%.4f val_lz_iou=%.4f (%.1fs)",
            epoch + 1,
            cfg.train.epochs,
            losses["loss"],
            val["miou"],
            val["landing_zone_iou"],
            record["epoch_seconds"],
        )

        score = val["miou"] + val["landing_zone_iou"]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val": val,
                    "config": cfg.to_dict(),
                    "mode": mode,
                },
                run_dir / "best.pt",
            )
        else:
            patience += 1
            if cfg.train.early_stop_patience and patience >= cfg.train.early_stop_patience:
                log.info("early stopping at epoch %d", epoch + 1)
                break

    wall = time.perf_counter() - t0
    summary = {
        "run_dir": str(run_dir),
        "mode": mode,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_score": best_metric,
        "train_seconds": round(wall, 3),
        "train_seconds_per_epoch": round(wall / max(1, len(history)), 3),
        "peak_memory_mb": round(peak_mem, 2),
        "device": device_name(device),
        "history": history,
        "checkpoint": str(run_dir / "best.pt"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


@torch.no_grad()
def quick_val(model, loader, device: torch.device, cfg: Config) -> dict[str, float]:
    """Cheap per-epoch validation: seg metrics + landing-zone IoU (no COCO)."""
    model.eval()
    cm = ConfusionMatrix(cfg.data.num_classes)
    lz = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for images, masks, _, _ in loader:
        out = model(images.to(device))
        pred = out["seg_logits"].argmax(dim=1).cpu()
        cm.update(pred.numpy(), masks.numpy())
        c = landing_zone_counts(pred.numpy(), masks.numpy(), cfg.data.landable_indices)
        for k in lz:
            lz[k] += c[k]
    m = seg_metrics_from_confusion(cm, cfg.data.classes)
    tp, fp, fn = lz["tp"], lz["fp"], lz["fn"]
    return {
        "miou": m["miou"],
        "pixel_accuracy": m["pixel_accuracy"],
        "landing_zone_iou": round(tp / max(1, tp + fp + fn), 6),
    }
