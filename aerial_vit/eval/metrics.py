"""Segmentation, landing-zone and detection metrics.

Three groups:

* **Segmentation** -- a :class:`ConfusionMatrix` accumulator giving mIoU, per
  class IoU, pixel accuracy and frequency-weighted IoU.
* **Landing zone** -- IoU / F1 / precision / recall for the derived binary
  "flat + clear = landable" mask, which is the metric the mission actually cares
  about (a missed tree in a landing zone is a crash, not a 1 % mIoU drop).
* **Detection** -- COCO mAP@0.5 and mAP@[.5:.95] through ``pycocotools``, with
  the COCO-format ground truth and predictions written to disk as a side effect.

All metric functions are pure and cheap to unit test on hand-built toy inputs.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

__all__ = [
    "ConfusionMatrix",
    "box_iou",
    "coco_map",
    "evaluate_predictions",
    "landing_zone_counts",
    "landing_zone_metrics",
    "pixel_accuracy",
    "seg_metrics_from_confusion",
]


# --------------------------------------------------------------------------- #
# segmentation
# --------------------------------------------------------------------------- #


@dataclass
class ConfusionMatrix:
    """Accumulates a ``(K, K)`` confusion matrix over arbitrarily many batches."""

    num_classes: int
    matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.matrix = np.zeros((self.num_classes, self.num_classes), np.int64)

    def update(self, pred: np.ndarray | torch.Tensor, target: np.ndarray | torch.Tensor) -> None:
        pred = _to_numpy(pred).reshape(-1).astype(np.int64)
        target = _to_numpy(target).reshape(-1).astype(np.int64)
        valid = (target >= 0) & (target < self.num_classes)
        pred = np.clip(pred[valid], 0, self.num_classes - 1)
        target = target[valid]
        idx = target * self.num_classes + pred
        self.matrix += np.bincount(idx, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def merge(self, other: ConfusionMatrix) -> None:
        self.matrix += other.matrix

    def reset(self) -> None:
        self.matrix[:] = 0

    # -- derived ----------------------------------------------------------- #
    @property
    def total(self) -> int:
        return int(self.matrix.sum())

    def iou_per_class(self) -> np.ndarray:
        """IoU per class; NaN where the class has no GT and no prediction."""
        m = self.matrix
        tp = np.diag(m).astype(np.float64)
        fp = m.sum(axis=0) - np.diag(m)
        fn = m.sum(axis=1) - np.diag(m)
        denom = tp + fp + fn
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(denom > 0, tp / denom, np.nan)
        return iou

    def accuracy(self) -> float:
        return float(np.diag(self.matrix).sum() / max(1, self.total))

    def present_classes(self) -> list[int]:
        present = (self.matrix.sum(axis=1) > 0) | (self.matrix.sum(axis=0) > 0)
        return [int(i) for i in np.nonzero(present)[0]]


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def seg_metrics_from_confusion(
    cm: ConfusionMatrix, class_names: Sequence[str] | None = None
) -> dict[str, Any]:
    """mIoU / per-class IoU / pixel accuracy / frequency-weighted IoU."""
    iou = cm.iou_per_class()
    present = cm.present_classes()
    valid = iou[present] if present else np.array([])
    miou = float(np.nanmean(valid)) if valid.size else 0.0

    # frequency-weighted IoU: weight by GT frequency
    freq = cm.matrix.sum(axis=1).astype(np.float64)
    freq = freq / max(1.0, freq.sum())
    fwiou = float(np.nansum(np.where(np.isnan(iou), 0.0, iou) * freq))

    names = (
        list(class_names) if class_names is not None else [str(i) for i in range(cm.num_classes)]
    )
    per_class = {
        names[i]: (None if np.isnan(iou[i]) else round(float(iou[i]), 6))
        for i in range(cm.num_classes)
    }
    return {
        "miou": round(miou, 6),
        "pixel_accuracy": round(cm.accuracy(), 6),
        "frequency_weighted_iou": round(fwiou, 6),
        "per_class_iou": per_class,
        "classes_in_eval": len(present),
        "n_pixels": cm.total,
        "confusion_matrix": cm.matrix.tolist(),
        "class_names": names,
    }


def pixel_accuracy(pred, target) -> float:
    cm = ConfusionMatrix(int(max(_to_numpy(target).max(), _to_numpy(pred).max())) + 1)
    cm.update(pred, target)
    return cm.accuracy()


# --------------------------------------------------------------------------- #
# landing zone
# --------------------------------------------------------------------------- #


def landing_zone_counts(pred_mask, gt_mask, landable_indices: Sequence[int]) -> dict[str, int]:
    """TP / FP / FN / TN counts for the binary landable mask."""
    land = {int(i) for i in landable_indices}
    p = np.isin(_to_numpy(pred_mask), list(land))
    g = np.isin(_to_numpy(gt_mask), list(land))
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def landing_zone_metrics(
    pred_mask, gt_mask, landable_indices: Sequence[int], counts: dict[str, int] | None = None
) -> dict[str, float]:
    """IoU / Dice-F1 / precision / recall of the binary landing-zone mask."""
    c = counts or landing_zone_counts(pred_mask, gt_mask, landable_indices)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    iou = tp / max(1, tp + fp + fn)
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    acc = (tp + c["tn"]) / max(1, tp + fp + fn + c["tn"])
    return {
        "landing_zone_iou": round(float(iou), 6),
        "landing_zone_f1": round(float(f1), 6),
        "landing_zone_precision": round(float(prec), 6),
        "landing_zone_recall": round(float(rec), 6),
        "landing_zone_accuracy": round(float(acc), 6),
        **{f"landing_zone_{k}": int(v) for k, v in c.items()},
    }


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #


def box_iou(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> np.ndarray:
    """Pairwise IoU, xyxy, numpy in / ``(N, M)`` out."""
    a = _to_numpy(a).astype(np.float64).reshape(-1, 4)
    b = _to_numpy(b).astype(np.float64).reshape(-1, 4)
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-9, None)


def _coco_gt_dict(
    gts: Sequence[dict[str, Any]], class_names: Sequence[str], num_classes: int
) -> dict[str, Any]:
    images, annotations, categories = [], [], []
    ann_id = 1
    for img in gts:
        images.append(
            {"id": int(img["image_id"]), "width": int(img["width"]), "height": int(img["height"])}
        )
        for b, lab in zip(
            _to_numpy(img["boxes"]).reshape(-1, 4),
            _to_numpy(img["labels"]).reshape(-1),
            strict=False,
        ):
            x0, y0, x1, y1 = [float(v) for v in b]
            w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": int(img["image_id"]),
                    "category_id": int(lab),
                    "bbox": [x0, y0, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    for c in range(num_classes):
        categories.append({"id": c, "name": class_names[c] if c < len(class_names) else str(c)})
    return {"images": images, "annotations": annotations, "categories": categories}


def _coco_pred_list(preds: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for img in preds:
        boxes = _to_numpy(img["boxes"]).reshape(-1, 4)
        scores = _to_numpy(img.get("scores", np.ones(len(boxes)))).reshape(-1)
        labels = _to_numpy(img["labels"]).reshape(-1)
        for b, s, lab in zip(boxes, scores, labels, strict=False):
            x0, y0, x1, y1 = [float(v) for v in b]
            w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
            if w <= 0 or h <= 0:
                continue
            out.append(
                {
                    "image_id": int(img["image_id"]),
                    "category_id": int(lab),
                    "bbox": [x0, y0, w, h],
                    "score": float(s),
                }
            )
    return out


def coco_map(
    preds: Sequence[dict[str, Any]],
    gts: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    num_classes: int | None = None,
    eval_classes: Sequence[int] | None = None,
    dump_dir: str | Path | None = None,
    prefix: str = "det",
    max_dets: int = 100,
) -> dict[str, Any]:
    """COCO mAP@0.5 / mAP@[.5:.95], plus per-class AP@0.5.

    ``eval_classes`` restricts the *reporting* to a subset (used for the
    rare-class comparison) while the evaluation set itself stays unchanged.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    num_classes = num_classes or len(class_names)
    classes = (
        list(range(num_classes)) if eval_classes is None else sorted({int(c) for c in eval_classes})
    )

    gt_dict = _coco_gt_dict(gts, class_names, num_classes)
    pred_list = _coco_pred_list(preds)

    tmpdir = Path(tempfile.mkdtemp(prefix="aerialvit_coco_"))
    gt_path = tmpdir / f"{prefix}_gt.json"
    pred_path = tmpdir / f"{prefix}_pred.json"
    gt_path.write_text(json.dumps(gt_dict))
    pred_path.write_text(json.dumps(pred_list))

    if dump_dir is not None:
        d = Path(dump_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{prefix}_gt.json").write_text(json.dumps(gt_dict, indent=1))
        (d / f"{prefix}_pred.json").write_text(json.dumps(pred_list, indent=1))

    result: dict[str, Any] = {
        "map": 0.0,
        "map_50": 0.0,
        "map_75": 0.0,
        "per_class_ap50": {
            class_names[c] if c < len(class_names) else str(c): None for c in range(num_classes)
        },
        "n_gt": len(gt_dict["annotations"]),
        "n_pred": len(pred_list),
        "n_images": len(gt_dict["images"]),
        "eval_classes": [class_names[c] if c < len(class_names) else str(c) for c in classes],
    }

    n_gt_per_cat = dict.fromkeys(range(num_classes), 0)
    for a in gt_dict["annotations"]:
        n_gt_per_cat[a["category_id"]] = n_gt_per_cat.get(a["category_id"], 0) + 1
    result["n_gt_per_class"] = {
        (class_names[c] if c < len(class_names) else str(c)): int(n_gt_per_cat.get(c, 0))
        for c in range(num_classes)
    }

    if not pred_list or not gt_dict["annotations"]:
        return result

    with contextlib.suppress(Exception):
        coco_gt = COCO(str(gt_path))
        coco_dt = coco_gt.loadRes(str(pred_path))
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.maxDets = [1, 10, max_dets]
        ev.evaluate()
        ev.accumulate()

        # restrict summarisation to the requested categories
        precision = ev.eval["precision"]  # (T, R, K, A, M) for k in sorted catIds
        cat_ids = sorted(coco_gt.getCatIds())
        idx_of = {c: i for i, c in enumerate(cat_ids)}
        sel = [idx_of[c] for c in classes if c in idx_of]
        sub = precision[:, :, sel, 0, -1] if sel else precision[:, :, :, 0, -1]

        def _ap(t_idx: int) -> float:
            p = sub[t_idx] if sub.ndim == 3 else sub
            p = p[p > -1]
            return float(p.mean()) if p.size else 0.0

        # T = 10 IoU thresholds 0.5:0.05:0.95
        result["map_50"] = round(_ap(0), 6)
        result["map_75"] = round(_ap(5), 6)
        result["map"] = round(float(np.mean([_ap(t) for t in range(sub.shape[0])])), 6)

        for c in range(num_classes):
            key = class_names[c] if c < len(class_names) else str(c)
            if c in idx_of:
                p = precision[0, :, idx_of[c], 0, -1]
                p = p[p > -1]
                result["per_class_ap50"][key] = round(float(p.mean()), 6) if p.size else None
            else:
                result["per_class_ap50"][key] = None

    with contextlib.suppress(Exception):
        gt_path.unlink()
        pred_path.unlink()
        tmpdir.rmdir()

    return result


# --------------------------------------------------------------------------- #
# convenience bundle
# --------------------------------------------------------------------------- #


def evaluate_predictions(
    seg_preds: Sequence[np.ndarray],
    seg_gts: Sequence[np.ndarray],
    det_preds: Sequence[dict[str, Any]],
    det_gts: Sequence[dict[str, Any]],
    class_names: Sequence[str],
    landable_indices: Sequence[int],
    num_classes: int | None = None,
    rare_classes: Sequence[str] | None = None,
    dump_dir: str | Path | None = None,
    prefix: str = "det",
    max_dets: int = 100,
) -> dict[str, Any]:
    """Full evaluation bundle for one method on one split."""
    num_classes = num_classes or len(class_names)
    cm = ConfusionMatrix(num_classes)
    lz = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for p, g in zip(seg_preds, seg_gts, strict=False):
        cm.update(p, g)
        c = landing_zone_counts(p, g, landable_indices)
        for k in lz:
            lz[k] += c[k]

    report: dict[str, Any] = {}
    report.update(seg_metrics_from_confusion(cm, class_names))
    report.update(landing_zone_metrics(None, None, landable_indices, counts=lz))

    det = coco_map(
        det_preds,
        det_gts,
        class_names,
        num_classes=num_classes,
        dump_dir=dump_dir,
        prefix=prefix,
        max_dets=max_dets,
    )
    report.update({f"det_{k}": v for k, v in det.items() if not isinstance(v, dict)})
    report["det_per_class_ap50"] = det["per_class_ap50"]
    report["det_n_gt_per_class"] = det["n_gt_per_class"]

    if rare_classes:
        rare_ids = [class_names.index(c) for c in rare_classes if c in list(class_names)]
        rare = coco_map(
            det_preds,
            det_gts,
            class_names,
            num_classes=num_classes,
            eval_classes=rare_ids,
            prefix=f"{prefix}_rare",
            max_dets=max_dets,
        )
        report["rare_classes"] = list(rare_classes)
        report["rare_map"] = rare["map"]
        report["rare_map_50"] = rare["map_50"]
        report["rare_per_class_ap50"] = {
            k: v for k, v in rare["per_class_ap50"].items() if k in set(rare_classes)
        }
    else:
        report["rare_classes"] = []
        report["rare_map"] = None
        report["rare_map_50"] = None
        report["rare_per_class_ap50"] = {}

    return report
