"""Dense box operations: IoU/GIoU, NMS, and FCOS-style encode/decode.

Everything here is pure ``torch`` and box-format-explicit; every function takes
``xyxy`` unless the name says otherwise.  These are the primitives shared by the
``BoxHead`` loss and the COCO mAP evaluator, so a bug here shows up in both.
"""

from __future__ import annotations

import torch

__all__ = [
    "box_iou",
    "giou_loss",
    "nms",
    "fcos_encode",
    "fcos_decode",
    "generate_locations",
]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU between ``a`` (N, 4) and ``b`` (M, 4), both xyxy."""
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros((a.shape[0], b.shape[0]), dtype=a.dtype, device=a.device)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / union.clamp(min=1e-7)


def giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - GIoU, elementwise, for matched (N, 4) xyxy pairs."""
    if pred.numel() == 0:
        return pred.sum() * 0.0
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = box_area(pred) + box_area(target) - inter
    iou = inter / union.clamp(min=1e-7)
    enclose_lt = torch.min(pred[:, :2], target[:, :2])
    enclose_rb = torch.max(pred[:, 2:], target[:, 2:])
    enclose = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = (enclose[:, 0] * enclose[:, 1]).clamp(min=1e-7)
    giou = iou - (enclose_area - union) / enclose_area
    return 1.0 - giou


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5) -> torch.Tensor:
    """Greedy NMS; returns kept indices, highest score first."""
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        i = int(order[0])
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i : i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# --------------------------------------------------------------------------- #
# FCOS style encode / decode
# --------------------------------------------------------------------------- #


def generate_locations(h: int, w: int, stride: float, device, dtype=torch.float32) -> torch.Tensor:
    """Cell centres in pixel space: (h*w, 2) xy."""
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * stride
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * stride
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def fcos_encode(
    locations: torch.Tensor,  # (L, 2) pixel xy
    stride: float,
    boxes: torch.Tensor,  # (N, 4) xyxy pixels
    labels: torch.Tensor,  # (N,)
    num_classes: int,
    center_sampling_radius: float = 1.5,
) -> dict[str, torch.Tensor]:
    """Assign every location to a GT box (FCOS) and produce regression targets.

    Returns dict with ``cls`` (L, C) 0/1 targets, ``reg`` (L, 4) ltrb in *pixel*
    units normalised by ``stride``, ``centerness`` (L,), and ``pos_mask`` (L,).
    """
    n_loc = locations.shape[0]
    cls_t = torch.zeros((n_loc, num_classes), dtype=torch.float32, device=locations.device)
    reg_t = torch.zeros((n_loc, 4), dtype=torch.float32, device=locations.device)
    centerness = torch.zeros((n_loc,), dtype=torch.float32, device=locations.device)
    pos = torch.zeros((n_loc,), dtype=torch.bool, device=locations.device)
    if boxes.numel() == 0:
        return {"cls": cls_t, "reg": reg_t, "centerness": centerness, "pos_mask": pos}

    lt = locations[:, None, :] - boxes[None, :, :2]  # (L, N, 2)
    rb = boxes[None, :, 2:] - locations[:, None, :]
    ltrb = torch.cat([lt, rb], dim=2)  # (L, N, 4)
    inside = ltrb.min(dim=2).values > 0

    areas = box_area(boxes)[None, :].expand(n_loc, -1)
    areas = torch.where(inside, areas, torch.full_like(areas, float("inf")))
    min_area, matched = areas.min(dim=1)

    # centre sampling: require the location to be near the box centre
    if center_sampling_radius > 0:
        radius = center_sampling_radius * stride
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        dx = (locations[:, 0][:, None] - cx[None, :]).abs()
        dy = (locations[:, 1][:, None] - cy[None, :]).abs()
        near = (dx < radius) & (dy < radius)
        valid = inside & near
        has_valid = valid.any(dim=1)
        # fall back to plain-inside when centre sampling kills everything
        areas2 = torch.where(
            valid, box_area(boxes)[None, :].expand(n_loc, -1), torch.full_like(areas, float("inf"))
        )
        matched = torch.where(has_valid, areas2.argmin(dim=1), matched)

    pos = min_area < float("inf")
    if pos.any():
        idx = torch.nonzero(pos, as_tuple=False).squeeze(1)
        m = matched[idx]
        cls_t[idx, labels[m].long()] = 1.0
        ltrb_m = ltrb[idx, m]
        reg_t[idx] = ltrb_m / stride
        lr = torch.minimum(ltrb_m[:, 0], ltrb_m[:, 2])
        tb = torch.minimum(ltrb_m[:, 1], ltrb_m[:, 3])
        lr_max = torch.maximum(ltrb_m[:, 0], ltrb_m[:, 2]).clamp(min=1e-6)
        tb_max = torch.maximum(ltrb_m[:, 1], ltrb_m[:, 3]).clamp(min=1e-6)
        centerness[idx] = torch.sqrt((lr / lr_max) * (tb / tb_max)).clamp(0, 1)

    return {"cls": cls_t, "reg": reg_t, "centerness": centerness, "pos_mask": pos}


def fcos_decode(
    locations: torch.Tensor,  # (L, 2) pixel xy
    stride: float,
    cls_logits: torch.Tensor,  # (L, C)
    reg_pred: torch.Tensor,  # (L, 4) ltrb in stride units
    obj_logits: torch.Tensor,  # (L, 1)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turn dense predictions into ``(boxes, scores, labels)``."""
    scores_all = cls_logits.sigmoid() * obj_logits.sigmoid()
    best_score, best_label = scores_all.max(dim=1)
    ltrb = reg_pred * stride
    boxes = torch.stack(
        [
            locations[:, 0] - ltrb[:, 0],
            locations[:, 1] - ltrb[:, 1],
            locations[:, 0] + ltrb[:, 2],
            locations[:, 1] + ltrb[:, 3],
        ],
        dim=1,
    )
    return boxes, best_score, best_label
