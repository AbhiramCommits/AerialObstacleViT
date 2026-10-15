"""Geometric + photometric transforms that keep ``image``, ``mask`` and
``boxes`` consistent.

A sample is a plain dict::

    {
      "image":  np.ndarray (H, W, 3) uint8   RGB
      "mask":   np.ndarray (H, W)    uint8   canonical class indices
      "boxes":  np.ndarray (N, 4)    float32 pixel xyxy
      "labels": np.ndarray (N,)      int64   canonical class indices
    }

Transforms are functions ``fn(sample, rng) -> sample`` so the random draw is
explicit and seeded, which is what makes the deterministic tests meaningful.
The output sample replaces ``image`` with a normalised ``torch.Tensor`` (3, H, W).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Sample = dict[str, Any]


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #


def _resize_image(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)


def resize_sample(sample: Sample, size: tuple[int, int]) -> Sample:
    """Resize image (bilinear) and mask (nearest), rescaling boxes."""
    h0, w0 = sample["image"].shape[:2]
    h1, w1 = size
    sx, sy = w1 / w0, h1 / h0
    out = dict(sample)
    out["image"] = _resize_image(sample["image"], size)
    out["mask"] = _resize_mask(sample["mask"], size)
    boxes = np.asarray(sample.get("boxes", np.zeros((0, 4), np.float32)), np.float32).copy()
    if boxes.size:
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
    out["boxes"] = boxes
    return out


def _crop_pad(img: np.ndarray, mask: np.ndarray, boxes: np.ndarray, box: tuple[int, int, int, int]):
    """Crop ``box=(x0,y0,x1,y1)`` allowing out-of-bounds (pads with reflect/2)."""
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    pw_l, pt = max(0, -x0), max(0, -y0)
    pw_r, pb = max(0, x1 - w), max(0, y1 - h)
    if pw_l or pw_r or pt or pb:
        img = cv2.copyMakeBorder(img, pt, pb, pw_l, pw_r, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, pt, pb, pw_l, pw_r, cv2.BORDER_CONSTANT, value=2)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    img = img[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]
    if boxes.size:
        boxes = boxes.copy()
        boxes[:, [0, 2]] -= x0
        boxes[:, [1, 3]] -= y0
    return img, mask, boxes


def _random_scale_crop(sample: Sample, rng: np.random.Generator, scale_range, target) -> Sample:
    """Zoom by ``s`` then random-crop (or centre-crop) to ``target``."""
    h, w = sample["image"].shape[:2]
    s = float(rng.uniform(*scale_range))
    nh, nw = max(target[0], int(round(h * s))), max(target[1], int(round(w * s)))
    img = _resize_image(sample["image"], (nh, nw))
    mask = _resize_mask(sample["mask"], (nh, nw))
    boxes = np.asarray(sample.get("boxes", np.zeros((0, 4), np.float32)), np.float32).copy()
    if boxes.size:
        boxes[:, [0, 2]] *= nw / w
        boxes[:, [1, 3]] *= nh / h
    th, tw = target
    x0 = int(rng.integers(0, max(1, nw - tw + 1)))
    y0 = int(rng.integers(0, max(1, nh - th + 1)))
    img, mask, boxes = _crop_pad(img, mask, boxes, (x0, y0, x0 + tw, y0 + th))
    out = dict(sample)
    out.update(image=img, mask=mask, boxes=boxes)
    return out


def _flip(sample: Sample, horizontal: bool) -> Sample:
    out = dict(sample)
    img = out["image"][:, ::-1] if horizontal else out["image"][::-1]
    mask = out["mask"][:, ::-1] if horizontal else out["mask"][::-1]
    boxes = np.asarray(out.get("boxes", np.zeros((0, 4), np.float32)), np.float32).copy()
    if boxes.size:
        if horizontal:
            w = img.shape[1]
            x0 = boxes[:, 0].copy()
            boxes[:, 0] = w - boxes[:, 2]
            boxes[:, 2] = w - x0
        else:
            h = img.shape[0]
            y0 = boxes[:, 1].copy()
            boxes[:, 1] = h - boxes[:, 3]
            boxes[:, 3] = h - y0
    out.update(image=np.ascontiguousarray(img), mask=np.ascontiguousarray(mask), boxes=boxes)
    return out


def _colour_jitter(sample: Sample, rng: np.random.Generator) -> Sample:
    img = sample["image"].astype(np.float32)
    # brightness
    img *= float(rng.uniform(0.75, 1.3))
    # contrast
    mean = img.mean()
    img = (img - mean) * float(rng.uniform(0.75, 1.3)) + mean
    img = np.clip(img, 0, 255)
    # saturation
    gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)[..., None]
    img = gray + (img - gray) * float(rng.uniform(0.6, 1.4))
    img = np.clip(img, 0, 255)
    # sensor noise
    img += rng.normal(0, float(rng.uniform(0, 3.0)), img.shape)
    out = dict(sample)
    out["image"] = np.clip(img, 0, 255).astype(np.uint8)
    return out


def to_tensor(sample: Sample) -> Sample:
    """HWC uint8 RGB -> normalised CHW float tensor; mask -> int64 tensor."""
    from torchvision.transforms import functional as TF

    img = torch.from_numpy(np.ascontiguousarray(sample["image"])).permute(2, 0, 1).float() / 255.0
    img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)
    out = dict(sample)
    out["image"] = img
    out["mask"] = torch.from_numpy(np.ascontiguousarray(sample["mask"])).long()
    out["boxes"] = torch.from_numpy(
        np.asarray(sample.get("boxes", np.zeros((0, 4), np.float32)), np.float32)
    )
    out["labels"] = torch.from_numpy(
        np.asarray(sample.get("labels", np.zeros((0,), np.int64)), np.int64)
    )
    return out


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Invert ImageNet normalisation; returns HWC uint8 RGB numpy."""
    t = tensor.detach().float().cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    t = (t * std + mean).clamp(0, 1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _drop_degenerate_boxes(sample: Sample, min_area: float = 6.0) -> Sample:
    boxes = np.asarray(sample.get("boxes", np.zeros((0, 4), np.float32)), np.float32)
    labels = np.asarray(sample.get("labels", np.zeros((0,), np.int64)), np.int64)
    if boxes.size == 0:
        return sample
    h, w = sample["mask"].shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    wh = boxes[:, 2:] - boxes[:, :2]
    keep = (wh[:, 0] > 1) & (wh[:, 1] > 1) & (wh.prod(1) >= min_area)
    out = dict(sample)
    out["boxes"] = boxes[keep]
    out["labels"] = labels[keep]
    return out


# --------------------------------------------------------------------------- #
# public builders
# --------------------------------------------------------------------------- #


def build_train_transform(
    image_size: int,
    augment: bool = True,
    scale_range: tuple[float, float] = (0.75, 1.35),
):
    """Return ``fn(sample, rng) -> tensorised sample`` for training."""

    def _fn(sample: Sample, rng: np.random.Generator) -> Sample:
        s = resize_sample(sample, (image_size, image_size)) if not augment else _random_scale_crop(
            sample, rng, scale_range, (image_size, image_size)
        )
        if augment:
            if rng.random() < 0.5:
                s = _flip(s, horizontal=True)
            if rng.random() < 0.2:
                s = _flip(s, horizontal=False)
            if rng.random() < 0.7:
                s = _colour_jitter(s, rng)
        s = _drop_degenerate_boxes(s)
        return to_tensor(s)

    return _fn


def build_eval_transform(image_size: int):
    """Deterministic resize + normalise for evaluation."""

    def _fn(sample: Sample, rng: np.random.Generator | None = None) -> Sample:
        s = resize_sample(sample, (image_size, image_size))
        s = _drop_degenerate_boxes(s, min_area=1.0)
        return to_tensor(s)

    return _fn
