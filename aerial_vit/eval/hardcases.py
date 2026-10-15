"""Hard-case mining: scene statistics, clustering, and failure-mode auto-tagging.

The point of this module is to turn "the model is worse on some frames" into a
*named* failure mode with a measurable definition.  Every tag is computed from
image statistics or dataset metadata, never from eyeballing:

===================  ==================================================
tag                  measurement
===================  ==================================================
``glare``            fraction of pixels with a saturated channel (>= 250)
``low-contrast/haze``  luminance std after normalising by the split median
``occlusion``        mean pairwise intersection-over-area of GT boxes
``small-object``     median GT instance area (px)
``altitude-shift``   median GT instance area relative to the split median
``dense-objects``    GT instances per 10k px
``boundary-ambiguity`` low model confidence at GT object boundaries
===================  ==================================================
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "STAT_NAMES",
    "TAG_DIRECTIONS",
    "scene_statistics",
    "scene_statistics_from_dataset",
    "cluster_embeddings",
    "tag_clusters",
    "cluster_metric_table",
]

STAT_NAMES = [
    "glare",
    "contrast",
    "occlusion",
    "median_instance_area",
    "instance_density",
    "altitude_ratio",
    "boundary_ambiguity",
]

# (tag label, statistic, "high" means the label fires when the cluster is above
# the split median)  -- a lookup table, not a heuristic guess per cluster.
TAG_DIRECTIONS: list[tuple[str, str, bool]] = [
    ("glare", "glare", True),
    ("low-contrast/haze", "contrast", False),
    ("occlusion", "occlusion", True),
    ("small-object", "median_instance_area", False),
    ("altitude-shift (zoomed-out)", "altitude_ratio", False),
    ("altitude-shift (zoomed-in)", "altitude_ratio", True),
    ("dense-objects", "instance_density", True),
    ("boundary-ambiguity", "boundary_ambiguity", True),
]


# --------------------------------------------------------------------------- #
# scene statistics
# --------------------------------------------------------------------------- #


def scene_statistics(
    image_rgb: np.ndarray,
    boxes: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    probs: np.ndarray | None = None,
) -> dict[str, float]:
    """Measurable descriptors of one frame (all optional-but-composable)."""
    img = np.asarray(image_rgb)
    h, w = img.shape[:2]
    flat = img.reshape(-1, img.shape[2]) if img.ndim == 3 else img.reshape(-1, 1)

    # glare: any channel saturated
    glare = float((flat.max(axis=1) >= 250).mean())

    # contrast: luminance std, normalised to [0,1] by 255
    if img.ndim == 3:
        lum = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float64)
    else:
        lum = img.astype(np.float64)
    contrast = float(lum.std() / 255.0)

    boxes = (
        np.zeros((0, 4), np.float32)
        if boxes is None
        else np.asarray(boxes, np.float32).reshape(-1, 4)
    )
    if boxes.shape[0]:
        wh = np.clip(boxes[:, 2:] - boxes[:, :2], 0, None)
        areas = wh[:, 0] * wh[:, 1]
        median_area = float(np.median(areas))
        density = float(len(boxes) / (h * w / 1e4))
        occlusion = _mean_overlap(boxes)
    else:
        median_area = 0.0
        density = 0.0
        occlusion = 0.0

    boundary = 0.0
    if probs is not None and mask is not None:
        boundary = _boundary_ambiguity(np.asarray(probs), np.asarray(mask))

    return {
        "glare": glare,
        "contrast": contrast,
        "occlusion": occlusion,
        "median_instance_area": median_area,
        "instance_density": density,
        "altitude_ratio": 1.0,  # filled in by the caller, relative to the split
        "boundary_ambiguity": boundary,
        "n_instances": float(len(boxes)),
    }


def _mean_overlap(boxes: np.ndarray) -> float:
    """Mean intersection-over-area of every GT box with its most-overlapping peer."""
    from aerial_vit.eval.metrics import box_iou as _iou

    if len(boxes) < 2:
        return 0.0
    iou = _iou(boxes, boxes).astype(np.float64)
    np.fill_diagonal(iou, 0.0)
    area_a = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    inter = iou * (area_a[:, None] + area_a[None, :]) / np.clip(1.0 + iou, 1e-9, None)
    # intersection over the *smaller* area == degree to which one occludes another
    small = np.minimum(area_a[:, None], area_a[None, :])
    ioa = inter / np.clip(small, 1e-9, None)
    np.fill_diagonal(ioa, 0.0)
    return 0.0 if ioa.size == 0 else float(np.clip(ioa.max(axis=1), 0, 1).mean())


def _boundary_ambiguity(probs: np.ndarray, mask: np.ndarray) -> float:
    """Mean 1-confidence on GT-edge pixels (where mistakes actually matter)."""
    import cv2

    h, w = mask.shape
    # an edge pixel is one whose 3x3 neighbourhood contains a different label
    edges = np.zeros((h, w), np.uint8)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
        edges |= (shifted != mask).astype(np.uint8)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    if not edges.any():
        return 0.0
    conf = probs.max(axis=0)
    return float(1.0 - conf[edges].mean())


def scene_statistics_from_dataset(
    ds,
    indices: Sequence[int] | None = None,
    probs: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    """Compute statistics for a whole split, then resolve ``altitude_ratio``.

    ``altitude_ratio`` is defined *relative to the split median* instance area,
    so it measures apparent scale change (a proxy for ground-sample distance)
    rather than absolute object size.
    """
    from aerial_vit.data.transforms import denormalize

    idx = list(range(len(ds))) if indices is None else list(indices)
    rows: list[dict[str, Any]] = []
    for i in idx:
        image, mask, boxes, meta = ds[i]
        rgb = denormalize(image)
        probs_i = probs.get(meta["name"]) if probs else None
        stats = scene_statistics(rgb, boxes[:, :4].numpy(), mask.numpy(), probs_i)
        stats["name"] = meta["name"]
        stats["split"] = meta.get("split", "")
        stats["n_gt_boxes"] = int(boxes.shape[0])
        stats["classes_present"] = meta.get("classes_present", [])
        rows.append(stats)

    areas = np.array([r["median_instance_area"] for r in rows], np.float64)
    pos = areas[areas > 0]
    med = float(np.median(pos)) if pos.size else 1.0
    for r in rows:
        r["altitude_ratio"] = float(r["median_instance_area"] / med) if med > 0 else 1.0
    return rows


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #


def cluster_embeddings(
    embeddings: np.ndarray,
    k_range: tuple[int, int] = (2, 6),
    seed: int = 1337,
) -> dict[str, Any]:
    """KMeans with the ``k`` chosen by silhouette score (deterministic)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    x = np.asarray(embeddings, np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return {"labels": np.zeros(0, np.int64), "k": 0, "silhouette": None, "k_scores": {}}

    lo, hi = int(k_range[0]), int(k_range[1])
    n = x.shape[0]
    hi = max(lo, min(hi, max(2, n - 1)))
    if n < 3 or lo >= n:
        return {"labels": np.zeros(n, np.int64), "k": 1, "silhouette": None, "k_scores": {}}

    k_scores: dict[int, float] = {}
    best = None
    for k in range(max(2, lo), hi + 1):
        if k >= n:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(x)
        if len(set(labels)) < 2:
            continue
        s = float(silhouette_score(x, labels))
        k_scores[k] = round(s, 6)
        if best is None or s > best[1]:
            best = (k, s, labels)
    if best is None:
        return {"labels": np.zeros(n, np.int64), "k": 1, "silhouette": None, "k_scores": {}}
    return {
        "labels": np.asarray(best[2], np.int64),
        "k": int(best[0]),
        "silhouette": round(float(best[1]), 6),
        "k_scores": k_scores,
    }


# --------------------------------------------------------------------------- #
# auto-tagging
# --------------------------------------------------------------------------- #


def tag_clusters(
    stats: list[dict[str, Any]],
    labels: np.ndarray,
    split_stats: dict[str, float] | None = None,
    top_n: int = 2,
) -> dict[int, dict[str, Any]]:
    """Give every cluster a tag set derived from its most extreme statistics.

    Returns ``{cluster_id: {"tags": [...], "z_scores": {...}, "n": int, "means": {...}}}``.
    Every cluster gets exactly one entry, and every entry gets a tag set.
    """
    labels = np.asarray(labels).reshape(-1)
    split = split_stats or _split_stats(stats)
    out: dict[int, dict[str, Any]] = {}

    for cid in sorted({int(v) for v in labels}) if len(labels) else []:
        members = [stats[i] for i in range(len(stats)) if i < len(labels) and int(labels[i]) == cid]
        if not members:
            continue
        means = {s: float(np.mean([m.get(s, 0.0) for m in members])) for s in STAT_NAMES}
        stds = {s: float(np.std([m.get(s, 0.0) for m in members])) for s in STAT_NAMES}

        z: dict[str, float] = {}
        for s in STAT_NAMES:
            base = split.get(s, 0.0)
            spread = split.get(f"{s}__std", 0.0)
            if spread <= 1e-12:
                z[s] = 0.0
            else:
                z[s] = float((means[s] - base) / spread)

        ranked = sorted(TAG_DIRECTIONS, key=lambda t: -abs(z.get(t[1], 0.0)))
        tags: list[str] = []
        for label, stat, high in ranked:
            zz = z.get(stat, 0.0)
            if (high and zz > 0) or ((not high) and zz < 0):
                tags.append(label)
            if len(tags) >= top_n:
                break
        if not tags:
            tags = ["within-normal-range"]

        out[cid] = {
            "cluster": cid,
            "n_frames": len(members),
            "tags": tags,
            "tag": tags[0],  # primary tag, for compact tables
            "z_scores": {k: round(v, 4) for k, v in z.items()},
            "means": {k: round(v, 6) for k, v in means.items()},
            "stds": {k: round(v, 6) for k, v in stds.items()},
            "split_means": {k: round(float(split.get(k, 0.0)), 6) for k in STAT_NAMES},
        }
    return out


def _split_stats(stats: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for s in STAT_NAMES:
        vals = [float(r.get(s, 0.0)) for r in stats]
        out[s] = float(np.mean(vals)) if vals else 0.0
        out[f"{s}__std"] = float(np.std(vals)) if vals else 0.0
    return out


# --------------------------------------------------------------------------- #
# per-cluster metrics
# --------------------------------------------------------------------------- #


def cluster_metric_table(
    cluster_info: dict[int, dict[str, Any]],
    per_frame: dict[str, dict[str, float]],
    split_metrics: dict[str, float],
    metric_keys: Sequence[str] = ("miou", "map_50", "landing_zone_iou"),
) -> list[dict[str, Any]]:
    """Per-cluster mean metrics next to the split average, plus the delta.

    ``per_frame`` maps frame name -> {"miou": ..., "map_50": ...}; frames with a
    missing metric are skipped rather than imputed.
    """
    rows: list[dict[str, Any]] = []
    for cid, info in sorted(cluster_info.items()):
        # "test_frames" are the held-out frames assigned to this cluster;
        # "frames" are the mined hard validation frames that *defined* it.
        names = info.get("test_frames") or info.get("frames", [])
        row: dict[str, Any] = {
            "cluster": cid,
            "tag": info.get("tag", ""),
            "tags": ";".join(info.get("tags", [])),
            "n_frames": len(names) if names else info.get("n_frames", 0),
        }
        for k in metric_keys:
            vals = [
                per_frame[n][k] for n in names if n in per_frame and per_frame[n].get(k) is not None
            ]
            mean = float(np.mean(vals)) if vals else None
            base = split_metrics.get(k)
            row[k] = None if mean is None else round(mean, 6)
            row[f"split_{k}"] = None if base is None else round(float(base), 6)
            row[f"delta_{k}"] = (
                None if (mean is None or base is None) else round(mean - float(base), 6)
            )
        rows.append(row)
    return rows
