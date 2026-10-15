"""Evaluation: segmentation + detection metrics, uncertainty, clustering."""

from __future__ import annotations

from aerial_vit.eval.hardcases import (
    STAT_NAMES,
    cluster_embeddings,
    cluster_metric_table,
    scene_statistics,
    tag_clusters,
)
from aerial_vit.eval.metrics import (
    ConfusionMatrix,
    box_iou,
    coco_map,
    evaluate_predictions,
    landing_zone_counts,
    landing_zone_metrics,
    pixel_accuracy,
    seg_metrics_from_confusion,
)
from aerial_vit.eval.uncertainty import (
    frame_uncertainty,
    knn_ood_scores,
    margin,
    mc_dropout_stats,
    normalized_entropy,
    predictive_entropy,
    topk_uncertainty,
)

__all__ = [
    "ConfusionMatrix",
    "STAT_NAMES",
    "box_iou",
    "cluster_embeddings",
    "cluster_metric_table",
    "coco_map",
    "evaluate_predictions",
    "frame_uncertainty",
    "knn_ood_scores",
    "landing_zone_counts",
    "landing_zone_metrics",
    "margin",
    "mc_dropout_stats",
    "normalized_entropy",
    "pixel_accuracy",
    "predictive_entropy",
    "scene_statistics",
    "seg_metrics_from_confusion",
    "tag_clusters",
    "topk_uncertainty",
]
