"""Data layer: dataset loading, taxonomy remapping, transforms, synthetic fallback."""

from __future__ import annotations

from aerial_vit.data.semantic_drone import (
    AerialSceneDataset,
    SegmentationSample,
    build_datasets,
    build_manifest,
    load_manifest,
    manifest_path,
)
from aerial_vit.data.synthetic import SyntheticAerialGenerator, generate_synthetic_dataset
from aerial_vit.data.transforms import (
    build_eval_transform,
    build_train_transform,
    denormalize,
)

__all__ = [
    "AerialSceneDataset",
    "SegmentationSample",
    "SyntheticAerialGenerator",
    "build_datasets",
    "build_eval_transform",
    "build_manifest",
    "build_train_transform",
    "denormalize",
    "generate_synthetic_dataset",
    "load_manifest",
    "manifest_path",
]
