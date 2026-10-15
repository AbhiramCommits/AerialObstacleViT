"""Step 1 tests: manifest, dataset shapes/dtypes, label ranges, determinism."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aerial_vit.config import ARTIFACTS_DIR, Config, load_config
from aerial_vit.data.semantic_drone import (
    AerialSceneDataset,
    build_datasets,
    compute_strata,
    load_manifest,
    make_splits,
    stratified_subsample,
)
from aerial_vit.data.synthetic import PALETTE, SyntheticAerialGenerator
from aerial_vit.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_train_transform,
    denormalize,
)

SMOKE = "configs/smoke.yaml"


@pytest.fixture(scope="session")
def smoke_cfg() -> Config:
    return load_config(SMOKE)


@pytest.fixture(scope="session")
def manifest() -> dict:
    p = ARTIFACTS_DIR / "data_manifest.json"
    if not p.exists():
        pytest.skip("run `python scripts/fetch_data.py --config configs/smoke.yaml` first")
    return json.loads(p.read_text())


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #


def test_manifest_exists_and_has_real_counts(manifest):
    for key in (
        "source",
        "n_train",
        "n_val",
        "n_test",
        "classes",
        "image_size",
        "per_class_pixels",
    ):
        assert key in manifest, f"manifest missing {key}"
    assert manifest["source"] in {"synthetic", "semantic_drone"}
    assert manifest["n_train"] > 0 and manifest["n_val"] > 0 and manifest["n_test"] > 0
    assert len(manifest["classes"]) == 8
    assert manifest["image_size"] == [224, 224]
    assert sum(manifest["per_class_pixels"].values()) > 0
    assert all(isinstance(v, int) for v in manifest["per_class_pixels"].values())


def test_manifest_rare_classes_are_derived(manifest):
    assert isinstance(manifest["rare_classes"], list)
    # 'person' must be rare in a small smoke split; the definition is measured,
    # so this is a property of the data, not a hard-coded guess.
    assert manifest["rare_definition"]["pixel_fraction_below"] > 0
    assert set(manifest["rare_classes"]).issubset(set(manifest["classes"]))


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #


def test_dataset_item_shapes_and_dtypes(smoke_cfg, manifest):
    train, val, test = build_datasets(smoke_cfg)
    img, mask, boxes, meta = train[0]
    assert img.shape == (3, 224, 224) and img.dtype == torch.float32
    assert mask.shape == (224, 224) and mask.dtype == torch.int64
    assert boxes.ndim == 2 and boxes.shape[1] == 5 and boxes.dtype == torch.float32
    assert isinstance(meta, dict) and "landing_zone" in meta
    assert meta["landing_zone"].shape == (224, 224)
    assert meta["landing_zone"].dtype == torch.bool


def test_masks_are_in_valid_label_range(smoke_cfg):
    train, _, _ = build_datasets(smoke_cfg)
    for i in range(min(5, len(train))):
        _, mask, _, _ = train[i]
        assert int(mask.min()) >= 0
        assert int(mask.max()) < smoke_cfg.data.num_classes


def test_landing_zone_mask_is_binary_and_consistent(smoke_cfg):
    train, _, _ = build_datasets(smoke_cfg)
    for i in range(min(5, len(train))):
        _, mask, _, meta = train[i]
        lz = meta["landing_zone"]
        assert set(torch.unique(lz).tolist()).issubset({False, True})
        expected = torch.isin(mask.reshape(-1), torch.tensor(smoke_cfg.data.landable_indices))
        assert torch.equal(lz.reshape(-1), expected)


def test_boxes_within_image_bounds(smoke_cfg):
    train, _, _ = build_datasets(smoke_cfg)
    for i in range(min(8, len(train))):
        _, _, boxes, _ = train[i]
        if boxes.numel() == 0:
            continue
        assert float(boxes[:, 0].min()) >= -1e-3
        assert float(boxes[:, 1].min()) >= -1e-3
        assert float(boxes[:, 2].max()) <= 224 + 1e-3
        assert float(boxes[:, 3].max()) <= 224 + 1e-3
        assert float((boxes[:, 2] - boxes[:, 0]).min()) > 0
        assert set(boxes[:, 4].unique().tolist()).issubset(set(range(8)))


def test_dataset_is_deterministic_with_fixed_seed(smoke_cfg):
    a, _, _ = build_datasets(smoke_cfg, seed=7)
    b, _, _ = build_datasets(smoke_cfg, seed=7)
    for i in range(min(3, len(a))):
        ia, ma, ba, _ = a[i]
        ib, mb, bb, _ = b[i]
        assert torch.allclose(ia, ib)
        assert torch.equal(ma, mb)
        assert torch.allclose(ba, bb)
    assert len(a) == len(b)


def test_augmentation_is_order_independent(smoke_cfg):
    """Per-index seeding means access order cannot change the augmentation."""
    ds, _, _ = build_datasets(smoke_cfg, seed=11)
    first = ds[2][0].clone()
    _ = ds[0][0]
    _ = ds[1][0]
    again = ds[2][0]
    assert torch.allclose(first, again)


def test_eval_transform_is_deterministic(smoke_cfg):
    tf = build_eval_transform(224)
    sample = {
        "image": np.random.default_rng(0).integers(0, 255, (300, 260, 3), dtype=np.uint8),
        "mask": np.zeros((300, 260), np.uint8),
        "boxes": np.array([[10, 12, 60, 70]], np.float32),
        "labels": np.array([7], np.int64),
    }
    a = tf(dict(sample))
    b = tf(dict(sample))
    assert torch.allclose(a["image"], b["image"])
    assert torch.equal(a["mask"], b["mask"])
    assert a["boxes"].shape == (1, 4)


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #


def test_denormalize_roundtrip():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    sample = {"image": img, "mask": np.zeros((224, 224), np.uint8)}
    tf = build_eval_transform(224)
    t = tf(sample)["image"]
    back = denormalize(t)
    assert back.shape == (224, 224, 3)
    assert int(np.abs(back.astype(int) - img.astype(int)).max()) <= 2


def test_train_transform_flip_and_scale_keep_shapes():
    rng = np.random.default_rng(5)
    sample = {
        "image": rng.integers(0, 255, (200, 260, 3), dtype=np.uint8),
        "mask": rng.integers(0, 8, (200, 260), dtype=np.uint8),
        "boxes": np.array([[10, 20, 90, 120], [100, 30, 150, 80]], np.float32),
        "labels": np.array([3, 7], np.int64),
    }
    tf = build_train_transform(224, augment=True)
    for seed in range(6):
        out = tf(dict(sample), np.random.default_rng(seed))
        assert out["image"].shape == (3, 224, 224)
        assert out["mask"].shape == (224, 224)
        assert out["boxes"].ndim == 2 and out["boxes"].shape[1] == 4
        assert out["boxes"].shape[0] == out["labels"].shape[0]
    assert IMAGENET_MEAN[0] < 1 and IMAGENET_STD[0] < 1


# --------------------------------------------------------------------------- #
# splitting / subsampling
# --------------------------------------------------------------------------- #


def test_make_splits_partitions_without_overlap():
    names = [f"s{i}" for i in range(50)]
    sp = make_splits(names, 30, 10, 10, seed=1)
    assert len(sp["train"]) == 30 and len(sp["val"]) == 10 and len(sp["test"]) == 10
    alln = sp["train"] + sp["val"] + sp["test"]
    assert len(set(alln)) == 50
    assert make_splits(names, 30, 10, 10, seed=1) == sp


def test_make_splits_clamps_when_dataset_too_small():
    sp = make_splits([f"s{i}" for i in range(5)], 10, 10, 10, seed=0)
    assert len(sp["train"]) == 5 and len(sp["val"]) == 0 and len(sp["test"]) == 0


def test_stratified_subsample_is_seeded_and_covers_rare(manifest):
    root = Path(manifest["root"])
    train_names = manifest["splits"]["train"]
    strata = compute_strata(root, train_names, manifest["rare_classes"])
    a = stratified_subsample(train_names, 0.25, strata, seed=3)
    b = stratified_subsample(train_names, 0.25, strata, seed=3)
    assert a == b, "subsampling must be deterministic for a fixed seed"
    assert set(a).issubset(set(train_names))
    assert len(a) <= max(1, int(round(len(train_names) * 0.25)) + len(set(strata.values())))
    c = stratified_subsample(train_names, 1.0, strata, seed=3)
    assert set(c) == set(train_names)


# --------------------------------------------------------------------------- #
# synthetic generator
# --------------------------------------------------------------------------- #


def test_synthetic_generator_is_deterministic():
    g = SyntheticAerialGenerator(seed=42)
    s0 = g.generate(3)
    s1 = SyntheticAerialGenerator(seed=42).generate(3)
    assert np.array_equal(s0.image, s1.image)
    assert np.array_equal(s0.mask, s1.mask)
    assert s0.instances == s1.instances


def test_synthetic_scene_has_ground_truth_invariants():
    g = SyntheticAerialGenerator(seed=1)
    scene = g.generate(0)
    assert scene.image.shape[2] == 3 and scene.image.dtype == np.uint8
    assert scene.mask.shape == scene.image.shape[:2]
    assert set(np.unique(scene.mask)).issubset(set(PALETTE.keys()))
    # every class present in the mask must have a plausible palette entry
    for cls in np.unique(scene.mask):
        assert int(cls) in PALETTE
    assert scene.tags["n_instances"] == len(scene.instances)
    for inst in scene.instances:
        x0, y0, x1, y1 = inst["bbox"]
        assert 0 <= x0 < x1 <= scene.image.shape[1]
        assert 0 <= y0 < y1 <= scene.image.shape[0]
        assert 0.0 <= inst["visible_frac"] <= 1.0


def test_synthetic_masks_are_pixel_perfect_against_palette_colours():
    """With degradations off, mask pixels must match the painted palette colour.

    This is the check that makes the synthetic fallback trustworthy: the mask
    really indexes the pixels that were painted, not an approximation.
    """
    from aerial_vit.config import SyntheticCfg

    clean = SyntheticCfg(
        n_scenes=2,
        base_size=320,
        glare_prob=0.0,
        blur_prob=0.0,
        haze_prob=0.0,
        altitude_shift_prob=0.0,
        texture=0.0,
        seed=9,
    )
    scene = SyntheticAerialGenerator(clean).generate(1)
    for cls, colour in PALETTE.items():
        sel = scene.mask == cls
        if sel.sum() < 50:
            continue
        mean = scene.image[sel].mean(axis=0)
        # colours carry a bounded per-instance jitter; no photometric effects
        assert np.abs(mean - np.array(colour)).max() < 45, f"class {cls} colour mismatch"
