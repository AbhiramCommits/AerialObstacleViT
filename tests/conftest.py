"""Shared fixtures.

The committed ``artifacts/data_manifest.json`` describes a deterministic
synthetic dataset, but the images/masks themselves are gitignored. On a fresh
clone (or in CI) the dataset is rebuilt here, once per session, from the
generator settings recorded in the manifest, so the data-dependent tests run
without a manual ``fetch_data`` step and without rewriting the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerial_vit.config import ARTIFACTS_DIR, load_config
from aerial_vit.data.semantic_drone import (
    MANIFEST_NAME,
    build_manifest,
    resolve_manifest_root,
    write_splits,
)
from aerial_vit.data.synthetic import generate_synthetic_dataset


def _dataset_present(root: Path, manifest: dict) -> bool:
    names = [n for split in manifest.get("splits", {}).values() for n in split]
    return (root / "splits.json").exists() and all(
        (root / "images" / f"{n}.png").exists() and (root / "masks" / f"{n}.png").exists()
        for n in names
    )


def _regenerate_synthetic(root: Path, manifest: dict) -> None:
    cfg = load_config("configs/base.yaml")
    for key, value in (manifest.get("synthetic_cfg") or {}).items():
        if hasattr(cfg.data.synthetic, key):
            setattr(cfg.data.synthetic, key, value)
    n_need = manifest["n_train"] + manifest["n_val"] + manifest["n_test"]
    n_scenes = max(cfg.data.synthetic.n_scenes, n_need)  # mirrors fetch_data
    generate_synthetic_dataset(root, cfg.data, n_scenes=n_scenes)
    write_splits(root, manifest["splits"])

    # The generator is deterministic; make sure the rebuilt data really is the
    # dataset the committed manifest describes.
    rebuilt = build_manifest(
        root,
        cfg.data,
        source="synthetic",
        splits=manifest["splits"],
        image_size=cfg.data.image_size,
    )
    if rebuilt["per_class_pixels"] != manifest["per_class_pixels"]:
        pytest.exit(
            f"regenerated synthetic data under {root} does not match {MANIFEST_NAME}; "
            "re-run `python scripts/fetch_data.py` with the config that produced it",
            returncode=1,
        )


@pytest.fixture(scope="session", autouse=True)
def ensure_synthetic_dataset() -> None:
    p = ARTIFACTS_DIR / MANIFEST_NAME
    if not p.exists():
        return
    manifest = json.loads(p.read_text())
    if manifest.get("source") != "synthetic" or not manifest.get("root"):
        return
    root = resolve_manifest_root(manifest["root"])
    if not _dataset_present(root, manifest):
        _regenerate_synthetic(root, manifest)
