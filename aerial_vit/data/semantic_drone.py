"""Aerial scene dataset + taxonomy remapping + data manifest.

Two sources feed the same on-disk layout:

* **Semantic Drone Dataset** (primary).  ``prepare_semantic_drone`` reads the
  official ``class_dict.csv`` and its colour-coded label images, remaps the 20+
  native classes onto the canonical 8-class taxonomy in ``configs/taxonomy.yaml``,
  and writes canonical index PNGs.
* **Synthetic fallback** (``synthetic.py``), written directly in canonical form.

Both land in the layout::

    <root>/images/<name>.png     RGB
    <root>/masks/<name>.png      uint8 canonical indices
    <root>/boxes/<name>.json     [{"class": int, "bbox": [x0,y0,x1,y1]}, ...]
    <root>/meta/<name>.json      per-scene statistics / degradation tags
    <root>/splits.json           {"train": [...], "val": [...], "test": [...]}

``AerialSceneDataset`` yields ``(image, mask, boxes, meta)`` where ``boxes`` is an
``(N, 5)`` float tensor ``[x1, y1, x2, y2, class]`` and ``meta`` carries the
derived binary ``landing_zone`` mask.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from aerial_vit.config import ARTIFACTS_DIR, REPO_ROOT, Config, DataCfg
from aerial_vit.data.synthetic import PALETTE
from aerial_vit.data.transforms import build_eval_transform, build_train_transform

MANIFEST_NAME = "data_manifest.json"


# --------------------------------------------------------------------------- #
# Manifest helpers
# --------------------------------------------------------------------------- #


def manifest_path(cfg: Config) -> Path:
    return default_manifest_path(cfg)


def default_manifest_path(cfg: Config | None = None) -> Path:
    if cfg is None:
        return ARTIFACTS_DIR / MANIFEST_NAME
    base = Path(cfg.run.out_dir)
    if not base.is_absolute():
        base = Path.cwd() / base
    return base / MANIFEST_NAME


def load_manifest(path: str | Path | None = None, cfg: Config | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else default_manifest_path(cfg)
    if not p.exists():
        raise FileNotFoundError(
            f"No data manifest at {p}. Run `python scripts/fetch_data.py --config configs/smoke.yaml` first."
        )
    manifest = json.loads(p.read_text())
    if manifest.get("root"):
        manifest["root"] = str(resolve_manifest_root(manifest["root"]))
    return manifest


def resolve_manifest_root(root: str | Path) -> Path:
    """Resolve a manifest ``root``; relative roots are relative to the repo root."""
    root = Path(root)
    return root if root.is_absolute() else (REPO_ROOT / root).resolve()


def portable_manifest_root(root: str | Path) -> str:
    """Store roots inside the repo as repo-relative POSIX paths so manifests are portable."""
    root = Path(root)
    if root.is_absolute():
        try:
            return root.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError:
            return str(root)
    return root.as_posix()


def build_manifest(
    root: str | Path,
    cfg: DataCfg,
    source: str,
    splits: dict[str, Sequence[str]],
    image_size: int | tuple[int, int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Scan a prepared dataset root and summarise it into a manifest dict."""
    root = Path(root)
    per_class_pixels = np.zeros(cfg.num_classes, np.int64)
    per_class_instances = np.zeros(cfg.num_classes, np.int64)
    per_split_instances: dict[str, dict[str, int]] = {}
    sizes: list[tuple[int, int]] = []

    for split, names in splits.items():
        split_counts = {c: 0 for c in cfg.classes}
        for name in names:
            mask = cv2.imread(str(root / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(root / "masks" / f"{name}.png")
            sizes.append((int(mask.shape[0]), int(mask.shape[1])))
            idx = np.where((mask >= 0) & (mask < cfg.num_classes), mask, 0).ravel()
            per_class_pixels += np.bincount(idx, minlength=cfg.num_classes)[: cfg.num_classes]
            for inst in read_boxes(root, name):
                cls = int(inst["class"])
                if 0 <= cls < cfg.num_classes:
                    per_class_instances[cls] += 1
                    split_counts[cfg.classes[cls]] += 1
        per_split_instances[split] = split_counts

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    total_px = int(per_class_pixels.sum()) or 1
    px_frac = {c: round(float(per_class_pixels[j]) / total_px, 6) for j, c in enumerate(cfg.classes)}

    # Rare classes: defined from *measured* statistics, never guessed.
    # A class is rare when it covers less than `rare_pixel_threshold` of all
    # pixels, or when it *has instance annotations* but fewer than
    # `rare_instance_threshold` of them across the whole dataset.  Classes that
    # are only ever painted as regions (roads, grass, water) have no instance
    # annotations at all, so the instance rule does not apply to them.
    instance_threshold = cfg.rare_instance_threshold
    rare_classes = [
        c
        for c in cfg.classes
        if c != "background"
        and (
            px_frac[c] < cfg.rare_pixel_threshold
            or (
                per_class_instances[cfg.classes.index(c)] > 0
                and per_class_instances[cfg.classes.index(c)] < instance_threshold
            )
        )
    ]

    manifest: dict[str, Any] = {
        "source": source,
        "root": str(root),
        "n_train": len(splits.get("train", [])),
        "n_val": len(splits.get("val", [])),
        "n_test": len(splits.get("test", [])),
        "classes": list(cfg.classes),
        "landable_classes": list(cfg.landable_classes),
        "num_classes": cfg.num_classes,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "native_image_size": [int(sizes[0][0]), int(sizes[0][1])] if sizes else [0, 0],
        "per_class_pixels": {c: int(per_class_pixels[j]) for j, c in enumerate(cfg.classes)},
        "per_class_pixel_fraction": px_frac,
        "per_class_instances": {c: int(per_class_instances[j]) for j, c in enumerate(cfg.classes)},
        "per_split_instances": per_split_instances,
        "rare_classes": rare_classes,
        "rare_definition": {
            "pixel_fraction_below": cfg.rare_pixel_threshold,
            "dataset_instances_below": cfg.rare_instance_threshold,
            "note": "instance rule applies only to classes that carry instance annotations",
        },
        "instance_classes": [
            c for j, c in enumerate(cfg.classes) if per_class_instances[j] > 0
        ],
        "splits": {k: list(v) for k, v in splits.items()},
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(manifest)
    if out.get("root"):
        out["root"] = portable_manifest_root(out["root"])
    p.write_text(json.dumps(out, indent=2) + "\n")
    return p


# --------------------------------------------------------------------------- #
# Box / meta IO
# --------------------------------------------------------------------------- #


def read_boxes(root: str | Path, name: str) -> list[dict[str, Any]]:
    p = Path(root) / "boxes" / f"{name}.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return [d for d in data if d.get("bbox")]


def read_meta(root: str | Path, name: str) -> dict[str, Any]:
    p = Path(root) / "meta" / f"{name}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def write_splits(root: str | Path, splits: dict[str, Sequence[str]]) -> Path:
    p = Path(root) / "splits.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({k: list(v) for k, v in splits.items()}, indent=1) + "\n")
    return p


def read_splits(root: str | Path) -> dict[str, list[str]]:
    p = Path(root) / "splits.json"
    if not p.exists():
        raise FileNotFoundError(f"No splits.json in {root}")
    return json.loads(p.read_text())


def make_splits(
    names: Sequence[str], n_train: int, n_val: int, n_test: int, seed: int = 1337
) -> dict[str, list[str]]:
    """Deterministically shuffle ``names`` and partition it."""
    names = list(names)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(names))
    shuffled = [names[i] for i in perm]
    n_train = min(n_train, len(shuffled))
    n_val = min(n_val, max(0, len(shuffled) - n_train))
    n_test = min(n_test, max(0, len(shuffled) - n_train - n_val))
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


# --------------------------------------------------------------------------- #
# Semantic Drone Dataset preparation
# --------------------------------------------------------------------------- #


def _parse_class_dict(path: Path) -> dict[tuple[int, int, int], str]:
    """colour -> native class name from the SDD ``class_dict.csv``."""
    lookup: dict[tuple[int, int, int], str] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            key = {k.strip().lower(): (v or "").strip() for k, v in row.items()}
            try:
                r, g, b = int(key["r"]), int(key["g"]), int(key["b"])
            except (KeyError, ValueError):
                continue
            lookup[(r, g, b)] = key.get("name", key.get("class", "unlabeled"))
    return lookup


def _find_sdd(root: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Locate (images_dir, labels_dir, class_dict) inside a possibly nested root."""
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()][:200]
    img_dir = lbl_dir = dict_path = None
    for c in candidates:
        if img_dir is None and c.name in {"images", "image", "original_images"}:
            img_dir = c
        if lbl_dir is None and c.name in {"label_images_semantic", "labels", "label_images", "gt"}:
            lbl_dir = c
        if dict_path is None and c.name == "class_dict.csv":
            dict_path = c
    return img_dir, lbl_dir, dict_path


def prepare_semantic_drone(
    src: str | Path,
    out_root: str | Path,
    cfg: DataCfg,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 1337,
) -> dict[str, Any]:
    """Convert a Semantic Drone Dataset tree into the canonical layout.

    Raises ``RuntimeError`` if the tree does not look like the real dataset, so
    ``fetch_data.py`` can fall back to the synthetic generator *honestly*.
    """
    src, out_root = Path(src), Path(out_root)
    img_dir, lbl_dir, dict_path = _find_sdd(src)
    if img_dir is None or lbl_dir is None:
        raise RuntimeError(f"could not find images/labels under {src}")
    if dict_path is None:
        raise RuntimeError(f"could not find class_dict.csv under {src}")

    colour_to_native = _parse_class_dict(dict_path)
    native_to_canon = cfg.sdd_class_map
    if not native_to_canon:
        raise RuntimeError("cfg.data.sdd_class_map is empty; check configs/taxonomy.yaml")

    # colour -> canonical index
    colour_to_canon: dict[tuple[int, int, int], int] = {}
    for colour, native in colour_to_native.items():
        canon = native_to_canon.get(native)
        if canon is None:
            canon = native_to_canon.get(native.replace(" ", "-"), 0)
        colour_to_canon[colour] = int(canon)

    for sub in ("images", "masks", "boxes", "meta"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    names: list[str] = []
    for img_p in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl_p = None
        for ext in (".png", ".jpg"):
            cand = lbl_dir / f"{img_p.stem}{ext}"
            if cand.exists():
                lbl_p = cand
                break
        if lbl_p is None:
            continue
        img = cv2.imread(str(img_p))
        lbl = cv2.imread(str(lbl_p))
        if img is None or lbl is None:
            continue
        name = img_p.stem
        mask = _remap_label_image(lbl, colour_to_canon, cfg.num_classes)
        cv2.imwrite(str(out_root / "images" / f"{name}.png"), img)
        cv2.imwrite(str(out_root / "masks" / f"{name}.png"), mask)
        boxes = _instances_from_mask(mask, cfg)
        (out_root / "boxes" / f"{name}.json").write_text(json.dumps(boxes, indent=1))
        (out_root / "meta" / f"{name}.json").write_text(json.dumps({"origin": "semantic_drone"}))
        names.append(name)

    if not names:
        raise RuntimeError(f"no image/label pairs found under {src}")

    splits = make_splits(names, n_train, n_val, n_test, seed)
    write_splits(out_root, splits)
    return {"n_scenes": len(names), "root": str(out_root), "splits": splits}


def _remap_label_image(lbl_bgr: np.ndarray, colour_to_canon: dict, num_classes: int) -> np.ndarray:
    rgb = cv2.cvtColor(lbl_bgr, cv2.COLOR_BGR2RGB)
    flat = rgb.reshape(-1, 3)
    out = np.zeros(flat.shape[0], np.uint8)
    # exact colour match first
    keys = np.array(list(colour_to_canon.keys()), np.int32)
    vals = np.array(list(colour_to_canon.values()), np.uint8)
    if keys.size:
        packed = flat[:, 0].astype(np.int32) * 65536 + flat[:, 1].astype(np.int32) * 256 + flat[:, 2]
        pkeys = keys[:, 0] * 65536 + keys[:, 1] * 256 + keys[:, 2]
        order = np.argsort(pkeys)
        pos = np.searchsorted(pkeys[order], packed)
        pos = np.clip(pos, 0, len(order) - 1)
        matched = pkeys[order][pos] == packed
        out[matched] = vals[order][pos[matched]]
    mask = out.reshape(rgb.shape[:2])
    return np.clip(mask, 0, num_classes - 1).astype(np.uint8)


def _instances_from_mask(mask: np.ndarray, cfg: DataCfg) -> list[dict[str, Any]]:
    """Connected components per class -> instance boxes (real data has no boxes)."""
    out: list[dict[str, Any]] = []
    for cls in range(cfg.num_classes):
        if cls == 0:  # background is not an object
            continue
        binary = (mask == cls).astype(np.uint8)
        if binary.sum() == 0:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 12:
                continue
            out.append({"class": int(cls), "bbox": [int(x), int(y), int(x + w), int(y + h)], "area": int(area)})
    return out


# --------------------------------------------------------------------------- #
# Stratified label subsampling (Step 4)
# --------------------------------------------------------------------------- #


def _resolve_rare_indices(rare_classes: Sequence[str | int]) -> set[int]:
    """Accept rare classes as canonical names or indices."""
    from aerial_vit.utils.viz import CLASS_NAMES

    out: set[int] = set()
    for c in rare_classes:
        if isinstance(c, str):
            if c in CLASS_NAMES:
                out.add(CLASS_NAMES.index(c))
        else:
            out.add(int(c))
    return out


def compute_strata(
    root: str | Path, names: Sequence[str], rare_classes: Sequence[str | int]
) -> dict[str, str]:
    """Assign each image a stratum key = sorted tuple of rare classes present.

    Stratifying by *rare-class presence* is what keeps a 1 % label budget from
    silently dropping every person/vehicle frame.
    """
    rare = _resolve_rare_indices(rare_classes)
    strata: dict[str, str] = {}
    for name in names:
        present = sorted({int(b["class"]) for b in read_boxes(root, name)} & rare)
        strata[name] = "rare:" + ",".join(str(c) for c in present) if present else "common"
    return strata


def stratified_subsample(
    names: Sequence[str],
    fraction: float,
    strata: dict[str, str],
    seed: int = 1337,
    min_per_stratum: int = 1,
) -> list[str]:
    """Proportional, seeded, stratified subsample of ``names``.

    Guarantees at least ``min_per_stratum`` image per non-empty stratum (as long
    as the budget allows) so rare classes survive aggressive budgets.
    """
    names = list(names)
    if fraction >= 1.0:
        return names
    budget = max(1, int(round(len(names) * float(fraction))))
    rng = np.random.default_rng(seed)

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(strata.get(n, "common"), []).append(n)
    for g in groups.values():
        rng.shuffle(g)

    # proportional quotas
    quota = {k: max(min_per_stratum, int(round(len(v) * fraction))) for k, v in groups.items()}
    # trim to budget, largest-first
    while sum(quota.values()) > budget:
        k = max(quota, key=lambda x: quota[x])
        if quota[k] <= min_per_stratum:
            k = max(quota, key=lambda x: len(groups[x]))
            if quota[k] <= 1:
                break
        quota[k] -= 1
    # top up if under budget
    keys = sorted(groups, key=lambda k: -len(groups[k]))
    i = 0
    while sum(quota.values()) < budget and keys:
        k = keys[i % len(keys)]
        if quota[k] < len(groups[k]):
            quota[k] += 1
        i += 1
        if i > 10 * budget + 100:
            break

    chosen: list[str] = []
    for k, v in groups.items():
        chosen.extend(v[: quota[k]])
    rng.shuffle(chosen)
    return chosen


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass
class SegmentationSample:
    image: torch.Tensor
    mask: torch.Tensor
    boxes: torch.Tensor  # (N, 5) [x1, y1, x2, y2, class]
    meta: dict[str, Any]


class AerialSceneDataset(Dataset):
    """Torch dataset over the canonical aerial layout.

    Returns ``(image, mask, boxes, meta)``:

    * ``image``  float32 ``(3, H, W)`` ImageNet-normalised
    * ``mask``   int64   ``(H, W)`` canonical class indices
    * ``boxes``  float32 ``(N, 5)`` ``[x1, y1, x2, y2, class]`` in pixels
    * ``meta``   dict with ``name``, binary ``landing_zone`` mask and stats
    """

    def __init__(
        self,
        root: str | Path,
        cfg: DataCfg,
        names: Sequence[str],
        split: str = "train",
        seed: int = 1337,
        augment: bool | None = None,
        image_size: int | None = None,
    ):
        self.root = Path(root)
        self.cfg = cfg
        self.names = list(names)
        self.split = split
        self.seed = int(seed)
        self.epoch = 0
        self.image_size = int(image_size or cfg.image_size)
        self.augment = (split == "train" and cfg.augment) if augment is None else bool(augment)
        self.transform = (
            build_train_transform(self.image_size, augment=True)
            if self.augment
            else build_eval_transform(self.image_size)
        )
        self.landable = torch.tensor(sorted(cfg.landable_index_set), dtype=torch.long)

    # -- Dataset API ------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.names)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, idx: int) -> np.random.Generator:
        # Seeded per (seed, epoch, index) so augmentation is order-independent
        # and therefore exactly reproducible regardless of worker scheduling.
        key = (self.seed * 1_000_003 + self.epoch * 7919 + idx * 104729) % (2**63)
        return np.random.default_rng(key)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        img_bgr = cv2.imread(str(self.root / "images" / f"{name}.png"), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(self.root / "masks" / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask is None:
            raise FileNotFoundError(f"missing image/mask for {name} under {self.root}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        recs = read_boxes(self.root, name)
        if recs:
            boxes = np.array([r["bbox"] for r in recs], np.float32)
            labels = np.array([int(r["class"]) for r in recs], np.int64)
        else:
            boxes = np.zeros((0, 4), np.float32)
            labels = np.zeros((0,), np.int64)

        sample = {"image": img, "mask": mask.astype(np.uint8), "boxes": boxes, "labels": labels}
        out = self.transform(sample, self._rng(idx))

        n = out["boxes"].shape[0]
        cls = out["labels"].reshape(n, 1).float()
        det_boxes = torch.cat([out["boxes"], cls], dim=1) if n else torch.zeros((0, 5))

        mask_t = out["mask"]
        landing = torch.isin(mask_t.reshape(-1), self.landable).reshape(mask_t.shape)

        meta = {
            "name": name,
            "split": self.split,
            "landing_zone": landing,
            "n_instances": int(n),
            "classes_present": sorted({int(c) for c in out["labels"].tolist()}),
            **read_meta(self.root, name),
        }
        return out["image"], mask_t, det_boxes, meta


class SubsetAerialSceneDataset(AerialSceneDataset):
    """An ``AerialSceneDataset`` restricted to a subset of names (same root)."""

    @classmethod
    def from_dataset(cls, ds: AerialSceneDataset, names: Sequence[str]) -> SubsetAerialSceneDataset:
        sub = cls(ds.root, ds.cfg, names, split=ds.split, seed=ds.seed, augment=ds.augment, image_size=ds.image_size)
        return sub


# --------------------------------------------------------------------------- #
# Convenience builder
# --------------------------------------------------------------------------- #


def build_datasets(
    cfg: Config,
    root: str | Path | None = None,
    label_fraction: float | None = None,
    seed: int | None = None,
    manifest: dict[str, Any] | None = None,
):
    """Build (train, val, test) datasets from a prepared root + manifest."""
    manifest = manifest or load_manifest(cfg=cfg)
    root = Path(root or manifest["root"])
    splits = manifest.get("splits") or read_splits(root)
    seed = int(seed if seed is not None else cfg.run.seed)
    frac = float(label_fraction if label_fraction is not None else cfg.train.label_fraction)

    train_names = list(splits["train"])
    if frac < 1.0:
        strata = compute_strata(root, train_names, manifest.get("rare_classes", []))
        train_names = stratified_subsample(train_names, frac, strata, seed=seed)

    train = AerialSceneDataset(root, cfg.data, train_names, split="train", seed=seed)
    val = AerialSceneDataset(root, cfg.data, splits.get("val", []), split="val", seed=seed + 1)
    test = AerialSceneDataset(root, cfg.data, splits.get("test", []), split="test", seed=seed + 2)
    return train, val, test


def describe_root(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    names = sorted(p.stem for p in (root / "images").glob("*.png"))
    return {"root": str(root), "n_images": len(names), "names": names}


__all__ = [
    "AerialSceneDataset",
    "MANIFEST_NAME",
    "SegmentationSample",
    "SubsetAerialSceneDataset",
    "build_datasets",
    "build_manifest",
    "compute_strata",
    "default_manifest_path",
    "describe_root",
    "load_manifest",
    "make_splits",
    "manifest_path",
    "portable_manifest_root",
    "prepare_semantic_drone",
    "read_boxes",
    "read_meta",
    "read_splits",
    "resolve_manifest_root",
    "stratified_subsample",
    "write_manifest",
    "write_splits",
    "PALETTE",
]
