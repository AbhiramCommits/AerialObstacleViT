"""Procedural synthetic aerial dataset: the offline fallback for this benchmark.

The primary target is the real Semantic Drone Dataset.  When it is unavailable
(no network, no local copy) this module *generates* a deterministic dataset of
nadir-looking aerial scenes with **pixel-perfect** semantic masks and derived
instance boxes, so every downstream number in the repo is measurable rather
than estimated.

Each scene is composed of, in painter's order (later layers occlude earlier):

    textured ground -> dirt/background patches -> roads/parking (paved-area)
    -> ponds (water) -> tree canopies -> building roofs -> vehicles -> persons

and then degraded with any of the realistic UAV imaging artefacts:

    * **glare**      -- a saturated specular hotspot + exposure lift
    * **motion-blur**-- a directional linear blur kernel (platform vibration)
    * **haze**       -- atmospheric scattering (contrast compression)
    * **altitude**   -- a global scale change (ground-sample-distance shift)

Because objects are painted one at a time onto an instance-id map, the visible
pixel count of every instance is known, which gives a calibrated occlusion
statistic for free.  That statistic is what Step 5's failure-mode tagger reads.

Determinism: an ``numpy.random.Generator`` seeded from ``(cfg.seed, scene_index)``
drives every random choice, so scene *i* is byte-identical across machines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from aerial_vit.config import DataCfg, SyntheticCfg

# RGB palette per canonical class.  Kept close to the Semantic Drone Dataset's
# own colours so figures "look like" the real thing.
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (122, 104, 82),  # background / dirt
    1: (108, 110, 116),  # paved-area
    2: (84, 128, 62),  # grass
    3: (156, 74, 62),  # roof
    4: (46, 94, 44),  # tree
    5: (58, 110, 158),  # water
    6: (196, 62, 58),  # person
    7: (198, 200, 208),  # vehicle
}


@dataclass
class SyntheticScene:
    """One generated scene plus full ground truth."""

    index: int
    image: np.ndarray  # (H, W, 3) uint8
    mask: np.ndarray  # (H, W) uint8, canonical class indices
    instances: list[dict[str, Any]] = field(default_factory=list)
    tags: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.image.shape[0]), int(self.image.shape[1])


# --------------------------------------------------------------------------- #
# Low-level painting helpers
# --------------------------------------------------------------------------- #


def _add_texture(rng: np.random.Generator, img: np.ndarray, strength: float) -> np.ndarray:
    """Add smooth multi-scale luminance texture to an image."""
    h, w = img.shape[:2]
    noise = np.zeros((h, w), np.float32)
    for sigma in (1.5, 4.0, 12.0):
        n = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        noise += n / (sigma * 0.6)
    noise *= 255.0 * strength / (np.abs(noise).max() + 1e-6)
    out = img.astype(np.float32) + noise[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _fill_blob(
    rng: np.random.Generator,
    canvas: np.ndarray,
    inst_map: np.ndarray,
    class_idx: int,
    colour: tuple[int, int, int],
    center: tuple[int, int],
    radius: float,
    n_verts: int = 9,
    jitter: float = 0.35,
) -> np.ndarray:
    """Paint an irregular polygon blob; return the painted region mask."""
    angles = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    radii = radius * (1.0 + jitter * rng.normal(0, 1, n_verts))
    radii = np.clip(radii, radius * 0.35, radius * 1.65)
    pts = np.stack(
        [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)], axis=1
    ).astype(np.int32)
    region = np.zeros(canvas.shape[:2], np.uint8)
    cv2.fillPoly(region, [pts], 255)
    canvas[region > 0] = colour
    inst_map[region > 0] = class_idx
    return region


def _paint_rect(
    rng: np.random.Generator,
    canvas: np.ndarray,
    inst_map: np.ndarray,
    class_idx: int,
    colour: tuple[int, int, int],
    x: int,
    y: int,
    w: int,
    h: int,
    angle: float = 0.0,
    edge: bool = False,
) -> np.ndarray:
    """Paint a (possibly rotated) rectangle; return the painted region."""
    region = np.zeros(canvas.shape[:2], np.uint8)
    box = cv2.boxPoints(((x + w / 2, y + h / 2), (w, h), angle)).astype(np.int32)
    cv2.fillPoly(region, [box], 255)
    canvas[region > 0] = colour
    if edge:
        shade = tuple(int(max(0, c * 0.72)) for c in colour)
        cv2.polylines(canvas, [box], True, shade, 2, lineType=cv2.LINE_AA)
    inst_map[region > 0] = class_idx
    return region


def _paint_ellipse(
    rng: np.random.Generator,
    canvas: np.ndarray,
    inst_map: np.ndarray,
    class_idx: int,
    colour: tuple[int, int, int],
    center: tuple[int, int],
    axes: tuple[int, int],
    angle: float = 0.0,
) -> np.ndarray:
    region = np.zeros(canvas.shape[:2], np.uint8)
    cv2.ellipse(region, center, axes, angle, 0, 360, 255, -1, lineType=cv2.LINE_AA)
    canvas[region > 0] = colour
    inst_map[region > 0] = class_idx
    return region


def _random_colour(rng: np.random.Generator, base: tuple[int, int, int], spread: int = 22):
    return tuple(int(np.clip(c + rng.integers(-spread, spread + 1), 0, 255)) for c in base)


def _bbox_from_region(region: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(region > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


# --------------------------------------------------------------------------- #
# Degradations
# --------------------------------------------------------------------------- #


def _apply_glare(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
    r = float(rng.uniform(0.25, 0.6) * max(h, w))
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    glow = np.clip(1.0 - d / r, 0.0, 1.0) ** 2
    intensity = float(rng.uniform(90, 170))
    out = img.astype(np.float32) + glow[..., None] * intensity
    out += float(rng.uniform(0, 18))  # slight exposure lift
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_motion_blur(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    length = int(rng.integers(9, 19))
    angle = float(rng.uniform(0, 180))
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    kernel /= kernel.sum() + 1e-6
    return cv2.filter2D(img, -1, kernel)


def _apply_haze(rng: np.random.Generator, img: np.ndarray) -> np.ndarray:
    alpha = float(rng.uniform(0.15, 0.42))
    out = img.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha
    # contrast compression around mid-grey
    out = 128.0 + (out - 128.0) * (1.0 - 0.45 * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_altitude_zoom(
    img: np.ndarray, mask: np.ndarray, inst_map: np.ndarray, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resize the whole scene by ``scale`` and centre-crop/pad back to size."""
    h, w = img.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_s = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    inst_s = cv2.resize(inst_map, (nw, nh), interpolation=cv2.INTER_NEAREST)

    def _fit(a: np.ndarray, fill: Any) -> np.ndarray:
        out = np.full((h, w) + a.shape[2:], fill, dtype=a.dtype)
        ys = max(0, (nh - h) // 2)
        xs = max(0, (nw - w) // 2)
        ys2, xs2 = ys + min(h, nh), xs + min(w, nw)
        out[: ys2 - ys, : xs2 - xs] = a[ys:ys2, xs:xs2]
        return out

    return _fit(img_s, 90), _fit(mask_s, 2), _fit(inst_s, 0)


# --------------------------------------------------------------------------- #
# Scene generator
# --------------------------------------------------------------------------- #

# Relative frequency of each "object" class among the sampled objects.
_OBJECT_MIX = {3: 0.22, 4: 0.40, 6: 0.10, 7: 0.28}  # roof, tree, person, vehicle


class SyntheticAerialGenerator:
    """Deterministic procedural generator of nadir aerial scenes."""

    def __init__(self, cfg: SyntheticCfg | DataCfg | None = None, seed: int | None = None):
        if isinstance(cfg, DataCfg):
            syn = cfg.synthetic
        else:
            syn = cfg or SyntheticCfg()
        self.syn = syn
        self.seed = int(seed if seed is not None else syn.seed)

    # -- public ------------------------------------------------------------ #
    def generate(self, index: int) -> SyntheticScene:
        rng = np.random.default_rng((self.seed * 1_000_003 + index) % (2**63))
        size = int(self.syn.base_size)
        tags: dict[str, Any] = {}

        img, mask, inst_map, drawn = self._render_base(rng, size)

        # --- altitude shift (geometric, applies to GT too) --------------- #
        scale = 1.0
        if rng.random() < self.syn.altitude_shift_prob:
            scale = float(rng.choice(self.syn.altitude_scales))
            img, mask, inst_map = _apply_altitude_zoom(img, mask, inst_map, scale)
        tags["altitude_scale"] = round(float(scale), 4)

        # --- photometric degradations (image only) ----------------------- #
        glare = rng.random() < self.syn.glare_prob
        blur = rng.random() < self.syn.blur_prob
        haze = rng.random() < self.syn.haze_prob
        if glare:
            img = _apply_glare(rng, img)
        if blur:
            img = _apply_motion_blur(rng, img)
        if haze:
            img = _apply_haze(rng, img)
        img = np.clip(
            img.astype(np.float32) + rng.normal(0, 2.4, img.shape), 0, 255
        ).astype(np.uint8)
        tags.update(glare=bool(glare), motion_blur=bool(blur), haze=bool(haze))

        instances = self._collect_instances(inst_map, mask, drawn)
        tags.update(self._scene_stats(img, mask, instances))
        return SyntheticScene(index=index, image=img, mask=mask, instances=instances, tags=tags)

    # -- internals --------------------------------------------------------- #
    def _render_base(self, rng: np.random.Generator, size: int) -> tuple:
        """Paint the un-degraded scene. Returns (img, mask, inst_map, drawn)."""
        img = np.empty((size, size, 3), np.uint8)
        img[:] = PALETTE[2]
        mask = np.full((size, size), 2, np.uint8)  # everything starts as grass
        inst_map = np.zeros((size, size), np.int32)  # 0 == unlabelled background

        img = _add_texture(rng, img, self.syn.texture)

        drawn: list[dict[str, Any]] = []

        # -- dirt / background patches ------------------------------------
        for _ in range(int(rng.integers(0, 3))):
            c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            r = float(rng.uniform(0.06, 0.14) * size)
            reg = _fill_blob(rng, img, inst_map, 0, _random_colour(rng, PALETTE[0]), c, r)
            mask[reg > 0] = 0

        # -- roads (paved-area) -------------------------------------------
        for _ in range(int(rng.integers(1, 4))):
            horizontal = rng.random() < 0.5
            width = int(rng.integers(int(0.05 * size), int(0.12 * size)))
            thick = int(rng.integers(int(0.035 * size), int(0.075 * size)))
            colour = _random_colour(rng, PALETTE[1], 12)
            if horizontal:
                y = int(rng.integers(thick, max(thick + 1, size - thick)))
                reg = _paint_rect(
                    rng, img, inst_map, 1, colour, 0, y - thick // 2, size, thick
                )
            else:
                x = int(rng.integers(thick, max(thick + 1, size - thick)))
                reg = _paint_rect(
                    rng, img, inst_map, 1, colour, x - thick // 2, 0, thick, size
                )
            mask[reg > 0] = 1
            # occasional parking lot
            if rng.random() < 0.4:
                w = int(rng.integers(width, int(0.3 * size)))
                h = int(rng.integers(int(0.06 * size), int(0.18 * size)))
                x = int(rng.integers(0, max(1, size - w)))
                y = int(rng.integers(0, max(1, size - h)))
                reg = _paint_rect(rng, img, inst_map, 1, colour, x, y, w, h)
                mask[reg > 0] = 1

        # -- water ---------------------------------------------------------
        if rng.random() < 0.45:
            c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            r = float(rng.uniform(0.07, 0.16) * size)
            reg = _fill_blob(rng, img, inst_map, 5, _random_colour(rng, PALETTE[5], 18), c, r)
            mask[reg > 0] = 5

        # -- objects: roofs, trees, persons, vehicles --------------------- #
        n_obj = int(rng.integers(self.syn.min_objects, self.syn.max_objects + 1))
        classes = list(_OBJECT_MIX)
        probs = np.array([_OBJECT_MIX[c] for c in classes], float)
        probs /= probs.sum()
        # Guarantee at least one of each: roof, tree, person, vehicle.
        chosen = list(classes) + list(rng.choice(classes, size=max(0, n_obj - len(classes)), p=probs))

        for cls in chosen:
            if cls == 3:  # roof
                w = int(rng.integers(int(0.10 * size), int(0.30 * size)))
                h = int(rng.integers(int(0.08 * size), int(0.26 * size)))
                x = int(rng.integers(0, max(1, size - w)))
                y = int(rng.integers(0, max(1, size - h)))
                ang = float(rng.uniform(0, 180))
                reg = _paint_rect(
                    rng, img, inst_map, 3, _random_colour(rng, PALETTE[3], 26), x, y, w, h, ang, edge=True
                )
            elif cls == 4:  # tree canopy
                c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
                r = float(rng.uniform(0.035, 0.095) * size)
                reg = _fill_blob(
                    rng, img, inst_map, 4, _random_colour(rng, PALETTE[4], 20), c, r, n_verts=11
                )
            elif cls == 6:  # person (deliberately small + rare)
                a = float(rng.uniform(0.014, 0.021) * size)
                b = float(rng.uniform(0.010, 0.016) * size)
                c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
                reg = _paint_ellipse(
                    rng, img, inst_map, 6, _random_colour(rng, PALETTE[6], 30), c,
                    (int(a), int(b)), float(rng.uniform(0, 180)),
                )
            else:  # vehicle
                w = int(rng.integers(int(0.045 * size), int(0.085 * size)))
                h = int(rng.integers(int(0.024 * size), int(0.045 * size)))
                x = int(rng.integers(0, max(1, size - w)))
                y = int(rng.integers(0, max(1, size - h)))
                reg = _paint_rect(
                    rng, img, inst_map, 7, _random_colour(rng, PALETTE[7], 20), x, y, w, h,
                    float(rng.uniform(0, 180)), edge=True,
                )
            mask[reg > 0] = cls
            drawn.append(
                {
                    "class": int(cls),
                    "region": reg,
                    "instance_id": len(drawn) + 1,
                    "painted_area": int((reg > 0).sum()),
                }
            )

        # instance ids live on the id map so occlusion is measurable
        id_map = np.zeros((size, size), np.int32)
        for d in drawn:
            id_map[d["region"] > 0] = d["instance_id"]
        return img, mask, id_map, drawn

    @staticmethod
    def _collect_instances(inst_map: np.ndarray, mask: np.ndarray, drawn: list) -> list[dict]:
        """Turn the instance-id map into visible boxes + occlusion stats."""
        out: list[dict] = []
        for d in drawn:
            iid = d["instance_id"]
            vis = (inst_map == iid) & (mask == d["class"])
            box = _bbox_from_region(vis.astype(np.uint8))
            if box is None:
                continue
            visible = int(vis.sum())
            painted = max(1, int(d["painted_area"]))
            out.append(
                {
                    "class": int(d["class"]),
                    "bbox": [int(v) for v in box],
                    "visible_area": visible,
                    "painted_area": painted,
                    "visible_frac": round(visible / painted, 4),
                    "occluded": bool(visible / painted < 0.75),
                }
            )
        out.sort(key=lambda o: (o["class"], o["bbox"][1], o["bbox"][0]))
        return out

    @staticmethod
    def _scene_stats(img: np.ndarray, mask: np.ndarray, instances: list[dict]) -> dict:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        areas = np.array([i["visible_area"] for i in instances], float) if instances else np.zeros(1)
        over = float((img.max(axis=2) >= 250).mean())
        return {
            "n_instances": len(instances),
            "instance_density": round(len(instances) / (mask.size / 1e4), 4),  # per 100x100 px
            "median_instance_area": float(np.median(areas)),
            "min_instance_area": float(areas.min()),
            "mean_occlusion": round(
                float(np.mean([1.0 - i["visible_frac"] for i in instances])) if instances else 0.0, 4
            ),
            "overexposed_frac": round(over, 4),
            "contrast": round(float(gray.std()), 3),
            "mean_luma": round(float(gray.mean()), 3),
        }


# --------------------------------------------------------------------------- #
# Dataset materialisation
# --------------------------------------------------------------------------- #


def generate_synthetic_dataset(
    out_dir: str | Path,
    cfg: DataCfg,
    n_scenes: int | None = None,
    split_sizes: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Write a synthetic dataset to ``out_dir`` and return its summary.

    Layout::

        out_dir/images/scene_0000.png
        out_dir/masks/scene_0000.png
        out_dir/boxes/scene_0000.json
        out_dir/meta/scene_0000.json
    """
    out_dir = Path(out_dir)
    for sub in ("images", "masks", "boxes", "meta"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    total = int(n_scenes or cfg.synthetic.n_scenes)
    gen = SyntheticAerialGenerator(cfg.synthetic, seed=cfg.synthetic.seed)

    per_class_pixels = np.zeros(cfg.num_classes, np.int64)
    per_class_instances = np.zeros(cfg.num_classes, np.int64)
    records: list[dict[str, Any]] = []

    for i in range(total):
        scene = gen.generate(i)
        name = f"scene_{i:04d}"
        cv2.imwrite(str(out_dir / "images" / f"{name}.png"), cv2.cvtColor(scene.image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "masks" / f"{name}.png"), scene.mask)
        boxes = [{"class": inst["class"], "bbox": inst["bbox"]} for inst in scene.instances]
        (out_dir / "boxes" / f"{name}.json").write_text(json.dumps(boxes, indent=1))
        (out_dir / "meta" / f"{name}.json").write_text(json.dumps(scene.tags, indent=1))

        per_class_pixels += np.bincount(scene.mask.ravel(), minlength=cfg.num_classes)[: cfg.num_classes]
        for inst in scene.instances:
            per_class_instances[inst["class"]] += 1
        records.append({"name": name, "n_instances": len(boxes), **scene.tags})

    return {
        "root": str(out_dir),
        "n_scenes": total,
        "image_size": [int(gen.syn.base_size), int(gen.syn.base_size)],
        "classes": list(cfg.classes),
        "per_class_pixels": {c: int(per_class_pixels[j]) for j, c in enumerate(cfg.classes)},
        "per_class_instances": {c: int(per_class_instances[j]) for j, c in enumerate(cfg.classes)},
        "scenes": records,
    }
