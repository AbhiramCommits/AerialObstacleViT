"""Visualisation helpers: colourised masks, box overlays, montages."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Fixed colours per canonical class (RGB) used by every figure.
CLASS_COLOURS: dict[int, tuple[int, int, int]] = {
    0: (60, 60, 60),
    1: (108, 110, 116),
    2: (84, 128, 62),
    3: (200, 96, 76),
    4: (46, 124, 54),
    5: (58, 110, 200),
    6: (230, 60, 60),
    7: (240, 240, 240),
}

CLASS_NAMES = [
    "background",
    "paved-area",
    "grass",
    "roof",
    "tree",
    "water",
    "person",
    "vehicle",
]


def colorize_mask(mask: np.ndarray, num_classes: int = 8) -> np.ndarray:
    """(H, W) index mask -> (H, W, 3) uint8 RGB."""
    out = np.zeros(mask.shape + (3,), np.uint8)
    for c in range(num_classes):
        out[mask == c] = CLASS_COLOURS.get(c, (255, 0, 255))
    return out


def overlay_mask(
    image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, num_classes: int = 8
) -> np.ndarray:
    """Blend a colourised mask over an RGB image."""
    colour = colorize_mask(mask, num_classes)
    return np.clip(
        image.astype(np.float32) * (1 - alpha) + colour.astype(np.float32) * alpha, 0, 255
    ).astype(np.uint8)


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    colour: tuple[int, int, int] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw xyxy boxes on an RGB (H, W, 3) uint8 image. Returns a copy."""
    import cv2

    img = image.copy()
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    for i, b in enumerate(boxes):
        c = colour or CLASS_COLOURS.get(int(labels[i]) if labels is not None else 0, (255, 255, 0))
        x0, y0, x1, y1 = (int(round(v)) for v in b)
        cv2.rectangle(img, (x0, y0), (x1, y1), c, thickness)
        if labels is not None:
            txt = (
                f"{CLASS_NAMES[int(labels[i])] if int(labels[i]) < len(CLASS_NAMES) else labels[i]}"
            )
            if scores is not None:
                txt += f" {float(scores[i]):.2f}"
            cv2.putText(
                img, txt, (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1, cv2.LINE_AA
            )
    return img


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return colorize_mask(img)
    return img


def montage(
    panels: list[np.ndarray],
    cols: int = 4,
    pad: int = 4,
    bg: int = 20,
) -> np.ndarray:
    """Tile RGB (or index-mask) images into a single grid image."""
    if not panels:
        return np.zeros((32, 32, 3), np.uint8)
    panels = [_to_rgb(p) for p in panels]
    h = max(p.shape[0] for p in panels)
    w = max(p.shape[1] for p in panels)
    n = len(panels)
    rows = int(np.ceil(n / cols))
    out = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), bg, np.uint8)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        out[y : y + p.shape[0], x : x + p.shape[1]] = p
    return out


def save_image(path: str | Path, img: np.ndarray) -> Path:
    import cv2

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if img.ndim == 3 and img.shape[2] == 3:
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(p), img)
    return p


def label_panel(image: np.ndarray, title: str, scale: float = 1.0) -> np.ndarray:
    """Draw a title bar above an image (simple, dependency-free)."""
    import cv2

    img = _to_rgb(image)
    bar = np.full((20, img.shape[1], 3), 15, np.uint8)
    cv2.putText(
        bar, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA
    )
    out = np.concatenate([bar, img], axis=0)
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return out


__all__ = [
    "CLASS_COLOURS",
    "CLASS_NAMES",
    "colorize_mask",
    "draw_boxes",
    "label_panel",
    "montage",
    "overlay_mask",
    "save_image",
]
