"""Open-vocabulary zero-shot baselines: OWL-ViT (detection) and CLIP (dense).

Both are used *frozen and zero-shot* -- no fine-tuning, no prompt tuning -- which
is the whole point of the comparison: how far does a generic vision-language
model get on a domain (nadir UAV imagery) it was never trained for?

``OWLViTZeroShot``
    ``google/owlvit-base-patch32``.  Emits boxes + scores per text query.  The
    canonical taxonomy's 7 object classes are turned into prompts via
    ``configs/prompts.yaml`` and ensembled by averaging text embeddings.  Boxes
    are converted to a segmentation map by filling them (detection -> seg).

``CLIPZeroShot``
    ``openai/clip-vit-base-patch32`` used as a *dense patch classifier*: a sliding
    window is classified against every canonical class (including background),
    producing a score map that gives both a segmentation map and, by
    connected-component extraction on non-background classes, boxes.

Both write predictions in the *same* label space and image geometry as the
task-trained model, so ``eval/metrics.py`` computes identical metrics on all
three methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import yaml

from aerial_vit.utils.logging import get_logger

log = get_logger("aerialvit.openvocab")

__all__ = [
    "load_prompts",
    "ensemble_text_embeddings",
    "OWLViTZeroShot",
    "CLIPZeroShot",
    "boxes_to_segmentation",
    "connected_components_boxes",
]


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #


def load_prompts(path: str | Path = "configs/prompts.yaml") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml"
    return yaml.safe_load(p.read_text())


def class_prompt_lists(classes: Sequence[str], prompts: dict[str, Any]) -> list[list[str]]:
    """One template list per canonical class, in taxonomy order.

    Every class gets a non-empty list; a class missing from the prompt file
    falls back to a templated name so the label mapping stays **total**.
    """
    bank = prompts.get("classes", {})
    out: list[list[str]] = []
    for c in classes:
        tpl = list(bank.get(c, []) or [])
        if not tpl:
            tpl = [f"an aerial photo of {c.replace('-', ' ')}"]
        out.append(tpl)
    return out


def ensemble_text_embeddings(model, tokenizer_fn, prompt_lists: Sequence[Sequence[str]], device):
    """Average the text embeddings of every template for each class.

    ``tokenizer_fn(texts) -> embeddings`` is supplied by the caller so this
    helper is testable without loading a model.
    """
    embs = []
    for templates in prompt_lists:
        e = tokenizer_fn(list(templates))
        e = F.normalize(e, dim=-1)
        embs.append(F.normalize(e.mean(dim=0), dim=-1))
    return torch.stack(embs)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def boxes_to_segmentation(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    size: tuple[int, int],
    num_classes: int,
    score_thresh: float = 0.0,
) -> np.ndarray:
    """Rasterise detections into a label map (detection -> segmentation).

    Boxes are painted in ascending confidence order so that the *most confident*
    detection wins where boxes overlap.
    """
    h, w = size
    seg = np.zeros((h, w), np.uint8)
    if boxes is None or len(boxes) == 0:
        return seg
    order = np.argsort(scores)
    for i in order:
        if scores[i] < score_thresh:
            continue
        x0, y0, x1, y1 = boxes[i]
        x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
        x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
        if x1 <= x0 or y1 <= y0:
            continue
        seg[y0:y1, x0:x1] = int(np.clip(labels[i], 1, num_classes - 1))
    return seg


def connected_components_boxes(
    score_map: np.ndarray,
    label_map: np.ndarray,
    min_area: int = 6,
    max_boxes: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract boxes from a dense label map via connected components.

    Returns ``(boxes xyxy, scores, labels)`` sorted by score.
    """
    import cv2

    boxes, scores, labels = [], [], []
    h, w = label_map.shape
    for cls in np.unique(label_map):
        if int(cls) == 0:
            continue
        binary = (label_map == cls).astype(np.uint8)
        n, comp, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < min_area:
                continue
            m = comp == i
            sc = float(score_map[m].mean()) if score_map.shape == label_map.shape else 1.0
            boxes.append([x, y, x + bw, y + bh])
            scores.append(sc)
            labels.append(int(cls))
    if not boxes:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.int64)
    boxes = np.asarray(boxes, np.float32)
    scores = np.asarray(scores, np.float32)
    labels = np.asarray(labels, np.int64)
    keep = np.argsort(-scores)[:max_boxes]
    return boxes[keep], scores[keep], labels[keep]


def _to_pil(image_rgb: np.ndarray):
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(image_rgb.astype(np.uint8)))


def _pooled(output) -> torch.Tensor:
    """Extract the projected embedding from a CLIP feature call.

    ``transformers`` v5 returns a ``BaseModelOutputWithPooling`` whose
    ``pooler_output`` is *already* the projected feature; v4 returned a bare
    tensor.  This handles both, plus the tuple form.
    """
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _pooled(output[0])
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        val = getattr(output, attr, None)
        if val is not None:
            return val
    last = getattr(output, "last_hidden_state", None)
    if last is not None:
        return last[:, 0]
    raise TypeError(f"cannot extract embeddings from {type(output).__name__}")


# --------------------------------------------------------------------------- #
# OWL-ViT
# --------------------------------------------------------------------------- #


class OWLViTZeroShot:
    """Zero-shot open-vocabulary detection with OWL-ViT."""

    name = "owlvit"

    def __init__(
        self,
        classes: Sequence[str],
        prompts: dict[str, Any],
        model_id: str = "google/owlvit-base-patch32",
        device: str | torch.device = "cpu",
        score_thresh: float = 0.05,
        max_dets: int = 50,
        use_ensembling: bool = True,
        model: Any = None,
        processor: Any = None,
    ):
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.classes = list(classes)
        self.device = torch.device(device)
        self.score_thresh = float(score_thresh)
        self.max_dets = int(max_dets)
        self.use_ensembling = bool(use_ensembling)
        self.model_id = model_id
        self.processor = processor or OwlViTProcessor.from_pretrained(model_id)
        if model is not None:
            self.model = model.to(self.device).eval()
        else:
            self.model = OwlViTForObjectDetection.from_pretrained(model_id).to(self.device).eval()
        self.prompt_lists = class_prompt_lists(self.classes, prompts)
        # OWL-ViT cannot be asked for "background"; queries cover classes 1..K-1.
        # Every template becomes its own text query (the standard multi-template
        # OWL-ViT recipe) and each query index is mapped back to its class.
        self.query_class_indices = list(range(1, len(self.classes)))
        self.flat_queries: list[str] = []
        query_to_class: list[int] = []
        for ci in self.query_class_indices:
            tpl = self.prompt_lists[ci]
            use = tpl if use_ensembling else tpl[:1]
            for t in use:
                self.flat_queries.append(t)
                query_to_class.append(ci)
        self.query_to_class = np.asarray(query_to_class, dtype=np.int64)

    def _queries(self) -> list[list[str]]:
        return [self.prompt_lists[i] for i in self.query_class_indices]

    def predict(self, image_rgb: np.ndarray) -> dict[str, Any]:
        """Returns boxes/scores/labels in canonical label space, pixels."""
        h, w = image_rgb.shape[:2]
        out = self.predict_batch([image_rgb])[0]
        out["image_size"] = (h, w)
        return out

    @torch.no_grad()
    def predict_batch(self, images, batch_size: int | None = None) -> list[dict[str, Any]]:
        """Predict on a list of RGB uint8 images, one forward per image."""
        results: list[dict[str, Any]] = []
        for img in images:
            pil = _to_pil(img)
            h, w = img.shape[:2]
            inp = self.processor(images=pil, text=[self.flat_queries], return_tensors="pt")
            inp = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
            outputs = self.model(**inp)
            res = self.processor.post_process_grounded_object_detection(
                outputs, threshold=self.score_thresh, target_sizes=[(h, w)]
            )[0]
            boxes = np.asarray(res["boxes"].cpu(), np.float32).reshape(-1, 4)
            scores = np.asarray(res["scores"].cpu(), np.float32).reshape(-1)
            qidx = self._query_indices(res.get("labels"), self.flat_queries)
            labels = self._to_canonical(qidx, len(boxes))

            if boxes.shape[0] > self.max_dets:
                keep = np.argsort(-scores)[: self.max_dets]
                boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            results.append(
                {
                    "boxes": boxes.astype(np.float32),
                    "scores": scores.astype(np.float32),
                    "labels": labels.astype(np.int64),
                }
            )
        return results

    # -- label plumbing ---------------------------------------------------- #
    @staticmethod
    def _query_indices(labels: Any, templates: Sequence[str] | None = None) -> np.ndarray:
        """Normalise HF's label encodings into query indices.

        v5 returns integer query indices; some versions return the matched text.
        Both are handled so the canonical mapping is **total** either way.
        """
        if labels is None:
            return np.zeros((0,), np.int64)
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        arr = np.asarray(labels)
        if arr.size == 0:
            return np.zeros((0,), np.int64)
        if arr.dtype.kind in "OUS":
            mapping: dict[str, int] = {}
            for i, t in enumerate(templates or []):
                mapping.setdefault(str(t), i)
            return np.array([mapping.get(str(s), 0) for s in arr.reshape(-1)], dtype=np.int64)
        return arr.reshape(-1).astype(np.int64)

    def _to_canonical(self, query_idx: np.ndarray, n: int) -> np.ndarray:
        """Map flattened query index -> canonical class index."""
        if n == 0:
            return np.zeros((0,), np.int64)
        table = self.query_to_class if len(self.query_to_class) else np.array([1], np.int64)
        qi = np.clip(query_idx.reshape(-1), 0, len(table) - 1)
        return table[qi]


# --------------------------------------------------------------------------- #
# CLIP (dense patch classification)
# --------------------------------------------------------------------------- #


class CLIPZeroShot:
    """Dense zero-shot classification with CLIP over a sliding window."""

    name = "clip"

    def __init__(
        self,
        classes: Sequence[str],
        prompts: dict[str, Any],
        model_id: str = "openai/clip-vit-base-patch32",
        device: str | torch.device = "cpu",
        window: int = 96,
        stride: int = 48,
        batch_windows: int = 32,
        score_thresh: float = 0.0,
        model: Any = None,
        processor: Any = None,
    ):
        from transformers import CLIPModel, CLIPProcessor

        self.classes = list(classes)
        self.device = torch.device(device)
        self.window = int(window)
        self.stride = int(stride)
        self.batch_windows = int(batch_windows)
        self.score_thresh = float(score_thresh)
        self.model_id = model_id
        self.processor = processor or CLIPProcessor.from_pretrained(model_id)
        if model is not None:
            self.model = model.to(self.device).eval()
        else:
            self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.prompt_lists = class_prompt_lists(self.classes, prompts)
        self.text_embeds = self._text_embeddings()

    # -- text side --------------------------------------------------------- #
    def _text_embeddings(self) -> torch.Tensor:
        embs = []
        with torch.no_grad():
            for templates in self.prompt_lists:
                inp = self.processor(text=list(templates), return_tensors="pt", padding=True)
                inp = {k: v.to(self.device) for k, v in inp.items() if torch.is_tensor(v)}
                e = _pooled(self.model.get_text_features(**inp))
                e = F.normalize(e, dim=-1).mean(dim=0)
                embs.append(F.normalize(e, dim=-1))
        return torch.stack(embs)  # (K, D)

    # -- image side -------------------------------------------------------- #
    def _windows(self, h: int, w: int) -> list[tuple[int, int, int, int]]:
        out = []
        ys = list(range(0, max(1, h - self.window + 1), self.stride))
        xs = list(range(0, max(1, w - self.window + 1), self.stride))
        for y in ys:
            for x in xs:
                out.append((x, y, min(x + self.window, w), min(y + self.window, h)))
        return out

    @torch.no_grad()
    def score_map(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(score_map KxHxW, label_map HxW)`` at window resolution."""
        h, w = image_rgb.shape[:2]
        wins = self._windows(h, w)
        patches, coords = [], []
        for x0, y0, x1, y1 in wins:
            patches.append(image_rgb[y0:y1, x0:x1])
            coords.append((x0, y0, x1, y1))

        probs = np.zeros((len(self.classes), h, w), np.float32)
        counts = np.zeros((h, w), np.float32)
        for i in range(0, len(patches), self.batch_windows):
            batch = patches[i : i + self.batch_windows]
            inp = self.processor(images=[_to_pil(p) for p in batch], return_tensors="pt")
            inp = {k: v.to(self.device) for k, v in inp.items() if torch.is_tensor(v)}
            feats = F.normalize(_pooled(self.model.get_image_features(**inp)), dim=-1)
            logits = 100.0 * feats @ self.text_embeds.t()
            p = logits.softmax(dim=-1).cpu().numpy()
            for j, (x0, y0, x1, y1) in enumerate(coords[i : i + self.batch_windows]):
                probs[:, y0:y1, x0:x1] += p[j][:, None, None]
                counts[y0:y1, x0:x1] += 1.0
        counts[counts == 0] = 1.0
        probs /= counts[None]
        label = probs.argmax(axis=0).astype(np.uint8)
        return probs, label

    def predict(self, image_rgb: np.ndarray) -> dict[str, Any]:
        h, w = image_rgb.shape[:2]
        probs, label = self.score_map(image_rgb)
        seg = label
        # Non-background components become detections.  Trees/grass/roads are
        # broad regions, so we only extract instances for the object-like
        # classes to keep the mAP comparison meaningful.
        obj_labels = np.zeros_like(label)
        for cls in (3, 4, 6, 7):  # roof, tree, person, vehicle
            obj_labels[label == cls] = cls
        conf = probs.max(axis=0)
        boxes, scores, labels = connected_components_boxes(conf, obj_labels)
        return {
            "seg": seg.astype(np.uint8),
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "probs": probs,
            "image_size": (h, w),
        }
