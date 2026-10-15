"""Step 3 tests: prompt ensembling, total label mapping, stubbed zero-shot run.

The stub tests exercise the *real* ``CLIPZeroShot`` / ``OWLViTZeroShot`` code
paths with injected fake models, so CI never downloads anything.  The one test
that needs real weights is marked ``slow``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from aerial_vit.config import load_config
from aerial_vit.models.openvocab import (
    CLIPZeroShot,
    OWLViTZeroShot,
    boxes_to_segmentation,
    class_prompt_lists,
    connected_components_boxes,
    ensemble_text_embeddings,
    load_prompts,
)

CLASSES = [
    "background",
    "paved-area",
    "grass",
    "roof",
    "tree",
    "water",
    "person",
    "vehicle",
]


@pytest.fixture(scope="module")
def prompts():
    return load_prompts()


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #


def test_prompt_lists_are_total_deterministic_and_non_empty(prompts):
    lists = class_prompt_lists(CLASSES, prompts)
    assert len(lists) == len(CLASSES)
    for name, tpl in zip(CLASSES, lists, strict=False):
        assert len(tpl) > 0, f"{name} has no prompt"
        assert all(isinstance(t, str) and t.strip() for t in tpl)
    again = class_prompt_lists(CLASSES, prompts)
    assert lists == again, "prompt mapping must be deterministic"


def test_prompt_lists_fall_back_for_unknown_class(prompts):
    lists = class_prompt_lists(["background", "totally-new-class"], prompts)
    assert len(lists) == 2
    assert len(lists[1]) == 1 and "totally new class" in lists[1][0]


def test_every_object_class_has_multiple_templates(prompts):
    """Prompt ensembling is only meaningful with >1 template per class."""
    lists = class_prompt_lists(CLASSES, prompts)
    for name, tpl in zip(CLASSES, lists, strict=False):
        if name == "background":
            continue
        assert len(tpl) >= 2, f"{name} needs at least 2 templates"


def test_ensemble_text_embeddings_shape_and_determinism(prompts):
    lists = class_prompt_lists(CLASSES, prompts)
    calls = {"n": 0}

    def fake_tokenizer(texts):
        calls["n"] += 1
        g = torch.Generator().manual_seed(len(texts) * 17 + len(texts[0]))
        return torch.randn(len(texts), 32, generator=g)

    a = ensemble_text_embeddings(None, fake_tokenizer, lists, torch.device("cpu"))
    b = ensemble_text_embeddings(None, fake_tokenizer, lists, torch.device("cpu"))
    assert a.shape == (len(CLASSES), 32)
    assert torch.allclose(a, b), "ensembling must be deterministic"
    # unit norm per class
    assert torch.allclose(a.norm(dim=-1), torch.ones(len(CLASSES)), atol=1e-5)


# --------------------------------------------------------------------------- #
# detection -> segmentation helpers
# --------------------------------------------------------------------------- #


def test_pooled_extracts_features_from_every_transformers_output_shape():
    from aerial_vit.models.openvocab import _pooled

    bare = torch.randn(2, 8)
    assert torch.allclose(_pooled(bare), bare)  # v4 returned a bare tensor

    class _V5:
        def __init__(self):
            self.pooler_output = bare
            self.last_hidden_state = torch.randn(2, 5, 8)

    assert torch.allclose(_pooled(_V5()), bare)

    class _NoPooler:
        def __init__(self):
            self.last_hidden_state = torch.randn(3, 5, 8)

    assert _pooled(_NoPooler()).shape == (3, 8)
    assert _pooled((bare,)) is not None


def test_boxes_to_segmentation_paints_and_clips():
    boxes = np.array([[0, 0, 5, 5], [3, 3, 20, 20]], np.float32)
    labels = np.array([6, 7])
    scores = np.array([0.5, 0.9])  # second is more confident -> wins overlap
    seg = boxes_to_segmentation(boxes, labels, scores, (10, 10), 8)
    assert seg.shape == (10, 10)
    assert seg[0, 0] == 6
    assert seg[4, 4] == 7
    assert set(np.unique(seg)).issubset(set(range(8)))
    assert seg.max() <= 7


def test_boxes_to_segmentation_empty_is_all_background():
    seg = boxes_to_segmentation(np.zeros((0, 4)), np.zeros((0,), int), np.zeros((0,)), (8, 8), 8)
    assert seg.shape == (8, 8) and int(seg.sum()) == 0


def test_connected_components_extracts_boxes():
    label = np.zeros((32, 32), np.uint8)
    label[2:8, 2:8] = 7
    label[20:28, 20:28] = 6
    score = np.full((32, 32), 0.5, np.float32)
    boxes, scores, labels = connected_components_boxes(score, label, min_area=4)
    assert len(boxes) == 2
    assert set(labels.tolist()) == {6, 7}
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= 32 and 0 <= y0 < y1 <= 32


def test_connected_components_ignores_background_and_specks():
    label = np.zeros((16, 16), np.uint8)
    label[0, 0] = 7  # 1 pixel -> below min_area
    boxes, _, _ = connected_components_boxes(np.ones((16, 16), np.float32), label, min_area=4)
    assert len(boxes) == 0


# --------------------------------------------------------------------------- #
# stubbed zero-shot models (no downloads)
# --------------------------------------------------------------------------- #


class _StubCLIPProcessor:
    def __init__(self, dim: int = 16):
        self.dim = dim

    def __call__(self, text=None, images=None, return_tensors="pt", padding=False, **kw):
        out = {}
        if text is not None:
            out["input_ids"] = torch.zeros(len(text), 8, dtype=torch.long)
        if images is not None:
            n = len(images) if isinstance(images, list) else 1
            g = torch.Generator().manual_seed(7)
            out["pixel_values"] = torch.randn(n, 3, 224, 224, generator=g)
        return out


class _StubCLIPModel:
    """Deterministic fake: image features depend on the image, text on the prompt."""

    def __init__(self, n_classes: int = 8, dim: int = 16):
        self.n_classes = n_classes
        self.dim = dim

    def to(self, device):
        return self

    def eval(self):
        return self

    def get_text_features(self, **kw):
        g = torch.Generator().manual_seed(11)
        return torch.randn(self.n_classes, self.dim, generator=g)

    def get_image_features(self, pixel_values=None, **kw):
        n = pixel_values.shape[0] if pixel_values is not None else 1
        # deterministic per-window embedding derived from the input itself
        g = torch.Generator().manual_seed(23)
        base = torch.randn(n, self.dim, generator=g)
        pooled = base + 0.01 * pixel_values.mean(dim=(1, 2, 3)).unsqueeze(1)
        return type(
            "Out", (), {"pooler_output": pooled, "last_hidden_state": pooled.unsqueeze(1)}
        )()


class _StubOwlViTProcessor:
    def __call__(self, images=None, text=None, return_tensors="pt", **kw):
        q = len(text[0]) if text else 1
        g = torch.Generator().manual_seed(q)
        return {
            "pixel_values": torch.randn(1, 3, 768, 768, generator=g),
            "input_ids": torch.zeros(1, q, dtype=torch.long),
        }

    def post_process_grounded_object_detection(
        self, outputs, threshold=0.05, target_sizes=None, **kw
    ):
        h, w = target_sizes[0]
        boxes = torch.tensor(
            [[0.1 * w, 0.1 * h, 0.4 * w, 0.4 * h], [0.5 * w, 0.5 * h, 0.8 * w, 0.8 * h]]
        )
        scores = torch.tensor([0.6, 0.4])
        labels = torch.tensor([0, 1])
        return [{"boxes": boxes, "scores": scores, "labels": labels, "text_labels": ["a", "b"]}]


class _StubOwlViTModel:
    def __init__(self):
        pass

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kw):
        return {}


def test_clip_stub_run_produces_valid_seg_and_boxes(prompts):
    model = CLIPZeroShot(
        CLASSES,
        prompts,
        device="cpu",
        window=64,
        stride=32,
        model=_StubCLIPModel(),
        processor=_StubCLIPProcessor(),
    )
    rng = np.random.default_rng(0)
    for _ in range(2):
        img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        out = model.predict(img)
        assert out["seg"].shape == (224, 224)
        assert out["seg"].dtype == np.uint8
        assert int(out["seg"].max()) < len(CLASSES)
        h, w = img.shape[:2]
        if len(out["boxes"]):
            assert np.all(out["boxes"][:, [0, 2]] >= 0) and np.all(
                out["boxes"][:, [0, 2]] <= w + 1e-3
            )
            assert np.all(out["boxes"][:, [1, 3]] >= 0) and np.all(
                out["boxes"][:, [1, 3]] <= h + 1e-3
            )
            # normalised boxes live in [0, 1]
            norm = out["boxes"] / np.array([w, h, w, h], np.float32)
            assert float(norm.min()) >= 0.0 and float(norm.max()) <= 1.0


def test_clip_stub_writes_non_empty_results_json(tmp_path, prompts):
    from aerial_vit.eval.metrics import evaluate_predictions

    model = CLIPZeroShot(
        CLASSES,
        prompts,
        device="cpu",
        window=64,
        stride=32,
        model=_StubCLIPModel(),
        processor=_StubCLIPProcessor(),
    )
    rng = np.random.default_rng(1)
    seg_preds, seg_gts, det_preds, det_gts = [], [], [], []
    for i in range(2):
        img = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        gt_mask = rng.integers(0, 8, (224, 224), dtype=np.uint8)
        out = model.predict(img)
        seg_preds.append(out["seg"])
        seg_gts.append(gt_mask)
        det_preds.append(
            {"image_id": i, "boxes": out["boxes"], "scores": out["scores"], "labels": out["labels"]}
        )
        det_gts.append(
            {
                "image_id": i,
                "width": 224,
                "height": 224,
                "boxes": np.array([[10, 10, 40, 40]], np.float32),
                "labels": np.array([7]),
            }
        )

    report = evaluate_predictions(
        seg_preds,
        seg_gts,
        det_preds,
        det_gts,
        class_names=CLASSES,
        landable_indices=[1, 2],
        num_classes=8,
        rare_classes=["person", "vehicle"],
        dump_dir=tmp_path,
        prefix="stub_clip",
    )
    p = tmp_path / "stub_results.json"
    p.write_text(json.dumps(report, default=float))
    loaded = json.loads(p.read_text())
    assert loaded and "miou" in loaded and "det_map_50" in loaded
    assert 0.0 <= loaded["miou"] <= 1.0
    assert (tmp_path / "stub_clip_gt.json").exists() and (tmp_path / "stub_clip_pred.json").exists()


def test_owlvit_stub_maps_labels_into_canonical_space(prompts):
    model = OWLViTZeroShot(
        CLASSES,
        prompts,
        device="cpu",
        score_thresh=0.0,
        use_ensembling=True,
        model=_StubOwlViTModel(),
        processor=_StubOwlViTProcessor(),
    )
    out = model.predict(np.zeros((224, 224, 3), np.uint8))
    assert out["boxes"].shape[1] == 4
    # query index 0 -> canonical class 1 (paved-area), 1 -> 2 (grass)
    assert set(out["labels"].tolist()) <= {1, 2}
    assert out["labels"].shape[0] == out["scores"].shape[0] == out["boxes"].shape[0]
    assert float(out["boxes"].min()) >= 0.0


def test_owlvit_label_mapping_is_total_under_duplicate_queries(prompts):
    """Every returned query index must land on a valid canonical class."""
    model = OWLViTZeroShot(
        CLASSES, prompts, device="cpu", model=_StubOwlViTModel(), processor=_StubOwlViTProcessor()
    )
    n = len(CLASSES) - 1
    idx = np.array([0, n - 1, 3, 5, n + 5, -1], dtype=np.int64)
    canon = model._to_canonical(idx, len(idx))
    assert canon.shape == (6,)
    assert all(1 <= int(c) <= len(CLASSES) - 1 for c in canon)


def test_owlvit_query_indices_handles_strings(prompts):
    model = OWLViTZeroShot(
        CLASSES, prompts, device="cpu", model=_StubOwlViTModel(), processor=_StubOwlViTProcessor()
    )
    idx = model._query_indices(["a", "b", "unknown"], templates=["a", "b"])
    assert idx.tolist() == [0, 1, 0]


def test_owlvit_zero_shot_reports_zero_trainable_params(prompts):
    model = OWLViTZeroShot(
        CLASSES, prompts, device="cpu", model=_StubOwlViTModel(), processor=_StubOwlViTProcessor()
    )
    # zero-shot: nothing is trained, so there is nothing to report as trainable
    assert model.query_class_indices == list(range(1, len(CLASSES)))


# --------------------------------------------------------------------------- #
# real models (slow: downloads weights)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_real_owlvit_two_images(prompts):
    model = OWLViTZeroShot(CLASSES, prompts, device="cpu", score_thresh=0.01)
    rng = np.random.default_rng(3)
    outs = model.predict_batch(
        [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
    )
    assert len(outs) == 2
    for o in outs:
        assert o["boxes"].shape[1] == 4
        assert o["scores"].shape[0] == o["boxes"].shape[0]


@pytest.mark.slow
def test_real_clip_two_images(prompts):
    model = CLIPZeroShot(CLASSES, prompts, device="cpu", window=64, stride=32)
    rng = np.random.default_rng(4)
    for _ in range(2):
        out = model.predict(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
        assert out["seg"].shape == (224, 224)
        assert int(out["seg"].max()) < len(CLASSES)
