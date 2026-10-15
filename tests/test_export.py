"""Step 6 tests: TorchScript/ONNX export, graph validity, and numerical parity.

Everything runs on the tiny preset backbone on CPU, so the suite needs neither a
GPU nor a download.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from aerial_vit.config import load_config
from aerial_vit.export import (
    ExportWrapper,
    export_onnx,
    export_torchscript,
    onnx_output_names,
    verify_onnx_parity,
    verify_torchscript,
)
from aerial_vit.models import build_model

IMG = 224
TINY = 14  # patch size -> 16x16 token grid at 224


@pytest.fixture(scope="module")
def tiny_model():
    cfg = load_config("configs/smoke.yaml")
    cfg.backbone.name = "tiny"
    model, _ = build_model(cfg, mode="linear", pretrained=False, device="cpu")
    model.eval()
    return model


@pytest.fixture(scope="module")
def example():
    torch.manual_seed(0)
    return torch.randn(2, 3, IMG, IMG)


# --------------------------------------------------------------------------- #
# wrapper
# --------------------------------------------------------------------------- #


def test_export_wrapper_flattens_to_a_tuple(tiny_model, example):
    out = ExportWrapper(tiny_model).eval()(example)
    assert isinstance(out, tuple)
    assert out[0].shape == (2, 8, IMG, IMG)
    # 1 seg + 3 outputs per detection level, 2 levels for box_strides (1, 2)
    assert len(out) == 1 + 3 * 2
    for t in out:
        assert torch.is_tensor(t)


def test_onnx_output_names_match_wrapper_arity(tiny_model, example):
    names = onnx_output_names(tiny_model)
    out = ExportWrapper(tiny_model).eval()(example)
    assert len(names) == len(out)
    assert names[0] == "seg_logits"


def test_wrapper_is_deterministic_in_eval(tiny_model, example):
    w = ExportWrapper(tiny_model).eval()
    with torch.no_grad():
        a = w(example)[0]
        b = w(example)[0]
    assert torch.allclose(a, b)


# --------------------------------------------------------------------------- #
# TorchScript
# --------------------------------------------------------------------------- #


def test_torchscript_roundtrip_and_parity(tiny_model, example, tmp_path):
    p = export_torchscript(tiny_model, tmp_path / "m.torchscript", example)
    assert p.exists() and p.stat().st_size > 1000
    loaded = torch.jit.load(str(p)).eval()
    with torch.no_grad():
        got = loaded(example)
    assert len(got) == len(ExportWrapper(tiny_model)(example))
    check = verify_torchscript(tiny_model, p, example)
    assert check["passed"], check
    assert check["max_abs_diff"] <= 1e-4
    assert check["argmax_agreement"] == pytest.approx(1.0)


def test_torchscript_handles_a_different_batch_size(tiny_model, tmp_path):
    p = export_torchscript(tiny_model, tmp_path / "m.ts", torch.randn(2, 3, IMG, IMG))
    loaded = torch.jit.load(str(p)).eval()
    with torch.no_grad():
        out = loaded(torch.randn(5, 3, IMG, IMG))
    assert out[0].shape == (5, 8, IMG, IMG)


# --------------------------------------------------------------------------- #
# ONNX
# --------------------------------------------------------------------------- #


def test_onnx_export_produces_a_valid_graph_at_opset_17(tiny_model, example, tmp_path):
    onnx = pytest.importorskip("onnx")
    meta = export_onnx(tiny_model, tmp_path / "m.onnx", example, opset=17)
    assert meta["path"].endswith(".onnx")
    assert meta["opset"] == 17
    assert meta["dynamic_batch"] is True
    model = onnx.load(meta["path"])
    onnx.checker.check_model(model)
    assert model.ir_version >= 7
    opsets = [o.version for o in model.opset_import if o.domain in ("", "ai.onnx")]
    assert 17 in opsets


def test_onnxruntime_matches_torch_within_tolerance(tiny_model, example, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    p = tmp_path / "m.onnx"
    export_onnx(tiny_model, p, example, opset=17)
    check = verify_onnx_parity(tiny_model, p, example, tolerance=1e-3)
    assert check["passed"], check
    assert check["max_abs_diff"] < 1e-3
    assert check["argmax_agreement"] >= 0.999, "segmentation argmax must be stable under export"

    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"images": example.numpy()})
    with torch.no_grad():
        ref = ExportWrapper(tiny_model).eval()(example)
    assert len(got) == len(ref)
    assert np.abs(got[0] - ref[0].numpy()).max() < 1e-3


def test_onnx_dynamic_batch_axis(tiny_model, example, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    p = tmp_path / "m.onnx"
    export_onnx(tiny_model, p, example, opset=17)
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    for bs in (1, 3, 8):
        out = sess.run(None, {"images": torch.randn(bs, 3, IMG, IMG).numpy()})
        assert out[0].shape == (bs, 8, IMG, IMG)


def test_onnx_parity_uses_the_segmentation_argmax(tiny_model, example, tmp_path):
    p = tmp_path / "m.onnx"
    export_onnx(tiny_model, p, example, opset=17)
    check = verify_onnx_parity(tiny_model, p, example)
    assert 0.0 <= check["miou"] <= 1.0
    assert check["miou"] > 0.99, "exported segmentation must reproduce the reference argmax"


def test_onnx_parity_detects_a_perturbed_graph(tiny_model, example, tmp_path):
    """A sanity check that the parity test can actually fail."""
    onnx = pytest.importorskip("onnx")
    p = tmp_path / "m.onnx"
    export_onnx(tiny_model, p, example, opset=17)
    model = onnx.load(str(p))
    # corrupt the final bias of the segmentation 1x1 conv
    for init in model.graph.initializer:
        if init.name.endswith("seg.conv.bias") or "seg" in init.name and "bias" in init.name:
            arr = onnx.numpy_helper.to_array(init) + 5.0
            init.CopyFrom(onnx.numpy_helper.from_array(arr, init.name))
            break
    bad = tmp_path / "bad.onnx"
    onnx.save(model, str(bad))
    check = verify_onnx_parity(tiny_model, bad, example, tolerance=1e-3)
    assert not check["passed"] or check["miou"] < 1.0


def test_export_reports_serialise_to_json(tiny_model, example, tmp_path):
    p = tmp_path / "m.onnx"
    meta = export_onnx(tiny_model, p, example, opset=17)
    check = verify_onnx_parity(tiny_model, p, example)
    blob = json.dumps({"onnx": {**meta, **check}}, default=float)
    back = json.loads(blob)
    assert back["onnx"]["exporter"] in {"dynamo", "torchscript-legacy"}
    assert "max_abs_diff" in back["onnx"]
