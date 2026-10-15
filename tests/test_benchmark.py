"""Step 6/7 tests: latency benchmark plumbing and the quality recorder's parsing."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from aerial_vit.benchmark import (
    BenchConfig,
    benchmark_callable,
    median_or_none,
    onboard_verdict,
    plot_latency,
    summarise_latencies,
)
from aerial_vit.commands.quality import parse_pytest_output

# --------------------------------------------------------------------------- #
# benchmark_callable
# --------------------------------------------------------------------------- #


def test_benchmark_callable_returns_the_documented_fields():
    cfg = BenchConfig(warmup=1, iterations=5, resolution=32, batch_size=2)
    r = benchmark_callable(lambda x: x + 1, cfg, "cpu")
    for key in (
        "iterations",
        "warmup",
        "batch_size",
        "resolution",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "fps",
        "device",
    ):
        assert key in r, key
    assert r["iterations"] == 5 and r["warmup"] == 1
    assert r["batch_size"] == 2 and r["resolution"] == "32x32"
    assert len(r["all_ms"]) == 5


def test_benchmark_callable_percentiles_are_ordered():
    cfg = BenchConfig(warmup=0, iterations=10, resolution=32, batch_size=1)
    r = benchmark_callable(lambda x: torch.relu(x), cfg, "cpu")
    assert r["min_ms"] <= r["p50_ms"] <= r["p95_ms"] <= r["max_ms"]
    assert r["mean_ms"] > 0


def test_benchmark_fps_matches_the_batch_formula():
    cfg = BenchConfig(warmup=1, iterations=5, resolution=16, batch_size=4)
    # a constant-cost matmul keeps per-iteration time in the millisecond range so
    # the 4-decimal rounding of mean_ms cannot dominate the comparison
    weight = torch.randn(192, 192)
    r = benchmark_callable(lambda x: torch.mm(weight, weight), cfg, "cpu")
    assert r["fps"] == pytest.approx(4 * r["fps_per_stream"], rel=1e-9)
    assert r["fps"] == pytest.approx(1000.0 * 4 / r["mean_ms"], rel=0.05)
    assert r["fps_per_stream"] == pytest.approx(1000.0 / r["mean_ms"], rel=0.05)
    assert r["mean_ms"] >= r["p50_ms"] * 0.5  # sanity: a real measurement


def test_benchmark_warmup_does_not_count_toward_iterations():
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1
        return x

    cfg = BenchConfig(warmup=7, iterations=3, resolution=16, batch_size=1)
    benchmark_callable(fn, cfg, "cpu")
    assert calls["n"] == 10


# --------------------------------------------------------------------------- #
# onboard verdict
# --------------------------------------------------------------------------- #


def _row(backend, fps, batch=1, res="224x224", p50=None, device="cpu"):
    return {
        "backend": backend,
        "fps": fps,
        "batch_size": batch,
        "resolution": res,
        "p50_ms": p50 if p50 is not None else round(1000.0 / fps, 3),
        "p95_ms": p50 if p50 is not None else round(1000.0 / fps, 3),
        "device": device,
    }


def test_onboard_verdict_feasible_when_fastest_beats_the_requirement():
    rows = [_row("torchscript", 40.0), _row("onnxruntime", 8.0)]
    ob = onboard_verdict(rows, requirement_hz=10.0)
    assert ob["verdict"] == "feasible"
    assert ob["best_backend"] == "torchscript"
    assert ob["best_fps"] == 40.0
    assert ob["headroom_ratio"] == pytest.approx(4.0)


def test_onboard_verdict_not_feasible_when_below_the_requirement():
    rows = [_row("onnxruntime", 3.0), _row("pytorch-eager", 6.0)]
    ob = onboard_verdict(rows, requirement_hz=10.0)
    assert ob["verdict"] == "not feasible"
    assert ob["best_backend"] == "pytorch-eager"


def test_onboard_verdict_reports_each_backend_separately():
    rows = [
        _row("onnxruntime", 5.0, res="224x224"),
        _row("onnxruntime", 3.0, res="320x320"),
        _row("pytorch-eager", 50.0, device="mps"),
    ]
    ob = onboard_verdict(rows, requirement_hz=10.0)
    per = ob["per_backend_batch1"]
    assert per["onnxruntime"]["fps"] == 5.0
    assert per["onnxruntime"]["resolution"] == "224x224"
    assert ob["preferred_backend_batch1_fps"] == 5.0
    assert ob["best_backend"] == "pytorch-eager"


def test_onboard_verdict_uses_batch1_only():
    rows = [_row("onnxruntime", 1.0, batch=1), _row("onnxruntime", 500.0, batch=8)]
    ob = onboard_verdict(rows, requirement_hz=10.0)
    assert ob["best_fps"] == 1.0
    assert ob["verdict"] == "not feasible"


def test_onboard_verdict_handles_no_measurements():
    ob = onboard_verdict([], requirement_hz=10.0)
    assert ob["verdict"] == "no measurements"
    assert ob["best_fps"] is None


# --------------------------------------------------------------------------- #
# summary + figure
# --------------------------------------------------------------------------- #


def test_summarise_latencies_keys_by_backend_resolution_batch():
    rows = [{**_row("onnxruntime", 10.0), "resolution": "224x224", "batch_size": 1}]
    s = summarise_latencies(rows)
    assert "onnxruntime|224x224|b1" in s
    assert s["onnxruntime|224x224|b1"]["fps"] == 10.0


def test_plot_latency_renders(tmp_path):
    rows = [
        _row("pytorch-eager", 20.0, batch=1),
        _row("pytorch-eager", 60.0, batch=8),
        _row("torchscript", 60.0, batch=1),
        _row("torchscript", 200.0, batch=8),
        _row("onnxruntime", 4.0, batch=1, res="320x320"),
        _row("onnxruntime", 5.0, batch=8, res="320x320"),
    ]
    p = plot_latency(rows, tmp_path / "latency.png", requirement_hz=10.0)
    assert p.exists() and p.stat().st_size > 3000


def test_median_or_none():
    assert median_or_none([1.0, 3.0, 2.0]) == pytest.approx(2.0)
    assert median_or_none([None, None]) is None
    assert median_or_none([]) is None


# --------------------------------------------------------------------------- #
# quality parsing
# --------------------------------------------------------------------------- #

SAMPLE = """\
tests/test_data.py .................                                      [ 12%]
---------- coverage: platform darwin, python 3.11.14 ----------
Name                          Stmts   Miss  Cover
-------------------------------------------------
aerial_vit/config.py             120     18    85%
TOTAL                           4200    980    77%
-------------------------------------------------
72 passed, 2 deselected in 11.02s
"""


def test_parse_pytest_output_reads_passed_and_deselected():
    stats = parse_pytest_output(SAMPLE)
    assert stats["tests_passed"] == 72
    assert stats["tests_deselected"] == 2
    assert stats["tests_failed"] is None
    assert stats["tests_collected"] == 74


def test_parse_pytest_output_counts_failures():
    text = "3 failed, 10 passed, 1 deselected in 4.2s"
    stats = parse_pytest_output(text)
    assert stats["tests_passed"] == 10
    assert stats["tests_failed"] == 3
    assert stats["tests_deselected"] == 1
    assert stats["tests_collected"] == 14


def test_parse_pytest_output_is_null_safe():
    stats = parse_pytest_output("")
    assert stats["tests_passed"] is None
    assert stats["tests_collected"] is None


def test_quality_report_is_json_serialisable(tmp_path):
    blob = json.dumps(
        {"tests_passed": 72, "coverage_pct": 77.0, "np": float(np.float64(1.5))}, default=float
    )
    assert json.loads(blob)["coverage_pct"] == 77.0
