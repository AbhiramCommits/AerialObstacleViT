"""Report-generator tests: tables come from artifacts, never from thin air."""

from __future__ import annotations

import csv
import json

import pytest

from aerial_vit import reporting
from aerial_vit.reporting import (
    TABLE_ORDER,
    build_tables,
    table_ablation_params,
    table_foundation_vs_tasktrained,
    table_quality,
    write_report,
)


@pytest.fixture()
def tmp_results(tmp_path, monkeypatch):
    """Point the report builders at a scratch results directory."""
    monkeypatch.setattr(reporting, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(reporting, "ARTIFACTS_DIR", tmp_path)
    return tmp_path


def _write_csv(p, rows, fields):
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_missing_artifacts_render_an_explicit_notice(tmp_results):
    table = table_foundation_vs_tasktrained()
    assert "not available yet" in table
    assert "0.0" not in table


def test_foundation_table_reads_every_row(tmp_results):
    fields = [
        "method",
        "trainable_params",
        "total_params",
        "trainable_pct",
        "miou",
        "landing_zone_iou",
        "map_50",
        "map_50_95",
        "rare_map_50",
        "rare_classes",
        "ms_per_image",
        "device",
        "n_test",
        "zero_shot",
        "source_artifact",
    ]
    _write_csv(
        tmp_results / "foundation_vs_tasktrained.csv",
        [
            dict.fromkeys(fields, "")
            | {
                "method": "DINOv2+heads (linear)",
                "miou": "0.4967",
                "map_50": "0.3658",
                "trainable_params": "2083733",
                "device": "cpu",
                "zero_shot": "False",
            },
            dict.fromkeys(fields, "")
            | {
                "method": "owlvit (zero-shot)",
                "miou": "0.0751",
                "map_50": "0.1853",
                "trainable_params": "0",
                "device": "cpu",
                "zero_shot": "True",
            },
        ],
        fields,
    )
    table = table_foundation_vs_tasktrained()
    assert "DINOv2+heads (linear)" in table and "owlvit (zero-shot)" in table
    assert "0.4967" in table and "2,083,733" in table


def test_ablation_pivot_ignores_the_config_block(tmp_results):
    summary = {
        "linear@0.100": {
            "mode": "linear",
            "label_fraction": 0.1,
            "n_seeds": 2,
            "trainable_params": 1000,
            "total_params": 10000,
            "trainable_pct": 10.0,
            "miou_mean": 0.5,
            "miou_std": 0.02,
            "miou_n": 2,
            "train_seconds_mean": 3.0,
            "peak_memory_mb_mean": 10.0,
        },
        "config": {"epochs": 8, "batch_size": 8},
        "_few_label_fraction": 0.1,
        "_few_label_gain_vs_linear": {"linear@0.100": 0.0},
    }
    (tmp_results / "ablation_summary.json").write_text(json.dumps(summary))
    table = reporting._ablation_metric_table("miou")
    assert "linear" in table and "0.5000" in table
    assert "epochs" not in table
    params = table_ablation_params()
    assert "1,000" in params and "10.000 %" in params


def test_ablation_table_shows_mean_and_std_with_n(tmp_results):
    (tmp_results / "ablation_summary.json").write_text(
        json.dumps(
            {
                "full@0.010": {
                    "mode": "full",
                    "label_fraction": 0.01,
                    "n_seeds": 2,
                    "trainable_params": 22000000,
                    "total_params": 22000000,
                    "trainable_pct": 100.0,
                    "miou_mean": 0.1234,
                    "miou_std": 0.0123,
                    "miou_n": 2,
                }
            }
        )
    )
    table = reporting._ablation_metric_table("miou")
    assert "0.1234 ± 0.0123 (n=2)" in table
    assert "1%" in table, "fraction column header must be the label budget"


def test_ablation_modes_are_ordered_sensibly(tmp_results):
    cells = {}
    for m in ("full", "lora_r16", "linear", "lora_r4"):
        cells[f"{m}@1.000"] = {
            "mode": m,
            "label_fraction": 1.0,
            "n_seeds": 1,
            "miou_mean": 0.1,
            "miou_std": 0.0,
            "miou_n": 1,
        }
    (tmp_results / "ablation_summary.json").write_text(json.dumps(cells))
    table = reporting._ablation_metric_table("miou")
    pos = [table.index(m) for m in ("linear", "lora_r4", "lora_r16", "full")]
    assert pos == sorted(pos), "modes must read linear -> lora -> full"


def test_quality_table_reports_counts_and_coverage(tmp_results):
    (tmp_results / "quality.json").write_text(
        json.dumps(
            {
                "tests_collected": 90,
                "tests_passed": 88,
                "tests_deselected": 2,
                "coverage_pct": 71.5,
                "python": "3.11.14",
                "ruff": "clean",
            }
        )
    )
    table = table_quality()
    assert "88" in table and "71.5 %" in table


def test_latency_table_includes_the_onboard_verdict(tmp_results):
    (tmp_results / "latency.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "backend": "onnx",
                        "resolution": "224x224",
                        "batch_size": 1,
                        "mean_ms": 12.5,
                        "p50_ms": 12.0,
                        "p95_ms": 15.0,
                        "fps": 80.0,
                        "device": "cpu",
                    }
                ],
                "onboard": {
                    "requirement_hz": 10,
                    "best_backend": "onnx",
                    "best_resolution": "224x224",
                    "best_fps": 80.0,
                    "verdict": "feasible",
                },
            }
        )
    )
    table = reporting.table_latency()
    assert "onnx" in table and "10 Hz target" in table and "feasible" in table


def test_hard_case_retrain_table_shows_before_and_after(tmp_results):
    (tmp_results / "hardcase_retrain.json").write_text(
        json.dumps(
            {
                "before": {"miou": 0.40, "map_50": 0.20},
                "control": {"miou": 0.42, "map_50": 0.21},
                "after": {"miou": 0.45, "map_50": 0.25},
                "delta": {"miou": 0.05, "map_50": 0.05},
                "delta_vs_control": {"miou": 0.03, "map_50": 0.04},
                "n_hard_frames_added": 24,
                "repeat": 3,
                "note": "control isolates the hard-example effect",
                "per_cluster": [
                    {
                        "cluster": 0,
                        "tag": "small-object",
                        "miou_before": 0.1,
                        "miou_control": 0.12,
                        "miou_after": 0.2,
                        "delta_miou": 0.1,
                        "delta_miou_vs_control": 0.08,
                        "map_50_before": 0.0,
                        "map_50_control": 0.0,
                        "map_50_after": 0.05,
                        "delta_map_50": 0.05,
                        "delta_map_50_vs_control": 0.05,
                    }
                ],
            }
        )
    )
    table = reporting.table_hardcase_retrain()
    assert "0.4000" in table and "0.4200" in table and "0.4500" in table
    assert "small-object" in table
    assert "control" in table


def test_write_report_emits_every_table_file(tmp_path, monkeypatch):
    monkeypatch.setattr(reporting, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(reporting, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(reporting, "REPORTS_DIR", tmp_path / "reports")
    payload = write_report(
        tables_dir=tmp_path / "reports" / "tables", results_md=tmp_path / "reports" / "RESULTS.md"
    )
    for name in TABLE_ORDER:
        assert (tmp_path / "reports" / "tables" / f"{name}.md").exists()
    assert (tmp_path / "reports" / "RESULTS.md").exists()
    assert set(payload["tables"]) == set(TABLE_ORDER)
    assert "artifacts_present" in payload


def test_build_tables_never_raises_on_a_bare_repo(tmp_results):
    tables = build_tables()
    assert len(tables) == len(TABLE_ORDER)
    assert all(isinstance(v, str) and v.strip() for v in tables.values())


def test_number_formatting_helpers():
    assert reporting._num(None) == "n/a"
    assert reporting._num("0.5", 2) == "0.50"
    assert reporting._int("12345") == "12,345"
    assert reporting._int(None) == "n/a"
