"""Report generation: every number in the README comes from here.

``make_report`` reads *only* the JSON/CSV artifacts written by earlier pipeline
stages and renders markdown tables into ``reports/tables/*.md`` plus an
assembled ``reports/RESULTS.md``.  The README embeds the exact same snippet
files, which is what makes "README numbers match RESULTS.md byte-for-byte"
mechanically true rather than aspirational.

Nothing here invents or rounds a value it did not read from an artifact: any
missing artifact produces an explicit "not available" row instead of a guess.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from aerial_vit.ablation import read_csv
from aerial_vit.config import ARTIFACTS_DIR, REPORTS_DIR, RESULTS_DIR

__all__ = ["build_tables", "write_report", "TABLES_DIR", "TABLE_ORDER"]

TABLES_DIR = REPORTS_DIR / "tables"

TABLE_ORDER = [
    "provenance",
    "foundation_vs_tasktrained",
    "ablation_miou",
    "ablation_landing_zone",
    "ablation_map50",
    "ablation_params",
    "hard_cases",
    "hardcase_retrain",
    "latency",
    "quality",
]

TITLES = {
    "provenance": "Provenance",
    "foundation_vs_tasktrained": "Foundation features vs open-vocabulary VLMs",
    "ablation_miou": "Label-scarcity ablation - mIoU",
    "ablation_landing_zone": "Label-scarcity ablation - landing-zone IoU",
    "ablation_map50": "Label-scarcity ablation - detection mAP@0.5",
    "ablation_params": "Label-scarcity ablation - trainable parameters and cost",
    "hard_cases": "Hard-case clusters",
    "hardcase_retrain": "Targeted retrain on mined hard cases",
    "latency": "Inference latency: eager vs TorchScript vs ONNX Runtime",
    "quality": "Test suite and coverage",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _f(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _num(v: Any, digits: int = 4) -> str:
    x = _f(v)
    return "n/a" if x is None else f"{x:.{digits}f}"


def _int(v: Any) -> str:
    x = _f(v)
    return "n/a" if x is None else f"{int(round(x)):,}"


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_not available yet - run the corresponding pipeline stage._"
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _sort_modes(modes) -> list[str]:
    order = {"linear": 0, "lora_r4": 1, "lora_r8": 2, "lora_r16": 3, "lora_r32": 4, "full": 5}
    return sorted(set(modes), key=lambda m: (order.get(m, 99), m))


# --------------------------------------------------------------------------- #
# individual tables
# --------------------------------------------------------------------------- #


def table_provenance(cfg=None) -> str:
    manifest = _load_json(ARTIFACTS_DIR / "data_manifest.json") or {}
    rows = [
        ["data source", manifest.get("source", "n/a")],
        ["data detail", manifest.get("source_detail", "n/a")],
        [
            "train / val / test frames",
            f"{manifest.get('n_train', 'n/a')} / {manifest.get('n_val', 'n/a')} / {manifest.get('n_test', 'n/a')}",
        ],
        ["classes", ", ".join(manifest.get("classes", [])) or "n/a"],
        ["rare classes (measured)", ", ".join(manifest.get("rare_classes", [])) or "(none)"],
        ["image size", str(manifest.get("image_size", "n/a"))],
        [
            "per-class pixel fraction",
            json.dumps(manifest.get("per_class_pixel_fraction", {}), sort_keys=True),
        ],
    ]
    return _md_table(["field", "value"], rows)


def table_foundation_vs_tasktrained() -> str:
    rows = read_csv(RESULTS_DIR / "foundation_vs_tasktrained.csv")
    if not rows:
        return _md_table([], [])
    out = []
    for r in rows:
        out.append(
            [
                r["method"],
                _int(r.get("trainable_params")),
                _int(r.get("total_params")),
                _num(r.get("trainable_pct"), 3) + " %",
                _num(r.get("miou")),
                _num(r.get("landing_zone_iou")),
                _num(r.get("map_50")),
                _num(r.get("rare_map_50")),
                _num(r.get("ms_per_image"), 1),
                r.get("device", "n/a"),
            ]
        )
    return _md_table(
        [
            "method",
            "trainable params",
            "total params",
            "trainable %",
            "mIoU",
            "landing-zone IoU",
            "mAP@0.5",
            "rare-class mAP@0.5",
            "ms/img",
            "device",
        ],
        out,
    )


def _ablation_cells() -> dict[str, dict[str, Any]]:
    """Only the per-cell entries: an ablation_summary.json also stores a `config` block."""
    summary = _load_json(RESULTS_DIR / "ablation_summary.json")
    if not summary:
        return {}
    return {
        k: v
        for k, v in summary.items()
        if not k.startswith("_") and isinstance(v, dict) and "mode" in v and "label_fraction" in v
    }


def _ablation_metric_table(metric: str) -> str:
    cells = _ablation_cells()
    if not cells:
        return _md_table([], [])
    fractions = sorted({v["label_fraction"] for v in cells.values()})
    modes = _sort_modes(v["mode"] for v in cells.values())
    headers = ["mode", "trainable params", "trainable %"] + [f"{f * 100:g}%" for f in fractions]
    rows = []
    for mode in modes:
        ref = next(v for v in cells.values() if v["mode"] == mode)
        row = [mode, _int(ref.get("trainable_params")), _num(ref.get("trainable_pct"), 2) + " %"]
        for f in fractions:
            cell = next(
                (
                    v
                    for v in cells.values()
                    if v["mode"] == mode and abs(v["label_fraction"] - f) < 1e-9
                ),
                None,
            )
            if cell is None:
                row.append("n/a")
            else:
                mean, std = cell.get(f"{metric}_mean"), cell.get(f"{metric}_std")
                n = cell.get(f"{metric}_n", 0)
                row.append(f"{_num(mean)} ± {_num(std)} (n={n})")
        rows.append(row)
    return _md_table(headers, rows)


def table_ablation_params() -> str:
    cells = _ablation_cells()
    if not cells:
        return _md_table([], [])
    modes = _sort_modes(v["mode"] for v in cells.values())
    rows = []
    for mode in modes:
        pts = [v for v in cells.values() if v["mode"] == mode]
        if not pts:
            continue
        ref = pts[0]
        rows.append(
            [
                mode,
                _int(ref.get("trainable_params")),
                _int(ref.get("total_params")),
                _num(ref.get("trainable_pct"), 3) + " %",
                len(pts),
                _num(sum(p.get("n_seeds", 0) for p in pts) / max(1, len(pts)), 1),
                _num(sum(p.get("train_seconds_mean") or 0 for p in pts) / max(1, len(pts)), 1),
                _num(max((p.get("peak_memory_mb_mean") or 0) for p in pts), 1),
            ]
        )
    return _md_table(
        [
            "mode",
            "trainable params",
            "total params",
            "trainable %",
            "cells",
            "seeds/cell (avg)",
            "train s/cell (avg)",
            "peak MB (max)",
        ],
        rows,
    )


def table_hard_cases() -> str:
    per_cluster = _load_csv(RESULTS_DIR / "per_cluster_metrics.csv")
    if not per_cluster:
        return _md_table([], [])
    headers = [
        "cluster",
        "auto tag",
        "n frames",
        "mIoU",
        "split mIoU",
        "Δ mIoU",
        "mAP@0.5",
        "split mAP@0.5",
        "Δ mAP@0.5",
    ]
    rows = []
    for r in per_cluster:
        rows.append(
            [
                r.get("cluster", "?"),
                r.get("tag", r.get("auto_tag", "?")),
                _int(r.get("n_frames")),
                _num(r.get("miou")),
                _num(r.get("split_miou")),
                _num(r.get("delta_miou")),
                _num(r.get("map_50")),
                _num(r.get("split_map_50")),
                _num(r.get("delta_map_50")),
            ]
        )
    return _md_table(headers, rows)


def table_hardcase_retrain() -> str:
    data = _load_json(RESULTS_DIR / "hardcase_retrain.json")
    if not data:
        return _md_table([], [])
    before = data.get("before", {})
    control = data.get("control", {})
    after = data.get("after", {})
    rows = [
        [
            "**overall**",
            _num(before.get("miou")),
            _num(control.get("miou")),
            _num(after.get("miou")),
            _num(data.get("delta", {}).get("miou")),
            _num(data.get("delta_vs_control", {}).get("miou")),
            _num(data.get("delta", {}).get("map_50")),
            _num(data.get("delta_vs_control", {}).get("map_50")),
        ]
    ]
    for c in data.get("per_cluster", []):
        rows.append(
            [
                f"cluster {c.get('cluster')} ({c.get('tag', '')})",
                _num(c.get("miou_before")),
                _num(c.get("miou_control")),
                _num(c.get("miou_after")),
                _num(c.get("delta_miou")),
                _num(c.get("delta_miou_vs_control")),
                _num(c.get("delta_map_50")),
                _num(c.get("delta_map_50_vs_control")),
            ]
        )
    table = _md_table(
        [
            "scope",
            "mIoU before",
            "mIoU control",
            "mIoU after",
            "Δ mIoU vs before",
            "Δ mIoU vs control",
            "Δ mAP@0.5 vs before",
            "Δ mAP@0.5 vs control",
        ],
        rows,
    )
    if data.get("note"):
        table += (
            "\n\n_control = identical warm-start and epochs, trained **without** the mined hard "
            f"frames; `vs control` therefore isolates the effect of the hard examples themselves. "
            f"{data.get('n_hard_frames_added')} hard frames were added, up-weighted x{data.get('repeat')}._\n"
        )
    return table


def table_latency() -> str:
    data = _load_json(RESULTS_DIR / "latency.json")
    if not data:
        return _md_table([], [])
    rows = []
    for r in data.get("rows", []):
        rows.append(
            [
                r.get("backend", "?"),
                r.get("resolution", "?"),
                _int(r.get("batch_size")),
                _num(r.get("mean_ms"), 2),
                _num(r.get("p50_ms"), 2),
                _num(r.get("p95_ms"), 2),
                _num(r.get("fps"), 2),
                r.get("device", "n/a"),
            ]
        )
    verdict = data.get("onboard", {})
    table = _md_table(
        ["backend", "resolution", "batch", "mean ms", "p50 ms", "p95 ms", "FPS", "device"],
        rows,
    )
    if verdict:
        table += (
            f"\n\n**Onboard feasibility at {verdict.get('requirement_hz', 'n/a')} Hz target:** "
            f"best backend `{verdict.get('best_backend', 'n/a')}` at {verdict.get('best_resolution', 'n/a')} "
            f"reaches {_num(verdict.get('best_fps'), 2)} FPS -> "
            f"**{verdict.get('verdict', 'n/a')}**.\n"
        )
    return table


def table_quality() -> str:
    data = _load_json(RESULTS_DIR / "quality.json")
    if not data:
        return _md_table([], [])
    rows = [
        ["pytest tests collected", _int(data.get("tests_collected"))],
        ["pytest tests passed", _int(data.get("tests_passed"))],
        ["pytest tests deselected (slow)", _int(data.get("tests_deselected"))],
        ["coverage (aerial_vit)", _num(data.get("coverage_pct"), 1) + " %"],
        ["ruff", data.get("ruff", "n/a")],
        ["python", data.get("python", "n/a")],
    ]
    return _md_table(["check", "result"], rows)


_BUILDERS = {
    "provenance": table_provenance,
    "foundation_vs_tasktrained": table_foundation_vs_tasktrained,
    "ablation_miou": lambda: _ablation_metric_table("miou"),
    "ablation_landing_zone": lambda: _ablation_metric_table("landing_zone_iou"),
    "ablation_map50": lambda: _ablation_metric_table("map_50"),
    "ablation_params": table_ablation_params,
    "hard_cases": table_hard_cases,
    "hardcase_retrain": table_hardcase_retrain,
    "latency": table_latency,
    "quality": table_quality,
}


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #


def build_tables() -> dict[str, str]:
    return {name: _BUILDERS[name]() for name in TABLE_ORDER}


def _headline(tables: dict[str, str]) -> str:
    """A short, artifact-derived summary block (no invented numbers)."""
    fvt = read_csv(RESULTS_DIR / "foundation_vs_tasktrained.csv")
    summary = _load_json(RESULTS_DIR / "ablation_summary.json") or {}
    latency = _load_json(RESULTS_DIR / "latency.json") or {}
    quality = _load_json(RESULTS_DIR / "quality.json") or {}
    lines: list[str] = []

    task = next((r for r in fvt if r.get("zero_shot") in ("False", False)), None)
    zs = [r for r in fvt if r.get("zero_shot") in ("True", True)]
    if task and zs:
        lines.append(
            f"- Task-trained DINOv2 + heads reaches **mIoU {_num(task['miou'])}** and "
            f"**mAP@0.5 {_num(task['map_50'])}**, vs "
            + " / ".join(
                f"{r['method']} mIoU {_num(r['miou'])}, mAP@0.5 {_num(r['map_50'])}" for r in zs
            )
            + "."
        )
    gains = summary.get("_few_label_gain_vs_linear")
    if gains:
        frac = summary.get("_few_label_fraction")
        best = max(gains.items(), key=lambda kv: kv[1] if kv[1] is not None else -1)
        lines.append(
            f"- At a **{frac * 100:g}% label budget**, `{best[0].split('@')[0]}` improves mIoU by "
            f"**{_num(best[1])}** over the linear probe."
        )
    onboard = latency.get("onboard")
    if onboard:
        lines.append(
            f"- Best ONNX/TorchScript throughput **{_num(onboard.get('best_fps'), 2)} FPS** "
            f"(`{onboard.get('best_backend')}`, {onboard.get('best_resolution')}) -> "
            f"{onboard.get('verdict')} against a {onboard.get('requirement_hz')} Hz requirement."
        )
    if quality:
        lines.append(
            f"- Test suite: **{_int(quality.get('tests_passed'))} passed**, "
            f"**{_num(quality.get('coverage_pct'), 1)}% coverage** of `aerial_vit`."
        )
    return "\n".join(lines) if lines else "_no headline artifacts yet._"


def write_report(
    tables_dir: str | Path = TABLES_DIR,
    results_md: str | Path = REPORTS_DIR / "RESULTS.md",
    table_order: list[str] | None = None,
) -> dict[str, Any]:
    tables = build_tables()
    order = table_order or TABLE_ORDER
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    for name in order:
        (tables_dir / f"{name}.md").write_text(tables[name] + "\n")

    body = ["# Results", "", "_Generated by `scripts/make_report.py` from `artifacts/` only._", ""]
    body += ["## Headline", "", _headline(tables), ""]
    for name in order:
        if name == "provenance":
            body += [f"## {TITLES[name]}", "", tables[name], ""]
            continue
        body += [f"## {TITLES[name]}", "", tables[name], ""]
        fig = {
            "ablation_miou": "reports/figures/ablation_miou_vs_labels.png",
            "ablation_params": "reports/figures/params_vs_miou.png",
            "hard_cases": "reports/figures/hard_case_clusters.png",
            "latency": "reports/figures/latency.png",
        }.get(name)
        if fig:
            body += [f"![{name}]({fig})", ""]

    out = Path(results_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body).rstrip() + "\n")

    manifest = _load_json(ARTIFACTS_DIR / "data_manifest.json") or {}
    try:
        from aerial_vit.utils.provenance import write_provenance

        prov = write_provenance(ARTIFACTS_DIR / "results" / "provenance.json")
    except Exception:  # noqa: BLE001
        prov = {}
    payload = {
        "tables": tables,
        "table_order": order,
        "headline": _headline(tables),
        "data_manifest": manifest,
        "provenance": prov,
        "artifacts_present": {
            "foundation_vs_tasktrained.csv": (
                RESULTS_DIR / "foundation_vs_tasktrained.csv"
            ).exists(),
            "ablation.csv": (RESULTS_DIR / "ablation.csv").exists(),
            "ablation_summary.json": (RESULTS_DIR / "ablation_summary.json").exists(),
            "hard_cases.json": (RESULTS_DIR / "hard_cases.json").exists(),
            "per_cluster_metrics.csv": (RESULTS_DIR / "per_cluster_metrics.csv").exists(),
            "hardcase_retrain.json": (RESULTS_DIR / "hardcase_retrain.json").exists(),
            "latency.json": (RESULTS_DIR / "latency.json").exists(),
            "quality.json": (RESULTS_DIR / "quality.json").exists(),
        },
    }
    (REPORTS_DIR / "results_tables.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return payload
