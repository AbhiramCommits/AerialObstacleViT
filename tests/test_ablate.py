"""Step 4 tests: grid construction, stratified seeding, aggregation math, figures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerial_vit.ablation import (
    ABLATION_FIELDS,
    aggregate_rows,
    build_grid,
    cell_key,
    plot_miou_vs_labels,
    plot_params_vs_miou,
    read_csv,
    write_csv,
    write_summary,
)
from aerial_vit.config import load_config
from aerial_vit.data.semantic_drone import (
    compute_strata,
    load_manifest,
    read_boxes,
    stratified_subsample,
)

ABLATE = "configs/ablate.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(ABLATE)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(cfg=load_config(ABLATE))


# --------------------------------------------------------------------------- #
# grid
# --------------------------------------------------------------------------- #


def test_grid_is_the_full_cartesian_product(cfg):
    grid = build_grid(
        cfg,
        modes=["linear", "full"],
        fractions=[0.1, 1.0],
        seeds=[1, 2],
        multiseed_max_fraction=0.1,
    )
    # fraction 0.1 -> both seeds; fraction 1.0 -> one seed
    assert len(grid) == 2 * (2 + 1)
    keys = {(g["mode"], g["label_fraction"], g["seed"]) for g in grid}
    assert len(keys) == len(grid), "grid must not contain duplicate runs"


def test_grid_multiseed_rule_only_applies_to_small_cells(cfg):
    grid = build_grid(
        cfg,
        modes=["linear"],
        fractions=[0.01, 0.25, 1.0],
        seeds=[7, 8, 9],
        multiseed_max_fraction=0.25,
    )
    small = [g for g in grid if g["label_fraction"] <= 0.25]
    large = [g for g in grid if g["label_fraction"] > 0.25]
    assert len(small) == 2 * 3 and len(large) == 1


def test_grid_is_deterministic_and_names_are_unique(cfg):
    a = build_grid(cfg)
    b = build_grid(cfg)
    assert a == b
    names = [g["run_name"] for g in a]
    assert len(set(names)) == len(names)


def test_grid_covers_every_configured_mode_and_fraction(cfg):
    grid = build_grid(cfg)
    assert {g["mode"] for g in grid} == set(cfg.ablation.modes)
    assert {g["label_fraction"] for g in grid} == set(cfg.ablation.label_fractions)


def test_cell_key_is_stable():
    assert cell_key("linear", 0.05) == "linear@0.050"
    assert cell_key("lora_r16", 1.0) == "lora_r16@1.000"


# --------------------------------------------------------------------------- #
# stratified label sub-sampling
# --------------------------------------------------------------------------- #


def test_stratified_subsample_keeps_rare_classes_alive(manifest):
    root = Path(manifest["root"])
    names = manifest["splits"]["train"]
    rare = manifest["rare_classes"]
    strata = compute_strata(root, names, rare)

    rare_ids = [manifest["classes"].index(c) for c in rare]
    budget = stratified_subsample(names, 0.05, strata, seed=1337)
    assert budget, "a 5% budget must not be empty"
    present = set()
    for n in budget:
        present |= {int(b["class"]) for b in read_boxes(root, n)}
    for cls in rare_ids:
        assert cls in present, f"rare class {manifest['classes'][cls]} was dropped by the 5% budget"


def test_stratified_subsample_is_seeded_and_actually_subsamples(manifest):
    root = Path(manifest["root"])
    names = manifest["splits"]["train"]
    strata = compute_strata(root, names, manifest["rare_classes"])
    a = stratified_subsample(names, 0.1, strata, seed=5)
    b = stratified_subsample(names, 0.1, strata, seed=5)
    assert a == b
    assert len(a) < len(names)
    assert set(a).issubset(set(names))
    assert stratified_subsample(names, 0.1, strata, seed=6) != []


def test_stratified_subsample_full_fraction_returns_everything(manifest):
    root = Path(manifest["root"])
    names = manifest["splits"]["train"]
    strata = compute_strata(root, names, manifest["rare_classes"])
    assert set(stratified_subsample(names, 1.0, strata, seed=0)) == set(names)


def test_compute_strata_accepts_names_or_indices(manifest):
    root = Path(manifest["root"])
    names = manifest["splits"]["train"][:10]
    by_name = compute_strata(root, names, ["person"])
    by_idx = compute_strata(root, names, [6])
    assert by_name == by_idx


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #


def _row(
    mode, frac, seed, miou, lz=0.5, m50=0.1, trainable=1000, total=100000, secs=10.0, mem=100.0
):
    return {
        "mode": mode,
        "label_fraction": frac,
        "seed": seed,
        "miou": miou,
        "landing_zone_iou": lz,
        "map_50": m50,
        "map_50_95": m50 / 2,
        "rare_map_50": 0.0,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100.0 * trainable / total,
        "train_seconds": secs,
        "peak_memory_mb": mem,
        "n_train_images": 10,
        "device": "cpu",
        "run_name": f"{mode}_{frac}_{seed}",
    }


def test_aggregate_mean_and_std_are_correct():
    rows = [
        _row("linear", 0.05, 1, 0.10),
        _row("linear", 0.05, 2, 0.30),
        _row("linear", 1.0, 1, 0.50),
    ]
    s = aggregate_rows(rows)
    cell = s[cell_key("linear", 0.05)]
    assert cell["miou_mean"] == pytest.approx(0.20, abs=1e-9)
    assert cell["miou_std"] == pytest.approx(0.10, abs=1e-9)
    assert cell["n_seeds"] == 2
    single = s[cell_key("linear", 1.0)]
    assert single["miou_mean"] == pytest.approx(0.50)
    assert single["miou_std"] == 0.0


def test_aggregate_computes_few_label_gain_vs_linear():
    rows = [
        _row("linear", 0.01, 1, 0.10, trainable=1000),
        _row("lora_r16", 0.01, 1, 0.25, trainable=20000),
        _row("full", 0.01, 1, 0.30, trainable=100000),
    ]
    s = aggregate_rows(rows)
    gains = s["_few_label_gain_vs_linear"]
    assert gains[cell_key("lora_r16", 0.01)] == pytest.approx(0.15, abs=1e-9)
    assert gains[cell_key("full", 0.01)] == pytest.approx(0.20, abs=1e-9)
    assert gains[cell_key("linear", 0.01)] == pytest.approx(0.0)
    assert s["_few_label_fraction"] == 0.01


def test_aggregate_ignores_missing_metrics():
    rows = [
        {"mode": "linear", "label_fraction": 0.1, "seed": 1, "miou": 0.4, "map_50": None},
        {"mode": "linear", "label_fraction": 0.1, "seed": 2, "miou": 0.6, "map_50": ""},
    ]
    s = aggregate_rows(rows)[cell_key("linear", 0.1)]
    assert s["miou_mean"] == pytest.approx(0.5)
    assert s["map_50_mean"] is None
    assert s["map_50_n"] == 0


def test_aggregate_is_empty_safe():
    assert aggregate_rows([]) == {}


# --------------------------------------------------------------------------- #
# IO + figures
# --------------------------------------------------------------------------- #


def test_csv_roundtrip_preserves_every_field(tmp_path):
    rows = [_row("linear", 0.05, 1, 0.2), _row("full", 1.0, 2, 0.5, trainable=20000)]
    p = write_csv(rows, tmp_path / "ablation.csv")
    back = read_csv(p)
    assert len(back) == 2
    assert list(back[0].keys()) == ABLATION_FIELDS
    assert float(back[1]["miou"]) == pytest.approx(0.5)


def test_write_summary_stores_extra_metadata(tmp_path):
    p = write_summary(
        {"a@0.1": {"miou_mean": 0.5}}, tmp_path / "s.json", extra={"n_runs": 3, "device": "cpu"}
    )
    data = json.loads(p.read_text())
    assert data["n_runs"] == 3 and data["device"] == "cpu" and "a@0.1" in data


def test_figures_render_from_a_small_summary(tmp_path):
    rows = []
    for mode in ("linear", "lora_r16", "full"):
        for frac in (0.05, 0.25, 1.0):
            for seed in (1, 2):
                rows.append(
                    _row(
                        mode,
                        frac,
                        seed,
                        0.2 + 0.1 * frac,
                        trainable={"linear": 1000, "lora_r16": 5000, "full": 100000}[mode],
                    )
                )
    summary = aggregate_rows(rows)
    f1 = plot_miou_vs_labels(summary, tmp_path / "a.png")
    f2 = plot_params_vs_miou(summary, tmp_path / "b.png")
    for f in (f1, f2):
        assert f.exists() and f.stat().st_size > 2000, f"{f} looks empty"


def test_figure_handles_single_cell_summary(tmp_path):
    summary = aggregate_rows([_row("linear", 1.0, 1, 0.3)])
    p = plot_miou_vs_labels(summary, tmp_path / "one.png")
    assert p.exists() and p.stat().st_size > 1000


def test_figure_handles_zero_metric(tmp_path):
    s = aggregate_rows([_row("linear", 0.1, 1, 0.0), _row("full", 0.1, 1, 0.0)])
    for plot in (plot_miou_vs_labels, plot_params_vs_miou):
        p = plot(s, tmp_path / f"{plot.__name__}.png")
        assert p.exists()
