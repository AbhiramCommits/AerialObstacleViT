"""``export`` -- TorchScript + ONNX export with numerical parity checks.

    aerialvit export --config configs/full.yaml --run full_linear

Writes ``artifacts/models/<run>.torchscript`` (gitignored), ``.onnx``
(gitignored) and ``artifacts/results/export_<run>.json`` (kept), which records
the parity numbers and asserts them against a tolerance.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from aerial_vit.checkpoint import load_checkpoint_model
from aerial_vit.commands._common import ConfigOpt, SetOpt, load_cli_config
from aerial_vit.config import ARTIFACTS_DIR, RESULTS_DIR, Config
from aerial_vit.export import (
    export_onnx,
    export_torchscript,
    verify_onnx_parity,
    verify_torchscript,
)
from aerial_vit.utils.logging import get_logger
from aerial_vit.utils.seed import device_name, resolve_device, seed_everything

app = typer.Typer(add_completion=False, help="Export the best checkpoint to TorchScript and ONNX.")
log = get_logger("aerialvit.export.cmd")


def run_export(
    cfg: Config,
    checkpoint: str | None = None,
    onnx_tolerance: float = 1e-3,
    ts_tolerance: float = 1e-4,
) -> dict:
    seed_everything(cfg.run.seed, cfg.run.deterministic)
    device = resolve_device(cfg.run.device)
    ckpt = checkpoint or str(Path(cfg.run.out_dir) / "runs" / cfg.run.name / "best.pt")
    model, info, ckpt_data = load_checkpoint_model(ckpt, cfg=cfg, device=torch.device("cpu"))
    model.eval()
    log.info("exporting %s (mode=%s)", ckpt, ckpt_data.get("mode"))

    size = int(cfg.data.image_size)
    example = torch.randn(2, 3, size, size)
    models_dir = ARTIFACTS_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    ts_path = models_dir / f"{cfg.run.name}.torchscript"
    onnx_path = models_dir / f"{cfg.run.name}.onnx"

    ts_path = export_torchscript(model, ts_path, example)
    ts_check = verify_torchscript(model, ts_path, example)
    log.info(
        "torchscript parity: max|Δ|=%.3e miou=%.6f passed=%s",
        ts_check["max_abs_diff"],
        ts_check["miou"],
        ts_check["passed"],
    )

    onnx_meta = export_onnx(model, onnx_path, example)
    onnx_check = verify_onnx_parity(model, onnx_path, example, tolerance=onnx_tolerance)
    log.info(
        "onnx parity: max|Δ|=%.3e miou=%.6f passed=%s (exporter=%s)",
        onnx_check["max_abs_diff"],
        onnx_check["miou"],
        onnx_check["passed"],
        onnx_meta["exporter"],
    )

    # batch-8 / dynamic-batch smoke: the same graph must serve another batch size
    dynamic_ok = True
    dynamic_err = ""
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out8 = sess.run(None, {"images": torch.randn(8, 3, size, size).numpy()})
        dynamic_ok = out8[0].shape == (8, cfg.data.num_classes, size, size)
    except Exception as exc:  # noqa: BLE001
        dynamic_ok = False
        dynamic_err = f"{type(exc).__name__}: {exc}"

    report = {
        "run": cfg.run.name,
        "checkpoint": ckpt,
        "mode": ckpt_data.get("mode"),
        "image_size": size,
        "device_used_for_parity": "cpu",
        "source_device": device_name(device),
        "torchscript": {
            **ts_check,
            "tolerance": ts_tolerance,
            "passed": ts_check["max_abs_diff"] <= ts_tolerance,
            "path": str(ts_path),
        },
        "onnx": {
            **onnx_meta,
            **onnx_check,
            "tolerance": onnx_tolerance,
            "passed": onnx_check["max_abs_diff"] <= onnx_tolerance,
        },
        "dynamic_batch_ok": dynamic_ok,
        "dynamic_batch_error": dynamic_err,
        "parameter_summary": info,
    }
    out = RESULTS_DIR / f"export_{cfg.run.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    log.info("wrote %s", out)
    return report


@app.command()
def main(
    config: str = ConfigOpt,
    checkpoint: str = typer.Option(None, "--checkpoint", help="Checkpoint to export."),
    onnx_tolerance: float = typer.Option(1e-3, "--onnx-tolerance"),
    ts_tolerance: float = typer.Option(1e-4, "--ts-tolerance"),
    run_name: str = typer.Option(None, "--run-name"),
    set_: list[str] = SetOpt,
) -> None:
    """Export and verify TorchScript + ONNX artefacts."""
    cfg = load_cli_config(config, set_)
    if run_name:
        cfg.run.name = run_name
    r = run_export(cfg, checkpoint, onnx_tolerance, ts_tolerance)
    typer.echo(
        f"torchscript : max|Δ|={r['torchscript']['max_abs_diff']:.3e}  mIoU={r['torchscript']['miou']}  passed={r['torchscript']['passed']}"
    )
    typer.echo(
        f"onnx        : max|Δ|={r['onnx']['max_abs_diff']:.3e}  mIoU={r['onnx']['miou']}  passed={r['onnx']['passed']}"
    )
    typer.echo(
        f"onnx exporter: {r['onnx']['exporter']}  opset={r['onnx']['opset']}  "
        f"size={r['onnx']['total_bytes'] / 1e6:.1f} MB (graph {r['onnx']['size_bytes'] / 1e6:.2f} MB "
        f"+ weights {r['onnx']['external_data_bytes'] / 1e6:.1f} MB)"
    )
    typer.echo(f"dynamic batch: {r['dynamic_batch_ok']} {r['dynamic_batch_error']}")
    typer.echo(f"report      : artifacts/results/export_{cfg.run.name}.json")


if __name__ == "__main__":  # pragma: no cover
    app()
