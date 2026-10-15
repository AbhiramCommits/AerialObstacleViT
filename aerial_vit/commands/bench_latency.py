"""``bench-latency`` -- eager vs TorchScript vs ONNX Runtime, with an onboard verdict.

    aerialvit bench-latency --config configs/full.yaml --run full_linear

Writes ``artifacts/results/latency.json`` and ``reports/figures/latency.png``.
Every row records the actual device, warmup count and iteration count used, so
the numbers can be re-derived rather than trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from aerial_vit.benchmark import (
    BenchConfig,
    benchmark_callable,
    onboard_verdict,
    plot_latency,
    summarise_latencies,
)
from aerial_vit.checkpoint import load_checkpoint_model
from aerial_vit.commands._common import ConfigOpt, SetOpt, load_cli_config
from aerial_vit.config import ARTIFACTS_DIR, FIGURES_DIR, RESULTS_DIR, Config
from aerial_vit.export import ExportWrapper, export_onnx, export_torchscript
from aerial_vit.utils.logging import get_logger
from aerial_vit.utils.seed import device_name, resolve_device, seed_everything

app = typer.Typer(add_completion=False, help="Benchmark eager / TorchScript / ONNX latency.")
log = get_logger("aerialvit.bench")


def _onnx_runner(path: Path, providers: list[str]):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=providers)

    def _run(x: torch.Tensor):
        return sess.run(None, {"images": x.cpu().numpy()})

    return _run, sess


def run_benchmark(
    cfg: Config,
    checkpoint: str | None = None,
    resolutions: tuple[int, ...] = (224, 320),
    batches: tuple[int, ...] = (1, 8),
    iterations: int = 50,
    warmup: int = 10,
    requirement_hz: float = 10.0,
    onnx_provider: str = "CPUExecutionProvider",
) -> dict:
    import copy as _copy

    seed_everything(cfg.run.seed, cfg.run.deterministic)
    accel = resolve_device(cfg.run.device)
    ckpt = checkpoint or str(Path(cfg.run.out_dir) / "runs" / cfg.run.name / "best.pt")
    model, info, ckpt_data = load_checkpoint_model(ckpt, cfg=cfg, device=torch.device("cpu"))
    model.eval()
    wrapper_cpu = ExportWrapper(model).eval()
    # A separate accelerator copy: the CPU wrapper must stay on CPU so the
    # ONNX/TorchScript parity checks and the CPU benchmarks are not perturbed.
    wrapper_accel = (
        ExportWrapper(_copy.deepcopy(model)).to(accel).eval()
        if accel.type != "cpu"
        else wrapper_cpu
    )
    log.info(
        "benchmarking %s (mode=%s); accelerator=%s", ckpt, ckpt_data.get("mode"), device_name(accel)
    )

    tmp = ARTIFACTS_DIR / "models"
    tmp.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    export_meta: dict[str, dict] = {}

    for res in resolutions:
        example = torch.randn(2, 3, res, res)
        ts_path = export_torchscript(model, tmp / f"{cfg.run.name}_{res}.torchscript", example)
        onnx_path = tmp / f"{cfg.run.name}_{res}.onnx"
        export_meta[str(res)] = export_onnx(model, onnx_path, example)
        ts_cpu = torch.jit.load(str(ts_path)).eval()
        ts_accel = ts_cpu
        if accel.type != "cpu":
            try:
                ts_accel = _copy.deepcopy(ts_cpu).to(accel).eval()
            except Exception as exc:  # noqa: BLE001 - backend may not support the graph
                failures.append(
                    {
                        "backend": "torchscript",
                        "device": device_name(accel),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                ts_accel = None

        for bs in batches:
            bc = BenchConfig(warmup=warmup, iterations=iterations, resolution=res, batch_size=bs)
            for backend, fn, dev, dname in (
                ("pytorch-eager", lambda x, m=wrapper_accel: m(x), accel, device_name(accel)),
                ("torchscript", lambda x, m=ts_accel: m(x), accel, device_name(accel)),
                ("pytorch-eager", lambda x, m=wrapper_cpu: m(x), torch.device("cpu"), "cpu"),
                ("torchscript", lambda x, m=ts_cpu: m(x), torch.device("cpu"), "cpu"),
            ):
                if fn is None or (
                    backend == "torchscript" and dev.type != "cpu" and ts_accel is None
                ):
                    continue
                try:
                    r = benchmark_callable(fn, bc, dev)
                except Exception as exc:  # noqa: BLE001
                    failures.append(
                        {
                            "backend": backend,
                            "device": dname,
                            "resolution": f"{res}x{res}",
                            "batch_size": bs,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    log.warning("%s on %s failed: %s", backend, dname, exc)
                    continue
                rows.append({**r, "backend": backend, "device": dname})
            # ONNX Runtime (CPU)
            try:
                providers = list(dict.fromkeys([onnx_provider, "CPUExecutionProvider"]))
                runner, _ = _onnx_runner(onnx_path, providers)
                r = benchmark_callable(runner, bc, "cpu")
                rows.append({**r, "backend": "onnxruntime", "device": f"cpu ({onnx_provider})"})
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "backend": "onnxruntime",
                        "device": onnx_provider,
                        "resolution": f"{res}x{res}",
                        "batch_size": bs,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                log.warning("onnxruntime failed: %s", exc)
            log.info("res=%d batch=%d done", res, bs)

    onboard = onboard_verdict(rows, requirement_hz)
    report = {
        "run": cfg.run.name,
        "checkpoint": ckpt,
        "mode": ckpt_data.get("mode"),
        "accelerator": device_name(accel),
        "onnx_provider": onnx_provider,
        "onnxruntime_available_providers": __import__("onnxruntime").get_available_providers(),
        "iterations": iterations,
        "warmup": warmup,
        "requirement_hz": requirement_hz,
        "rows": rows,
        "failures": failures,
        "summary": summarise_latencies(rows),
        "onboard": onboard,
        "exports": export_meta,
        "parameter_summary": info,
        "device_note": (
            "onnxruntime rows are CPU-only; eager and torchscript are reported on BOTH the "
            "accelerator and CPU so the comparison is not confounded by the device"
        ),
    }
    out = RESULTS_DIR / "latency.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    plot_latency(rows, FIGURES_DIR / "latency.png", requirement_hz)
    log.info("wrote %s and latency.png", out)
    return report


@app.command()
def main(
    config: str = ConfigOpt,
    checkpoint: str = typer.Option(None, "--checkpoint"),
    iterations: int = typer.Option(50, "--iterations", help="Timed iterations per configuration."),
    warmup: int = typer.Option(10, "--warmup"),
    resolutions: str = typer.Option("224,320", "--resolutions"),
    batches: str = typer.Option("1,8", "--batches"),
    requirement_hz: float = typer.Option(10.0, "--requirement-hz", help="Drone frame-rate target."),
    run_name: str = typer.Option(None, "--run-name"),
    set_: list[str] = SetOpt,
) -> None:
    """Benchmark all backends and write latency.json."""
    cfg = load_cli_config(config, set_)
    if run_name:
        cfg.run.name = run_name
    r = run_benchmark(
        cfg,
        checkpoint,
        tuple(int(x) for x in resolutions.split(",")),
        tuple(int(x) for x in batches.split(",")),
        iterations,
        warmup,
        requirement_hz,
    )
    for row in r["rows"]:
        if row["backend"] in {"onnxruntime"} or row["device"] == r["accelerator"]:
            typer.echo(
                f"{row['backend']:<15} {row['resolution']:<9} b{row['batch_size']:<2} "
                f"p50={row['p50_ms']:>9.2f} ms  p95={row['p95_ms']:>9.2f} ms  {row['fps']:>8.2f} FPS  [{row['device']}]"
            )
    ob = r["onboard"]
    typer.echo(
        f"\nonboard: best batch-1 = {ob['best_fps']:.2f} FPS ({ob['best_backend']}, "
        f"{ob['best_resolution']}) vs {ob['requirement_hz']:g} Hz -> {ob['verdict']}"
    )
    for b, v in ob.get("per_backend_batch1", {}).items():
        typer.echo(
            f"  {b:<16} batch-1 {v['fps']:>8.2f} FPS  (p50 {v['p50_ms']:>8.2f} ms, {v['device']})"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
