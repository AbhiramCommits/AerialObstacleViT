"""``train`` -- fine-tune lightweight heads (and optionally LoRA / the full net).

aerialvit train --config configs/smoke.yaml --mode linear
aerialvit train --config configs/ablate.yaml --mode lora_r16 --label-fraction 0.05
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from aerial_vit.commands._common import ConfigOpt, SetOpt, load_cli_config
from aerial_vit.config import Config
from aerial_vit.data.semantic_drone import build_datasets, default_manifest_path
from aerial_vit.models import build_model
from aerial_vit.train_engine import (
    compute_class_weights,
    make_loader,
    train_model,
)
from aerial_vit.utils.logging import get_logger
from aerial_vit.utils.seed import device_name, resolve_device, seed_everything

app = typer.Typer(add_completion=False, help="Train the DINOv2 + heads perception model.")
log = get_logger("aerialvit.train")


def run_train(
    cfg: Config,
    mode: str,
    label_fraction: float | None = None,
    seed: int | None = None,
    run_name: str | None = None,
    epochs: int | None = None,
    max_steps: int | None = None,
    pretrained: bool = True,
    run_dir: Path | None = None,
) -> dict:
    seed = int(seed if seed is not None else cfg.run.seed)
    seed_everything(seed, cfg.run.deterministic)
    device = resolve_device(cfg.run.device)
    if run_name:
        cfg.run.name = run_name
    if epochs is not None:
        cfg.train.epochs = int(epochs)
    if label_fraction is not None:
        cfg.train.label_fraction = float(label_fraction)
    max_steps = int(max_steps if max_steps is not None else cfg.train.max_steps)

    train_ds, val_ds, test_ds = build_datasets(
        cfg, label_fraction=cfg.train.label_fraction, seed=seed
    )
    log.info(
        "device=%s  train=%d val=%d test=%d  mode=%s  fraction=%.3f",
        device_name(device),
        len(train_ds),
        len(val_ds),
        len(test_ds),
        mode,
        cfg.train.label_fraction,
    )
    if device.type == "cpu":
        log.warning("no GPU available -> training on CPU with the configured (reduced) budget")

    train_loader = make_loader(train_ds, cfg.train.batch_size, True, cfg.data.num_workers, seed)
    val_loader = make_loader(val_ds, cfg.eval.batch_size, False, cfg.data.num_workers, seed)

    model, info = build_model(cfg, mode=mode, pretrained=pretrained, device=device)
    log.info(
        "params: trainable=%d (%.4f%%) total=%d  lora_modules=%d",
        info["trainable_params"],
        info["trainable_pct"],
        info["total_params"],
        len(info["lora_modules"]),
    )

    class_weights = compute_class_weights(cfg)
    out_dir = run_dir or cfg.run_dir
    summary = train_model(
        model, train_loader, val_loader, cfg, device, out_dir, mode, class_weights, max_steps
    )
    summary["seed"] = seed
    summary["label_fraction"] = cfg.train.label_fraction
    summary["n_train_images"] = len(train_ds)
    summary["n_val_images"] = len(val_ds)
    summary["n_test_images"] = len(test_ds)
    summary["adaptation"] = {k: v for k, v in info.items() if k != "lora_modules"}
    summary["lora_modules"] = info.get("lora_modules", [])[:8]
    summary["device"] = device_name(device)
    (Path(out_dir) / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


@app.command()
def main(
    config: str = ConfigOpt,
    mode: str = typer.Option("linear", "--mode", "-m", help="linear | lora(_rN) | full"),
    label_fraction: float = typer.Option(
        None, "--label-fraction", help="Fraction of train labels to use."
    ),
    seed: int = typer.Option(None, "--seed", help="Override the config seed."),
    run_name: str = typer.Option(None, "--run-name", help="Override artifacts/runs/<name>."),
    epochs: int = typer.Option(None, "--epochs", help="Override the epoch count."),
    max_steps: int = typer.Option(None, "--max-steps", help="Cap steps per epoch (smoke runs)."),
    pretrained: bool = typer.Option(True, "--pretrained/--no-pretrained", help="Load HF weights."),
    set_: list[str] = SetOpt,
) -> None:
    """Train the perception model in ``linear``, ``lora`` or ``full`` mode."""
    cfg = load_cli_config(config, set_)
    t0 = time.perf_counter()
    summary = run_train(cfg, mode, label_fraction, seed, run_name, epochs, max_steps, pretrained)
    typer.echo(
        "\n".join(
            [
                f"run            : {cfg.run.name}",
                f"mode           : {mode}",
                f"device         : {summary['device']}",
                f"epochs_run     : {summary['epochs_run']} (best @ {summary['best_epoch']})",
                f"trainable      : {summary['adaptation']['trainable_params']:,} "
                f"({summary['adaptation']['trainable_pct']:.4f}% of {summary['adaptation']['total_params']:,})",
                f"train_seconds  : {summary['train_seconds']}",
                f"checkpoint     : {summary['checkpoint']}",
                f"wall_seconds   : {round(time.perf_counter() - t0, 2)}",
            ]
        )
    )
    typer.echo(f"manifest       : {default_manifest_path(cfg)}")


if __name__ == "__main__":  # pragma: no cover
    app()
