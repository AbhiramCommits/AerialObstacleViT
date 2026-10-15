"""Shared CLI plumbing: config loading with ``--set`` overrides."""

from __future__ import annotations

from pathlib import Path

import typer

from aerial_vit.config import Config, load_config, parse_override_args


def load_cli_config(config: str, set_: list[str] | None = None) -> Config:
    return load_config(config, overrides=parse_override_args(set_))


ConfigOpt = typer.Option("configs/smoke.yaml", "--config", "-c", help="Path to a YAML config.")
SetOpt = typer.Option(
    None, "--set", "-s", help="Dotted override, repeatable, e.g. -s train.epochs=3"
)


def repo_path(p: str | Path) -> Path:
    """Resolve a possibly-relative path against the CWD."""
    return Path(p).expanduser()
