"""``aerialvit`` -- the single console entrypoint for every pipeline stage."""

from __future__ import annotations

import typer

from aerial_vit import __version__

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="AerialObstacleViT: foundation features vs open-vocabulary VLMs for UAV perception.",
)


def _register() -> None:
    # Import lazily so `aerialvit --help` stays fast. Commands that are not yet
    # installed in a given checkout are skipped with a warning rather than
    # breaking the whole CLI.
    import importlib

    mapping = [
        ("fetch_data", "fetch-data"),
        ("train", "train"),
        ("eval", "eval"),
        ("zeroshot_baseline", "zeroshot"),
        ("ablate", "ablate"),
        ("mine_hard_cases", "mine-hard-cases"),
        ("export", "export"),
        ("bench_latency", "bench-latency"),
        ("make_report", "make-report"),
        ("quality", "quality"),
    ]
    for module_name, cmd_name in mapping:
        try:
            module = importlib.import_module(f"aerial_vit.commands.{module_name}")
        except Exception as exc:  # noqa: BLE001 - keep the CLI usable
            typer.echo(
                f"[warn] command '{cmd_name}' unavailable: {type(exc).__name__}: {exc}", err=True
            )
            continue
        app.add_typer(module.app, name=cmd_name)


_register()


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
