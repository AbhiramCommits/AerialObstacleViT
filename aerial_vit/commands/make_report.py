"""``make-report`` -- render reports/RESULTS.md from artifacts only.

    aerialvit make-report

The README embeds the files in ``reports/tables/`` directly, so the numbers in
the README and in ``reports/RESULTS.md`` are the same bytes by construction.
"""

from __future__ import annotations

import typer

from aerial_vit.config import REPORTS_DIR
from aerial_vit.reporting import TABLE_ORDER, TABLES_DIR, write_report
from aerial_vit.utils.logging import get_logger

app = typer.Typer(add_completion=False, help="Render reports/RESULTS.md from artifacts.")
log = get_logger("aerialvit.report")


@app.command()
def main(
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Only print the summary line."),
) -> None:
    """Render the report tables."""
    payload = write_report()
    missing = [k for k, v in payload["artifacts_present"].items() if not v]
    if not quiet:
        typer.echo(payload["headline"])
        typer.echo("")
    typer.echo(f"tables   : {len(TABLE_ORDER)} -> {TABLES_DIR}")
    typer.echo(f"report   : {REPORTS_DIR / 'RESULTS.md'}")
    typer.echo(f"tables json: {REPORTS_DIR / 'results_tables.json'}")
    if missing:
        typer.echo(f"pending artifacts: {', '.join(missing)}")


if __name__ == "__main__":  # pragma: no cover
    app()
