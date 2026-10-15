"""``quality`` -- run the test suite with coverage and record the real numbers.

    aerialvit quality

Runs ``pytest -q -m "not slow" --cov=aerial_vit``, parses the actual summary
line and the coverage JSON report, and writes ``artifacts/results/quality.json``.
Nothing here estimates: if pytest or coverage cannot be parsed, the fields are
recorded as ``null`` rather than guessed.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

import typer

from aerial_vit import __version__
from aerial_vit.config import REPO_ROOT, RESULTS_DIR
from aerial_vit.utils.logging import get_logger

app = typer.Typer(add_completion=False, help="Run tests + coverage and record them.")
log = get_logger("aerialvit.quality")

_PASSED = re.compile(r"(\d+) passed")
_DESELECTED = re.compile(r"(\d+) deselected")
_FAILED = re.compile(r"(\d+) failed")
_ERROR = re.compile(r"(\d+) error")
_COLLECTED = re.compile(r"(\d+) tests? collected")
_COV_TOTAL = re.compile(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", re.MULTILINE)


def parse_pytest_output(text: str) -> dict[str, int | None]:
    def _one(rx):
        m = rx.search(text)
        return int(m.group(1)) if m else None

    passed = _one(_PASSED)
    failed = _one(_FAILED)
    errors = _one(_ERROR)
    deselected = _one(_DESELECTED)
    collected = None
    if passed is not None or failed is not None:
        collected = (passed or 0) + (failed or 0) + (errors or 0) + (deselected or 0)
    return {
        "tests_passed": passed,
        "tests_failed": failed,
        "tests_errors": errors,
        "tests_deselected": deselected,
        "tests_collected": collected,
    }


def run_quality(
    marker: str = "not slow",
    coverage_target: str = "aerial_vit",
    timeout: int = 3600,
) -> dict:
    cov_json = REPO_ROOT / "coverage.json"
    if cov_json.exists():
        cov_json.unlink()

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        marker,
        f"--cov={coverage_target}",
        "--cov-report=term-missing",
        f"--cov-report=json:{cov_json}",
    ]
    log.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    output = proc.stdout + "\n" + proc.stderr

    stats = parse_pytest_output(output)
    coverage_pct = None
    covered_lines = total_lines = None
    if cov_json.exists():
        data = json.loads(cov_json.read_text())
        totals = data.get("totals", {})
        coverage_pct = round(float(totals.get("percent_covered", 0.0)), 2)
        covered_lines = int(totals.get("covered_lines", 0))
        total_lines = int(totals.get("num_statements", 0))
    if coverage_pct is None:
        m = _COV_TOTAL.search(output)
        if m:
            coverage_pct = float(m.group(1))

    ruff = None
    try:
        r = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "."],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        ruff = "clean" if r.returncode == 0 else "issues"
    except Exception:  # noqa: BLE001
        ruff = None

    report = {
        **stats,
        "coverage_pct": coverage_pct,
        "covered_lines": covered_lines,
        "total_statements": total_lines,
        "marker": marker,
        "coverage_target": coverage_target,
        "returncode": proc.returncode,
        "ruff": ruff,
        "python": sys.version.split()[0],
        "pytest_command": " ".join(cmd),
        "package_version": __version__,
        "summary_line": next(
            (
                ln
                for ln in reversed(output.splitlines())
                if ("passed" in ln or "failed" in ln) and "=" in ln
            ),
            "",
        ).strip("= "),
    }
    out = RESULTS_DIR / "quality.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=float) + "\n")
    log.info("wrote %s", out)
    return report


@app.command()
def main(
    marker: str = typer.Option("not slow", "--marker", help="pytest -m expression."),
    coverage_target: str = typer.Option("aerial_vit", "--cov"),
    allow_fail: bool = typer.Option(
        False, "--allow-fail", help="Do not exit non-zero on test failure."
    ),
) -> None:
    """Run the marked test suite with coverage and record the result."""
    r = run_quality(marker, coverage_target)
    typer.echo(
        f"passed={r['tests_passed']} failed={r['tests_failed']} deselected={r['tests_deselected']} "
        f"coverage={r['coverage_pct']}% ruff={r['ruff']}"
    )
    if r["returncode"] != 0 and not allow_fail:
        raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
