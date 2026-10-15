#!/usr/bin/env python
"""Build README.md from a template + the table snippets rendered by ``make_report``.

This is what makes "README numbers match reports/RESULTS.md byte-for-byte"
mechanically true: the README does not restate any number, it *includes* the
exact files under ``reports/tables/`` that ``make_report`` wrote.

    python scripts/build_readme.py            # (re)write README.md
    python scripts/build_readme.py --check    # fail if README.md is stale
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aerial_vit.config import ARTIFACTS_DIR, REPORTS_DIR  # noqa: E402
from aerial_vit.utils.provenance import collect_provenance  # noqa: E402

TEMPLATE = REPORTS_DIR / "README.template.md"
INCLUDE = re.compile(r"\{\{\s*(table|artifact|file):([A-Za-z0-9_./-]+)\s*\}\}")


def _read(path: Path, default: str) -> str:
    return path.read_text() if path.exists() else default


def _resolve(kind: str, name: str, tables: dict[str, str]) -> str:
    if kind == "table":
        if name not in tables:
            raise KeyError(f"unknown table {name!r}; known: {sorted(tables)}")
        return tables[name].rstrip("\n")
    if kind == "artifact":
        p = ARTIFACTS_DIR / name
        if not p.exists():
            return f"_artifact `{name}` is not available yet._"
        if p.suffix == ".json":
            return "```json\n" + p.read_text().strip() + "\n```"
        return p.read_text().strip()
    if kind == "file":
        p = REPO_ROOT / name
        if not p.exists():
            return f"_file `{name}` is not available yet._"
        return p.read_text().rstrip("\n")
    raise ValueError(kind)


def build(check_only: bool = False) -> int:
    from aerial_vit.reporting import build_tables

    tables = build_tables()
    provenance = collect_provenance(repo_root=REPO_ROOT)
    template = _read(TEMPLATE, "# AerialObstacleViT\n\n(no template found)\n")

    def _sub(m: re.Match) -> str:
        return _resolve(m.group(1), m.group(2), tables)

    body = INCLUDE.sub(_sub, template)
    body = body.replace("{{provenance.device}}", str(provenance.get("accelerator") or "unknown"))
    body = body.replace("{{provenance.date}}", str(provenance.get("date") or "unknown"))
    body = body.replace("{{provenance.python}}", str(provenance.get("python") or "unknown"))
    body = body.replace("{{provenance.torch}}", str(provenance.get("torch") or "unknown"))
    body = body.replace(
        "{{provenance.git_commit}}", str(provenance.get("git_commit") or "uncommitted")
    )
    body = body.rstrip("\n") + "\n"

    readme = REPO_ROOT / "README.md"
    if check_only:
        current = readme.read_text() if readme.exists() else ""
        if current != body:
            print("README.md is stale; run `python scripts/build_readme.py`", file=sys.stderr)
            return 1
        print("README.md is up to date with reports/tables/")
        return 0

    readme.write_text(body)
    print(f"wrote {readme} ({len(body.splitlines())} lines)")
    return 0


def main() -> int:
    check_only = "--check" in sys.argv[1:]
    return build(check_only=check_only)


if __name__ == "__main__":
    raise SystemExit(main())
