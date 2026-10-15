#!/usr/bin/env python
"""Thin shim: ``python scripts/make_report.py`` == ``aerialvit make-report``.

The implementation lives in :mod:`aerial_vit.commands.make_report` so the installed
console script and the in-repo script are the same code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aerial_vit.commands.make_report import app  # noqa: E402

if __name__ == "__main__":
    app()
