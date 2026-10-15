#!/usr/bin/env python
"""Thin shim: ``python scripts/bench_latency.py`` == ``aerialvit bench-latency``.

The implementation lives in :mod:`aerial_vit.commands.bench_latency` so the installed
console script and the in-repo script are the same code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aerial_vit.commands.bench_latency import app  # noqa: E402

if __name__ == "__main__":
    app()
