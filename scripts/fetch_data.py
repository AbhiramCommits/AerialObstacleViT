#!/usr/bin/env python
"""Thin shim: ``python scripts/fetch_data.py`` == ``aerialvit fetch-data``.

The implementation lives in :mod:`aerial_vit.commands.fetch_data` so that the
installed console script and the in-repo script behave identically.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aerial_vit.commands.fetch_data import app  # noqa: E402

if __name__ == "__main__":
    app()
