#!/usr/bin/env python
"""Shim: ``python scripts/quality.py`` == ``aerialvit quality``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aerial_vit.commands.quality import app  # noqa: E402

if __name__ == "__main__":
    app()
