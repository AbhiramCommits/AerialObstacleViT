#!/usr/bin/env python
"""Shim generator: keeps ``scripts/*.py`` in sync with ``aerialvit <cmd>``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEMPLATE = '''#!/usr/bin/env python
"""Thin shim: ``python scripts/{mod}.py`` == ``aerialvit {cmd}``.

The implementation lives in :mod:`aerial_vit.commands.{mod}` so the installed
console script and the in-repo script are the same code path.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aerial_vit.commands.{mod} import app  # noqa: E402

if __name__ == "__main__":
    app()
'''

SHIMS = {
    "train": "train",
    "eval": "eval",
    "zeroshot_baseline": "zeroshot",
    "ablate": "ablate",
    "mine_hard_cases": "mine-hard-cases",
    "export": "export",
    "bench_latency": "bench-latency",
    "make_report": "make-report",
}


def main() -> None:
    here = Path(__file__).resolve().parent
    for mod, cmd in SHIMS.items():
        p = here / f"{mod}.py"
        p.write_text(TEMPLATE.format(mod=mod, cmd=cmd))
        p.chmod(0o755)
        print(f"wrote {p.relative_to(here.parent)}")


if __name__ == "__main__":
    main()
