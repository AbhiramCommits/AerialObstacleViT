#!/usr/bin/env python
"""Print the best ablation cell as ``<run_name> <checkpoint-path>``.

Used by ``scripts/run_all.sh`` so that the hard-case, export and latency stages
all operate on the *same* best model, chosen from the measured artifacts rather
than hard-coded.

    python scripts/best_run.py            # by mIoU
    python scripts/best_run.py --metric map_50
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def best_cell(summary_path: Path, metric: str = "miou") -> tuple[str, float] | None:
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    best = None
    for key, val in summary.items():
        if key.startswith("_") or not isinstance(val, dict) or "mode" not in val:
            continue
        score = val.get(f"{metric}_mean")
        if score is None:
            continue
        if best is None or score > best[1]:
            best = (key, score)
    return best


def main() -> int:
    metric = "miou"
    if "--metric" in sys.argv:
        metric = sys.argv[sys.argv.index("--metric") + 1]

    summary_path = REPO_ROOT / "artifacts" / "results" / "ablation_summary.json"
    best = best_cell(summary_path, metric)
    if best is None:
        print("")
        return 1

    key, score = best
    mode, _, frac = key.partition("@")
    frac_i = int(round(float(frac) * 100))
    run_name = f"{mode}_f{frac_i:03d}_s1337"
    ckpt = REPO_ROOT / "artifacts" / "runs" / "ablation" / run_name / "best.pt"
    if not ckpt.exists():
        print("", file=sys.stderr)
        print(f"# best cell {key} ({metric}={score}) has no checkpoint at {ckpt}", file=sys.stderr)
        return 1
    print(f"{run_name} {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
