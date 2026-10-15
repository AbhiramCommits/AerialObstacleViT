"""Environment provenance: the facts a reader needs to interpret the numbers.

Everything here is *collected*, never typed in: the device string comes from
PyTorch, the git hash from ``git rev-parse``, the versions from the installed
packages.  That is what makes the README's "measured on ..." line auditable.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["collect_provenance", "write_provenance"]

_COLLECTED = False


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def _git(*args: str, cwd: str | Path | None = None) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def _cpu_model() -> str | None:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or _run(
            ["sysctl", "-n", "hw.model"]
        )
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or None


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def collect_provenance(
    device: Any = None,
    repo_root: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect a JSON-serialisable provenance record."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]

    torch_version = accelerator = None
    cpu_count = None
    try:
        import torch

        torch_version = torch.__version__
        cpu_count = torch.get_num_threads()
        if device is not None:
            from aerial_vit.utils.seed import device_name

            accelerator = device_name(torch.device(str(device)))
        elif torch.cuda.is_available():
            accelerator = f"cuda:{torch.cuda.get_device_name(0)}"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            accelerator = "apple-mps (Metal Performance Shaders)"
        else:
            accelerator = "cpu"
    except Exception:  # noqa: BLE001
        accelerator = "cpu"

    record = {
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_threads": cpu_count,
        "accelerator": accelerator,
        "device": accelerator,
        "torch": torch_version,
        "packages": {
            name: _pkg_version(name)
            for name in (
                "torchvision",
                "timm",
                "transformers",
                "peft",
                "onnx",
                "onnxruntime",
                "numpy",
                "scikit-learn",
                "opencv-python-headless",
                "pycocotools",
                "matplotlib",
                "pandas",
                "scipy",
            )
        },
        "git_commit": _git("rev-parse", "HEAD", cwd=root),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root),
        "git_dirty": bool(_git("status", "--porcelain", cwd=root)),
        "aerialvit": _pkg_version("aerialvit"),
    }
    if extra:
        record.update(extra)
    return record


def write_provenance(
    path: str | Path, device: Any = None, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    record = collect_provenance(device=device, extra=extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, default=str) + "\n")
    return record
