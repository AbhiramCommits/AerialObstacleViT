"""Memory profiling helpers.

A plain ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is a *cumulative*
high-water mark, so using it per-cell produces a monotonically increasing number
that measures nothing.  Instead we sample the live resident set on a background
thread for the duration of a cell, and use ``torch.cuda.max_memory_allocated``
(exact, and resettable) when training on CUDA.
"""

from __future__ import annotations

import os
import threading
from typing import Any

import torch

__all__ = ["current_rss_mb", "MemorySampler", "device_memory_mb"]


def _read_proc_statm_mb() -> float | None:
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:  # noqa: BLE001 - not Linux
        return None


def current_rss_mb() -> float:
    """Live resident set size of this process, in MB."""
    proc = _read_proc_statm_mb()
    if proc is not None:
        return proc
    try:
        import subprocess

        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=10,
        )
        kb = float((out.stdout or "0").strip() or 0)
        if kb > 0:
            return kb / 1024.0
    except Exception:  # noqa: BLE001
        pass
    rss = 0
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:  # noqa: BLE001
        return 0.0
    return rss / 1e6 if rss > 1e7 else rss / 1e3


def device_memory_mb(device: torch.device) -> float:
    """Device-attributed memory in MB where the backend can report it."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            return torch.mps.driver_allocated_memory() / 1e6
        except Exception:  # noqa: BLE001
            pass
    return 0.0


class MemorySampler:
    """Context manager that records the peak live memory over a code block.

    Usage::

        with MemorySampler(device) as mem:
            train(...)
        mem.peak_mb   # -> float
    """

    def __init__(self, device: torch.device | str = "cpu", interval: float = 0.25):
        self.device = torch.device(device)
        self.interval = float(interval)
        self.peak_mb = 0.0
        self._samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- sampling ---------------------------------------------------------- #
    def sample(self) -> float:
        value = current_rss_mb()
        self.peak_mb = max(self.peak_mb, value)
        self._samples += 1
        return value

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    # -- context manager --------------------------------------------------- #
    def __enter__(self) -> MemorySampler:
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.sample()
        if self.device.type == "cuda":
            self.peak_mb = max(self.peak_mb, device_memory_mb(self.device))
        elif self.device.type == "mps":
            dev = device_memory_mb(self.device)
            if dev > 0:
                self.peak_mb = max(self.peak_mb, dev)

    def as_dict(self) -> dict[str, float]:
        return {
            "peak_memory_mb": round(float(self.peak_mb), 2),
            "memory_samples": int(self._samples),
            "memory_seconds": round(self.interval * self._samples, 2),
        }
