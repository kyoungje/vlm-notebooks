"""Memory-footprint poller for the backend server process.

The bench runs against an external OpenAI-compatible server (OV / vLLM /
llama.cpp) that the operator brought up beforehand. We measure two things:

  1. RSS of the server process tree (system RAM the backend holds).
  2. Intel iGPU memory the server holds, by reading `drm-resident-*` keys
     from `/proc/<pid>/fdinfo/<fd>` for every DRM fd in the process tree.

The xe driver (Panther Lake / Lunar Lake / Battlemage) exposes per-fd
memory accounting in fdinfo with no root needed — see
https://docs.kernel.org/gpu/drm-usage-stats.html. The older `intel_gpu_top`
tool only reads i915 perf counters and does not work for xe, so we
deliberately don't depend on it.

Both samplers are best-effort: if a probe can't read its source it returns
None for that sample, the rest of the bench still works.

Typical use:

    with MemProbe(pid=server_pid, interval_s=0.5) as probe:
        ... run requests ...
    samples = probe.samples           # list[Sample]
    peak = probe.peak()               # Sample with max rss_mb
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


def _read_rss_kb(pid: int) -> Optional[int]:
    """RSS of `pid` plus all descendants, in KB. Returns None if pid is gone."""
    try:
        with open(f"/proc/{pid}/status") as f:
            pass
    except FileNotFoundError:
        return None

    total = 0
    stack = [pid]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        try:
            with open(f"/proc/{cur}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
                        break
            with open(f"/proc/{cur}/task/{cur}/children") as f:
                for tok in f.read().split():
                    stack.append(int(tok))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


def _iter_pids(root_pid: int):
    """Yield root_pid and every descendant PID (best-effort)."""
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        yield cur
        try:
            with open(f"/proc/{cur}/task/{cur}/children") as f:
                for tok in f.read().split():
                    stack.append(int(tok))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue


def _read_igpu_mem_mb(pid: int) -> Optional[float]:
    """Sum `drm-resident-*` (KiB) across every DRM fd held by `pid` and its
    descendants. Returns MB, or None if the process tree holds no DRM fds.

    The xe driver writes per-fd memory accounting into
    `/proc/<pid>/fdinfo/<fd>` as documented in
    https://docs.kernel.org/gpu/drm-usage-stats.html. i915 also supports
    this. We only count `drm-resident-*` (system / gtt / stolen / vramN);
    `drm-total-*` over-counts shared buffers.

    No root required — the standard /proc visibility rules apply.
    """
    total_kib = 0
    found_any_drm_fd = False
    for p in _iter_pids(pid):
        fd_dir = f"/proc/{p}/fd"
        try:
            fds = os.listdir(fd_dir)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        for fd in fds:
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            if not target.startswith("/dev/dri/"):
                continue
            found_any_drm_fd = True
            try:
                with open(f"/proc/{p}/fdinfo/{fd}") as f:
                    for line in f:
                        if not line.startswith("drm-resident-"):
                            continue
                        # format: "drm-resident-gtt:\t335024 KiB"
                        _, _, rhs = line.partition(":")
                        parts = rhs.split()
                        if len(parts) >= 1:
                            try:
                                total_kib += int(parts[0])
                            except ValueError:
                                continue
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
    if not found_any_drm_fd:
        return None
    return total_kib / 1024  # KiB -> MiB


@dataclass
class Sample:
    t: float                       # seconds since probe start
    rss_mb: Optional[float]
    igpu_mb: Optional[float]


@dataclass
class MemProbe:
    pid: int
    interval_s: float = 0.5
    sample_igpu: bool = True
    samples: list[Sample] = field(default_factory=list)
    _thread: Optional[threading.Thread] = None
    _stop: threading.Event = field(default_factory=threading.Event)
    _t0: float = 0.0

    def __enter__(self) -> "MemProbe":
        self._t0 = time.perf_counter()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2 * self.interval_s + 1)

    def _loop(self) -> None:
        while not self._stop.is_set():
            rss_kb = _read_rss_kb(self.pid)
            rss_mb = (rss_kb / 1024) if rss_kb is not None else None
            igpu = _read_igpu_mem_mb(self.pid) if self.sample_igpu else None
            self.samples.append(Sample(
                t=time.perf_counter() - self._t0,
                rss_mb=rss_mb,
                igpu_mb=igpu,
            ))
            self._stop.wait(self.interval_s)

    def peak(self) -> Optional[Sample]:
        valid = [s for s in self.samples if s.rss_mb is not None]
        return max(valid, key=lambda s: s.rss_mb) if valid else None

    def baseline(self) -> Optional[Sample]:
        """First sample with a valid rss reading."""
        for s in self.samples:
            if s.rss_mb is not None:
                return s
        return None


def find_server_pid(port: int) -> Optional[int]:
    """Find the PID listening on `port` (best-effort, Linux only)."""
    try:
        out = subprocess.check_output(
            ["ss", "-ltnp", f"sport = :{port}"],
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    # ss prints e.g. users:(("python",pid=12345,fd=7))
    import re
    m = re.search(r"pid=(\d+)", out)
    if m:
        return int(m.group(1))
    # Fallback: walk /proc
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/net/tcp") as f:
                hex_port = f"{port:04X}"
                for line in f.readlines()[1:]:
                    parts = line.split()
                    if len(parts) > 1 and parts[1].endswith(f":{hex_port}"):
                        return int(entry)
        except (FileNotFoundError, PermissionError):
            continue
    return None
