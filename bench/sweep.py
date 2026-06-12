"""Concurrency sweep for single-node throughput characterization (notebook 04).

notebook 03's `load.py` drives an LLM and a Whisper path together for one window
to expose compute contention. This module does something simpler: it points a
single OpenAI-compatible LLM/VLM backend at an increasing number of in-flight
requests and, at each level, reports aggregate decode throughput and tail
latency. The concurrency level where aggregate tok/s stops rising is that
node's serving capacity for the model — the per-node number the
two-B60-vs-RTX-6000-Ada cost comparison is built from.

No NPU path, no STT: notebook 04 characterizes one discrete Arc Pro B60 (24 GB,
Battlemage, `xe` driver) in an ordinary desktop. Same contract as the rest of
the harness — the operator brings up exactly one backend beforehand; this
module only drives and records.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import client


def percentiles(xs: list[float], ps=(50, 95, 99)) -> dict[int, float]:
    if not xs:
        return {p: float("nan") for p in ps}
    arr = np.asarray(xs, dtype=float)
    return {p: float(np.percentile(arr, p)) for p in ps}


def _r(x: Optional[float], nd: int = 3) -> Optional[float]:
    """Round for JSON, mapping NaN/None to None so missing metrics serialize
    cleanly (and the plots can branch on them)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd)


@dataclass
class Req:
    """One request from a sweep level; times relative to the level's t0 (s)."""
    start: float
    end: float
    ok: bool = True
    error: str = ""
    ttft_s: Optional[float] = None
    decode_tps: Optional[float] = None
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None

    @property
    def latency_s(self) -> float:
        return self.end - self.start


@dataclass
class LevelResult:
    """All requests fired at one concurrency level, plus the peak memory the
    server held while doing so."""
    concurrency: int
    window_s: float
    reqs: list  # list[Req]
    peak_rss_mb: Optional[float] = None
    peak_igpu_mb: Optional[float] = None

    def completed(self) -> list:
        """Requests that finished *inside* the measured window. Anything still
        in flight at the deadline ends with `end > window_s` and is excluded, so
        the aggregate-throughput denominator is always exactly `window_s`."""
        return [r for r in self.reqs if r.ok and r.end <= self.window_s]

    def errors(self) -> list:
        return [r for r in self.reqs if not r.ok]

    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens or 0 for r in self.completed())

    def agg_tokens_per_s(self) -> float:
        """The headline: decode tokens emitted across all streams per wall-clock
        second. This is the curve whose knee defines serving capacity."""
        return self.total_completion_tokens() / self.window_s if self.window_s else 0.0

    def ttft(self) -> list[float]:
        return [r.ttft_s for r in self.completed() if r.ttft_s is not None]

    def decode_tps(self) -> list[float]:
        return [r.decode_tps for r in self.completed() if r.decode_tps is not None]

    def latency(self) -> list[float]:
        return [r.latency_s for r in self.completed()]

    def streaming(self) -> bool:
        """True when the backend actually streamed (so TTFT/decode are real, not
        the None the non-streaming fallback returns)."""
        return len(self.ttft()) > 0

    def summary(self) -> dict:
        tt = percentiles(self.ttft())
        dt = percentiles(self.decode_tps())
        lat = percentiles(self.latency())
        return {
            "concurrency": self.concurrency,
            "window_s": round(self.window_s, 1),
            "completed": len(self.completed()),
            "errors": len(self.errors()),
            "agg_tokens_per_s": round(self.agg_tokens_per_s(), 1),
            "ttft_p50": _r(tt[50]), "ttft_p95": _r(tt[95]), "ttft_p99": _r(tt[99]),
            "decode_tps_p50": _r(dt[50], 1),
            "latency_p50": _r(lat[50]), "latency_p95": _r(lat[95]),
            "latency_p99": _r(lat[99]),
            "streaming": self.streaming(),
            "peak_igpu_mb": _r(self.peak_igpu_mb, 1),
            "peak_rss_mb": _r(self.peak_rss_mb, 1),
        }


def single_stream(
    *,
    base_url: str,
    model: str,
    prompts: list[dict],
    max_tokens: int,
    repeats: int = 5,
) -> dict:
    """Uncontended baseline: `repeats` sequential streaming calls, no overlap.

    Returns medians (TTFT, decode tok/s, total latency) over the calls. This is
    the number a single RTX 6000 Ada will beat per request — and that's fine;
    the B60 story is the aggregate throughput `sweep_level` measures, not
    per-request speed. `ttft_s`/`decode_tps` come back None if the backend
    doesn't stream (see client.call's SSE fallback)."""
    calls = []
    for i in range(repeats):
        p = prompts[i % len(prompts)]
        calls.append(client.call(
            base_url=base_url, model=model, prompt=p["prompt"],
            max_tokens=max_tokens, stream=True,
        ))
    ttft = [c.ttft_s for c in calls if c.ttft_s is not None]
    dtps = [c.decode_tps for c in calls if c.decode_tps is not None]
    lat = [c.total_latency_s for c in calls]
    toks = [c.completion_tokens for c in calls if c.completion_tokens]
    return {
        "repeats": repeats,
        "streaming": len(ttft) > 0,
        "ttft_s": _r(float(np.median(ttft)), 3) if ttft else None,
        "decode_tps": _r(float(np.median(dtps)), 1) if dtps else None,
        "latency_s": _r(float(np.median(lat)), 3) if lat else None,
        "completion_tokens": int(np.median(toks)) if toks else 0,
    }


def sweep_level(
    *,
    concurrency: int,
    window_s: float,
    base_url: str,
    model: str,
    prompts: list[dict],
    max_tokens: int,
) -> LevelResult:
    """Drive `concurrency` workers firing streaming chat completions back-to-back
    for `window_s`, return a LevelResult.

    Each worker loops: take the next prompt (strided by `concurrency` so workers
    don't all hammer the same one), fire a streaming call, record it, repeat
    until the window closes. Requests still in flight at the deadline are let
    drain and recorded, but only those that completed within the window count
    toward aggregate throughput. Failures are recorded as `ok=False` events
    rather than killing the sweep."""
    reqs: list[Req] = []
    lock = threading.Lock()
    stop = threading.Event()
    t0 = time.perf_counter()
    deadline = t0 + window_s

    def record(r: Req) -> None:
        with lock:
            reqs.append(r)

    def worker(idx: int) -> None:
        i = idx
        while not stop.is_set() and time.perf_counter() < deadline:
            p = prompts[i % len(prompts)]
            i += concurrency  # stride so the prompt mix stays even across workers
            try:
                res = client.call(
                    base_url=base_url, model=model, prompt=p["prompt"],
                    max_tokens=max_tokens, stream=True,
                )
                now = time.perf_counter()
                record(Req(
                    start=(res.start_perf - t0) if res.start_perf else (now - t0),
                    end=(res.end_perf - t0) if res.end_perf else (now - t0),
                    ttft_s=res.ttft_s, decode_tps=res.decode_tps,
                    completion_tokens=res.completion_tokens,
                    prompt_tokens=res.prompt_tokens,
                ))
            except Exception as e:  # noqa: BLE001 — record, don't kill the sweep
                now = time.perf_counter() - t0
                record(Req(start=now, end=now, ok=False, error=repr(e)))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(worker, i) for i in range(concurrency)]
        while time.perf_counter() < deadline:
            time.sleep(0.1)
        stop.set()
        for f in futs:           # let each worker's in-flight request drain
            f.result()

    return LevelResult(concurrency=concurrency, window_s=window_s, reqs=reqs)


def _rows(levels) -> list:
    return sorted(
        [lv if isinstance(lv, dict) else lv.summary() for lv in levels],
        key=lambda s: s["concurrency"],
    )


def knee(levels, rel_gain: float = 0.10) -> dict:
    """Serving-capacity knee: the *cheapest* concurrency that reaches essentially
    peak aggregate throughput.

    Find the peak tok/s across every level, then return the lowest-concurrency
    level whose throughput is within `rel_gain` of that peak (default 10%). This
    is robust to the two things a naive per-step rule trips on:

      - a single flat or noisy *early* step — it sits far below the peak, so it
        can't be mistaken for the knee (the old rule stopped at the first step
        that didn't gain 10%, so one flat 1→2 step pinned the knee at 1);
      - throughput *dropping* past saturation — those levels are below the peak
        too, so they're never selected.

    For a genuine plateau it returns the smaller concurrency: same throughput,
    lower latency. A hint for the writeup, not a hard claim — eyeball the curve.
    """
    rows = _rows(levels)
    if not rows:
        return {}
    peak_tp = max((r["agg_tokens_per_s"] or 0.0) for r in rows)
    if peak_tp <= 0:
        return rows[0]
    threshold = peak_tp * (1.0 - rel_gain)
    for r in rows:
        if (r["agg_tokens_per_s"] or 0.0) >= threshold:
            return r
    return rows[0]


def capacity_report(levels, rel_gain: float = 0.10) -> dict:
    """Knee + peak + an over-saturation flag — what the notebook prints and the
    plot annotates.

      knee      — cheapest near-peak level (see `knee`).
      peak      — the highest-throughput level.
      saturated — True if any level *past* the peak concurrency has lower
                  throughput, i.e. adding in-flight requests made things worse:
                  the GPU is past its happy point and requests are queueing.
                  When True, the knee is a real capacity ceiling, not just the
                  end of the swept range.
    """
    rows = _rows(levels)
    if not rows:
        return {"knee": {}, "peak": {}, "saturated": False}
    peak = max(rows, key=lambda s: s["agg_tokens_per_s"] or 0.0)
    peak_tp = peak["agg_tokens_per_s"] or 0.0
    saturated = any(
        (r["agg_tokens_per_s"] or 0.0) < peak_tp
        for r in rows if r["concurrency"] > peak["concurrency"]
    )
    return {"knee": knee(rows, rel_gain), "peak": peak, "saturated": saturated}


def save_sweep(
    *,
    path: str | Path,
    node: str,
    model: str,
    quantization: str,
    vram_total_mb: int,
    max_tokens: int,
    base_url: str,
    baseline: dict,
    levels: list,
    timestamp: str,
    meta: Optional[dict] = None,
) -> Path:
    """Persist a node's full characterization (baseline + every level) as one
    self-describing JSON under `data/b60/`. These per-node files are the inputs
    the Phase-2 two-node cost chart reads, so everything needed to interpret a
    number later lives in the file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "node": node,
        "model": model,
        "quantization": quantization,
        "vram_total_mb": vram_total_mb,
        "max_tokens": max_tokens,
        "base_url": base_url,
        "timestamp": timestamp,
        "baseline": baseline,
        "levels": [lv if isinstance(lv, dict) else lv.summary() for lv in levels],
        "meta": meta or {},
    }
    p.write_text(json.dumps(doc, indent=2))
    return p


def load_sweep(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
