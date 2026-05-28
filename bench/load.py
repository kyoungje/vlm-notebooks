"""Concurrent load runner for the NPU-vs-shared-iGPU experiment.

The memory-footprint bench (notebook 02) fires one request at a time. The
interference story needs the opposite: drive the LLM path and the Whisper
STT path *simultaneously* for a fixed wall-clock window, the way an
always-on voice assistant does — mic streaming while the model is mid-answer.

Two configs, identical except where Whisper runs:

    Config A (split):  Gemma on iGPU (:9000) + Whisper on NPU   (:9010)
    Config B (shared): Gemma on iGPU (:9000) + Whisper on iGPU  (:9010)

This module only *drives and records*; it spawns no servers (same contract
as the rest of the harness — the operator brings both up beforehand and
flips WHISPER_DEVICE between passes). Every request is timestamped on a
single process-wide perf_counter clock so LLM token arrivals and STT spans
line up on one axis — that alignment is what makes the interference visible.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from . import client


@dataclass
class Event:
    """One completed request, times relative to the run's t0 (seconds)."""
    kind: str                       # 'llm' | 'stt'
    start: float
    end: float
    ok: bool = True
    error: str = ""
    # llm-only
    ttft_s: Optional[float] = None
    completion_tokens: Optional[int] = None
    decode_tps: Optional[float] = None
    token_times: Optional[list[float]] = None  # relative to t0
    # stt-only
    rtf: Optional[float] = None
    audio_dur_s: Optional[float] = None

    @property
    def latency_s(self) -> float:
        return self.end - self.start


@dataclass
class RunResult:
    label: str
    duration_s: float
    events: list[Event]
    meta: dict = field(default_factory=dict)

    def of(self, kind: str) -> list[Event]:
        return [e for e in self.events if e.kind == kind and e.ok]

    # ---- aggregate metrics ----
    def llm_decode_tps(self) -> list[float]:
        return [e.decode_tps for e in self.of("llm") if e.decode_tps is not None]

    def llm_ttft(self) -> list[float]:
        return [e.ttft_s for e in self.of("llm") if e.ttft_s is not None]

    def stt_latency(self) -> list[float]:
        return [e.latency_s for e in self.of("stt")]

    def stt_rtf(self) -> list[float]:
        return [e.rtf for e in self.of("stt") if e.rtf is not None]

    def total_completion_tokens(self) -> int:
        return sum(e.completion_tokens or 0 for e in self.of("llm"))

    def total_audio_s(self) -> float:
        return sum(e.audio_dur_s or 0.0 for e in self.of("stt"))

    def llm_tokens_per_s(self) -> float:
        """System-level LLM throughput over the whole window."""
        return self.total_completion_tokens() / self.duration_s if self.duration_s else 0.0

    def audio_s_per_s(self) -> float:
        """Audio-seconds transcribed per wall-clock second (>1 = keeping up
        with more than one realtime stream)."""
        return self.total_audio_s() / self.duration_s if self.duration_s else 0.0

    def errors(self) -> list[Event]:
        return [e for e in self.events if not e.ok]

    def instantaneous_llm_tps(self, bin_s: float = 0.5) -> tuple[list[float], list[float]]:
        """Bin every LLM token arrival into `bin_s` windows -> (t_centers, tok/s).

        This is the line in the interference timeline: in the split config it
        stays flat; in the shared config it dips each time a transcription
        steals the iGPU.
        """
        times = []
        for e in self.of("llm"):
            if e.token_times:
                times.extend(e.token_times)
        if not times:
            return [], []
        edges = np.arange(0.0, self.duration_s + bin_s, bin_s)
        counts, _ = np.histogram(times, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2.0
        tps = counts / bin_s
        return centers.tolist(), tps.tolist()

    def stt_spans(self) -> list[tuple[float, float]]:
        """(start, end) of each transcription, for shading the timeline."""
        return [(e.start, e.end) for e in self.of("stt")]


def percentiles(xs: list[float], ps=(50, 95, 99)) -> dict[int, float]:
    if not xs:
        return {p: float("nan") for p in ps}
    arr = np.asarray(xs, dtype=float)
    return {p: float(np.percentile(arr, p)) for p in ps}


def run_concurrent(
    *,
    label: str,
    duration_s: float,
    # LLM path
    llm_base_url: str,
    llm_model: str,
    llm_prompts: list[dict],
    llm_workers: int = 1,
    llm_max_tokens: Optional[int] = None,   # override per-prompt max_tokens if set
    # STT path
    stt_base_url: str,
    stt_model: str,
    stt_audio_path: str | Path,
    stt_audio_dur_s: Optional[float] = None,
    stt_language: Optional[str] = None,
    stt_workers: int = 1,
    stt_gap_s: float = 0.0,                 # pause between a worker's transcriptions
    meta: Optional[dict] = None,
) -> RunResult:
    """Drive LLM + STT concurrently for `duration_s`, return timestamped events.

    LLM workers fire streaming chat completions back-to-back (continuous
    generation). STT workers fire transcriptions back-to-back, optionally
    spaced by `stt_gap_s` to model utterance cadence. Increase `stt_workers`
    to sweep the number of simultaneous audio streams — the split config
    holds; the shared config degrades.
    """
    events: list[Event] = []
    lock = threading.Lock()
    stop = threading.Event()
    t0 = time.perf_counter()
    deadline = t0 + duration_s

    def record(e: Event) -> None:
        with lock:
            events.append(e)

    def llm_worker(idx: int) -> None:
        i = idx
        while not stop.is_set() and time.perf_counter() < deadline:
            p = llm_prompts[i % len(llm_prompts)]
            i += 1
            mt = llm_max_tokens if llm_max_tokens is not None else p.get("max_tokens", 128)
            try:
                r = client.call(
                    base_url=llm_base_url, model=llm_model,
                    prompt=p["prompt"], max_tokens=mt, stream=True,
                )
                tt = [(t - t0) for t in (r.token_times or [])]
                record(Event(
                    kind="llm",
                    start=(r.start_perf - t0) if r.start_perf else (time.perf_counter() - t0),
                    end=(r.end_perf - t0) if r.end_perf else (time.perf_counter() - t0),
                    ttft_s=r.ttft_s, completion_tokens=r.completion_tokens,
                    decode_tps=r.decode_tps, token_times=tt,
                ))
            except Exception as e:  # noqa: BLE001 — record, don't kill the run
                now = time.perf_counter() - t0
                record(Event(kind="llm", start=now, end=now, ok=False, error=repr(e)))

    def stt_worker(_idx: int) -> None:
        while not stop.is_set() and time.perf_counter() < deadline:
            try:
                r = client.transcribe(
                    base_url=stt_base_url, model=stt_model,
                    audio_path=stt_audio_path, audio_dur_s=stt_audio_dur_s,
                    language=stt_language,
                )
                record(Event(
                    kind="stt",
                    start=r.start_perf - t0, end=r.end_perf - t0,
                    rtf=r.rtf, audio_dur_s=r.audio_dur_s,
                ))
            except Exception as e:  # noqa: BLE001
                now = time.perf_counter() - t0
                record(Event(kind="stt", start=now, end=now, ok=False, error=repr(e)))
            if stt_gap_s > 0:
                stop.wait(stt_gap_s)

    n = llm_workers + stt_workers
    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(llm_worker, i) for i in range(llm_workers)]
        futs += [pool.submit(stt_worker, i) for i in range(stt_workers)]
        # Wait out the window, then let in-flight requests drain.
        while time.perf_counter() < deadline:
            time.sleep(0.1)
        stop.set()
        for f in futs:
            f.result()

    m = dict(meta or {})
    m.update(llm_workers=llm_workers, stt_workers=stt_workers,
             stt_gap_s=stt_gap_s, llm_max_tokens=llm_max_tokens)
    return RunResult(label=label, duration_s=duration_s, events=events, meta=m)


def save_run(run: RunResult, path: str | Path) -> None:
    """Persist a run as JSON so the two configs survive a kernel restart and
    can be charted together later (same cumulative pattern as the memory CSV)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "label": run.label,
        "duration_s": run.duration_s,
        "meta": run.meta,
        "events": [asdict(e) for e in run.events],
    }, indent=2))


def load_run(path: str | Path) -> RunResult:
    d = json.loads(Path(path).read_text())
    events = [Event(**e) for e in d["events"]]
    return RunResult(label=d["label"], duration_s=d["duration_s"],
                     events=events, meta=d.get("meta", {}))


def summary(run: RunResult) -> dict:
    """Compact dict for printing / CSV — the numbers behind the charts."""
    tpot = run.llm_decode_tps()
    ttft = run.llm_ttft()
    rtf = run.stt_rtf()
    stt_lat = run.stt_latency()
    return {
        "label": run.label,
        "duration_s": round(run.duration_s, 1),
        "llm_calls": len(run.of("llm")),
        "stt_calls": len(run.of("stt")),
        "errors": len(run.errors()),
        "llm_tokens_per_s": round(run.llm_tokens_per_s(), 1),
        "audio_s_per_s": round(run.audio_s_per_s(), 2),
        "llm_decode_tps_p50": round(percentiles(tpot)[50], 1),
        "ttft_p95": round(percentiles(ttft)[95], 3),
        "stt_latency_p50": round(percentiles(stt_lat)[50], 3),
        "stt_latency_p95": round(percentiles(stt_lat)[95], 3),
        "stt_latency_p99": round(percentiles(stt_lat)[99], 3),
        "stt_rtf_p50": round(percentiles(rtf)[50], 3),
        "stt_rtf_p95": round(percentiles(rtf)[95], 3),
    }
