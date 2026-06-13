"""Matplotlib helpers for the OV-vs-vLLM bench notebooks.

Kept tiny on purpose — the notebooks own layout/titles, this module only
encapsulates the chart shapes we use repeatedly so the notebooks read as
a story instead of a wall of matplotlib calls.
"""
from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt


# Stable backend colors so the same backend is the same color across notebooks.
BACKEND_COLORS = {
    "openvino": "#0071c5",     # Intel blue
    "vllm": "#d62728",
    "llamacpp": "#2ca02c",
}

# Config colors for the NPU-vs-shared notebook (03). The split config (Whisper
# on NPU) is the "good" Intel-blue line; the shared config (both on iGPU) is
# red, the contended one.
CONFIG_COLORS = {
    "split": "#0071c5",   # Gemma iGPU + Whisper NPU
    "shared": "#d62728",  # Gemma iGPU + Whisper iGPU
}

# The B60 throughput notebook (04) brands its one node Intel blue — it's an
# Arc card, and the cost-effectiveness story is the Intel-blue line rising.
B60_COLOR = "#0071c5"


def _color(backend: str) -> str:
    return BACKEND_COLORS.get(backend.lower(), "#7f7f7f")


def _config_color(label: str) -> str:
    """Pick a config color by substring so callers can pass human labels
    like 'split (NPU)' / 'shared (iGPU)'."""
    low = label.lower()
    for key, c in CONFIG_COLORS.items():
        if key in low:
            return c
    return "#7f7f7f"


def _lighten(hex_color: str, factor: float = 0.55) -> str:
    """Blend `hex_color` toward white by `factor` (0=no change, 1=white)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = round(r + (255 - r) * factor)
    g = round(g + (255 - g) * factor)
    b = round(b + (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def memory_vs_model_size(
    rows: list[dict],
    *,
    ax=None,
    title: Optional[str] = None,
    backend_order: Optional[list[str]] = None,
    annotate: bool = True,
):
    """Side-by-side bar chart: RSS and iGPU resident per (backend, model).

    Each (backend, model) cell produces two adjacent bars:
      - solid: peak server RSS (GB)
      - hatched/lighter: peak iGPU resident (GB) — from drm-resident-*

    NOTE: On a shared-memory iGPU (like Panther Lake) the iGPU pages also
    count toward RSS. RSS and iGPU are NOT additive — they're two different
    views of overlapping memory. Shown side-by-side so the reader can see
    both numbers; do not sum them.

    `rows` items shape:
        {
          "backend": "vllm" | "openvino" | "llamacpp",
          "model": "Qwen2.5-VL-3B-Instruct",
          "params_b": 3.0,
          "peak_rss_mb": 18234.5,
          "peak_igpu_mb": 1450.0 | None,
        }
    """
    import matplotlib.patches as mpatches

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5.5))

    # x-axis: models in ascending size order
    models = sorted({r["model"] for r in rows}, key=lambda m: next(
        r["params_b"] for r in rows if r["model"] == m
    ))
    # backend order: caller can pin; otherwise alphabetical (deterministic)
    present = {r["backend"] for r in rows}
    if backend_order:
        backends = [b for b in backend_order if b in present]
        backends += sorted(present - set(backends))
    else:
        backends = sorted(present)

    # geometry: each model gets a slot centered under its xtick. Within the
    # slot we place only the (backend, metric) pairs that actually have data
    # for THIS model, centered as a group so the tick label aligns with the
    # bars even when a model has just one backend.
    bar_w = 0.78 / max(2 * len(backends), 1)   # width of one bar
    gap   = bar_w * 0.08                       # small visual gap between bars

    max_y = 0.0
    for j, m in enumerate(models):
        # collect drawable units for this model, in (backend_order, RSS first then iGPU) order
        units = []  # list of (color, edge, hatch, height_gb, kind) where kind in {'rss','igpu'}
        for b in backends:
            match = next((r for r in rows if r["backend"] == b and r["model"] == m), None)
            if match is None:
                continue
            c = _color(b)
            c_light = _lighten(c, factor=0.65)
            units.append((c, "white", None, match["peak_rss_mb"] / 1024, "rss"))
            if match.get("peak_igpu_mb") is not None:
                units.append((c_light, c, "//", match["peak_igpu_mb"] / 1024, "igpu"))

        # center the row of bars under the tick
        total_w = len(units) * bar_w + max(0, len(units) - 1) * gap
        start_x = j - total_w / 2 + bar_w / 2
        for k, (face, edge, hatch, h, _kind) in enumerate(units):
            x = start_x + k * (bar_w + gap)
            ax.bar(x, h, width=bar_w, color=face, edgecolor=edge,
                   linewidth=0.8 if hatch is None else 1.0,
                   hatch=hatch, zorder=3)
            max_y = max(max_y, h)
            if annotate:
                ax.text(x, h, f"{h:.1f}", ha="center", va="bottom",
                        fontsize=8, zorder=4)

    # legend, deliberately constructed: backends on the upper-left in the
    # same left-to-right order they appear in the chart (so the eye doesn't
    # have to translate between legend and bars); metric legend on the
    # upper-right where the openvino bars leave empty headroom.
    backend_handles = [
        mpatches.Patch(facecolor=_color(b), edgecolor="white", label=b)
        for b in reversed(backends)
    ]
    metric_handles = [
        mpatches.Patch(facecolor="#bbbbbb", edgecolor="white", label="RSS (process tree)"),
        mpatches.Patch(facecolor="#e8e8e8", edgecolor="#666666", hatch="//",
                       label="iGPU resident (fdinfo)"),
    ]
    leg1 = ax.legend(handles=backend_handles, title="backend",
                     loc="upper left", fontsize=9, title_fontsize=9, frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=metric_handles, title="metric",
              loc="upper right", fontsize=9, title_fontsize=9, frameon=True)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(
        [f"{m}\n({next(r['params_b'] for r in rows if r['model']==m):.1f}B params)"
         for m in models],
        rotation=0, fontsize=10,
    )
    ax.set_ylabel("Peak memory (GB)", fontsize=11)
    ax.set_ylim(0, max_y * 1.18)         # headroom for value labels
    if title is not None:
        ax.set_title(title, fontsize=13, pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # caveat as a figure-level annotation so it stays put in screenshots
    fig = ax.get_figure()
    fig.text(0.5, 0.01,
             "RSS (/proc/<pid>/status VmRSS, process tree) and iGPU resident (drm-resident-* in fdinfo) "
             "overlap on shared-memory iGPUs — do not sum.",
             ha="center", va="bottom", fontsize=8, color="#555555", style="italic")
    return ax


def memory_timeseries(
    probes: dict[str, "list"],   # label -> list[Sample]
    *,
    ax=None,
    title: str = "Server-process RSS over time",
):
    """Line chart of RSS(t) for one or more probes (one per backend/run)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    for label, samples in probes.items():
        ts = [s.t for s in samples if s.rss_mb is not None]
        ys = [s.rss_mb / 1024 for s in samples if s.rss_mb is not None]
        ax.plot(ts, ys, label=label, color=_color(label.split()[0]))
    ax.set_xlabel("time since probe start (s)")
    ax.set_ylabel("server RSS (GB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(linestyle=":", alpha=0.5)
    return ax


# ── NPU-vs-shared (notebook 03) ──────────────────────────────────────────────
# These take RunResult-like objects (bench.load.RunResult) duck-typed via the
# methods they expose — plotting.py stays decoupled from load.py.

def interference_timeline(runs, *, bin_s: float = 0.5, sharey: bool = True,
                          figsize=(11, 6)):
    """THE headline chart. One stacked panel per run, shared time axis.

    Each panel: Gemma's instantaneous decode throughput (tok/s, binned) as a
    line, with every concurrent transcription drawn as a shaded band. In the
    split config (Whisper on NPU) the line is flat through the bands; in the
    shared config (Whisper on the iGPU) it sags inside each band — that's the
    LLM stalling while the GPU services speech. The contrast is the deliverable.
    """
    import matplotlib.patches as mpatches

    runs = list(runs)
    fig, axes = plt.subplots(len(runs), 1, figsize=figsize,
                             sharex=True, sharey=sharey, squeeze=False)
    axes = axes[:, 0]

    ymax = 0.0
    for ax, run in zip(axes, runs):
        c = _config_color(run.label)
        t, tps = run.instantaneous_llm_tps(bin_s=bin_s)
        if t:
            ax.plot(t, tps, color=c, lw=1.6, zorder=3)
            ymax = max(ymax, max(tps))
        for (s, e) in run.stt_spans():
            ax.axvspan(s, e, color="#888888", alpha=0.18, lw=0, zorder=1)
        ax.set_title(run.label, fontsize=11, loc="left")
        ax.set_ylabel("LLM tok/s")
        ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if sharey and ymax > 0:
        for ax in axes:
            ax.set_ylim(0, ymax * 1.12)
    axes[-1].set_xlabel("wall-clock time (s)")

    band = mpatches.Patch(facecolor="#888888", alpha=0.18,
                          label="transcription in flight")
    axes[0].legend(handles=[band], loc="upper right", fontsize=9, frameon=True)
    fig.suptitle("LLM decode throughput under concurrent STT load",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig, axes


def latency_cdf(series: dict, *, ax=None, xlabel: str = "latency (s)",
                title: Optional[str] = None, mark=(95, 99), logx: bool = False):
    """Empirical CDF per config, with p95/p99 markers.

    `series`: label -> list of per-request values (TPOT, STT latency, RTF…).
    The story is the tail: in the shared config the curve leans right and the
    p95/p99 ticks jump out past the split config's.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    for label, xs in series.items():
        xs = [x for x in xs if x is not None]
        if not xs:
            continue
        c = _config_color(label)
        s = sorted(xs)
        y = [(i + 1) / len(s) for i in range(len(s))]
        ax.step(s, y, where="post", color=c, lw=1.8, label=label, zorder=3)
        for p in mark:
            import numpy as np
            v = float(np.percentile(s, p))
            ax.axvline(v, color=c, ls=":", lw=1.0, alpha=0.7, zorder=2)
            ax.annotate(f"p{p}", xy=(v, 0.04), fontsize=8, color=c,
                        rotation=90, va="bottom", ha="right")

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cumulative fraction of requests")
    ax.set_ylim(0, 1.02)
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(linestyle=":", alpha=0.5, zorder=0)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def metric_bars(rows: list[dict], *, metric: str, ylabel: str, ax=None,
                title: Optional[str] = None, annotate: bool = True):
    """One bar per config for a single metric (`rows` = list of summary dicts).

    Used for the throughput panels: e.g. metric='llm_tokens_per_s' and
    metric='audio_s_per_s'. Bars are config-colored so the split config reads
    as the same blue across every chart in the notebook.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    labels = [r["label"] for r in rows]
    vals = [r.get(metric, 0.0) for r in rows]
    colors = [_config_color(l) for l in labels]
    xs = range(len(rows))
    ax.bar(xs, vals, color=colors, width=0.6, zorder=3)
    if annotate:
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9, zorder=4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    if vals:
        ax.set_ylim(0, max(vals) * 1.18)
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


# ── B60 single-node characterization (notebook 04) ───────────────────────────

def sweep_curves(levels, *, knee_concurrency=None, peak_concurrency=None,
                 saturated=False, node_label=None, figsize=(12, 4.8)):
    """THE notebook-04 chart: two panels sharing the concurrency x-axis.

    Left — aggregate decode tok/s vs in-flight requests. This is the capacity
    curve; the knee (marked if `knee_concurrency` given) is this B60's serving
    capacity for the model. If `peak_concurrency` differs from the knee it's
    marked too, and when `saturated` is True a caption flags that throughput
    fell past the peak (the GPU is over-subscribed beyond that point).
    Right — TTFT p50/p95 vs concurrency: latency climbs as the queue deepens.
    If the backend never streamed (every level's TTFT is None — see client.call's
    SSE fallback) the right panel shows a placeholder instead, and the left
    throughput curve stands on its own.

    `levels`: list of per-level summary dicts (bench.sweep.LevelResult.summary()),
    each with at least `concurrency` and `agg_tokens_per_s`.
    """
    levels = sorted(levels, key=lambda s: s["concurrency"])
    cs = [s["concurrency"] for s in levels]
    tp = [s["agg_tokens_per_s"] for s in levels]
    c = B60_COLOR

    fig, (axl, axr) = plt.subplots(1, 2, figsize=figsize)

    # left: capacity curve
    axl.plot(cs, tp, "-o", color=c, lw=2.0, zorder=3)
    for x, y in zip(cs, tp):
        axl.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, color="#444")
    if knee_concurrency is not None:
        ky = next((s["agg_tokens_per_s"] for s in levels
                   if s["concurrency"] == knee_concurrency), None)
        if ky is not None:
            axl.axvline(knee_concurrency, color="#444", ls="--", lw=1.0, alpha=0.7,
                        zorder=2)
            axl.annotate(f"knee ≈ {knee_concurrency} in-flight\n{ky:.0f} tok/s",
                         (knee_concurrency, ky), textcoords="offset points",
                         xytext=(10, -30), fontsize=9, color="#444",
                         arrowprops=dict(arrowstyle="->", color="#444", lw=0.8))
    # mark the peak only when it's a *different* level than the knee, so a
    # plateau (knee == peak) doesn't draw two overlapping labels
    if peak_concurrency is not None and peak_concurrency != knee_concurrency:
        py = next((s["agg_tokens_per_s"] for s in levels
                   if s["concurrency"] == peak_concurrency), None)
        if py is not None:
            axl.annotate(f"peak {py:.0f} tok/s", (peak_concurrency, py),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=8, color=c, fontweight="bold")
    axl.set_xlabel("in-flight requests (concurrency)")
    axl.set_ylabel("aggregate decode tok/s")
    axl.set_title("Serving capacity — aggregate throughput", fontsize=11)
    axl.set_xscale("log", base=2)
    axl.set_xticks(cs); axl.set_xticklabels([str(x) for x in cs])
    axl.set_ylim(0, max(tp) * 1.22 if tp else 1)
    axl.grid(linestyle=":", alpha=0.5, zorder=0)
    axl.spines["top"].set_visible(False); axl.spines["right"].set_visible(False)
    if saturated and peak_concurrency is not None:
        axl.text(0.5, -0.22,
                 f"⚠ throughput falls past {peak_concurrency} in-flight — "
                 "over-subscribed beyond the knee",
                 transform=axl.transAxes, ha="center", va="top",
                 fontsize=8, color="#b00", style="italic")

    # right: TTFT vs concurrency, or a placeholder when the backend didn't stream
    streamed = [s for s in levels if s.get("ttft_p50") is not None]
    if streamed:
        sc = [s["concurrency"] for s in streamed]
        p50 = [s["ttft_p50"] for s in streamed]
        p95 = [s.get("ttft_p95") for s in streamed]
        axr.plot(sc, p50, "-o", color=c, lw=2.0, label="TTFT p50", zorder=3)
        if all(v is not None for v in p95):
            axr.plot(sc, p95, "--s", color=_lighten(c, 0.35), lw=1.7,
                     label="TTFT p95", zorder=3)
        axr.set_xlabel("in-flight requests (concurrency)")
        axr.set_ylabel("time to first token (s)")
        axr.set_title("Tail latency vs concurrency", fontsize=11)
        axr.set_xscale("log", base=2)
        axr.set_xticks(sc); axr.set_xticklabels([str(x) for x in sc])
        axr.legend(fontsize=9, frameon=True)
        axr.grid(linestyle=":", alpha=0.5, zorder=0)
        axr.spines["top"].set_visible(False); axr.spines["right"].set_visible(False)
    else:
        axr.text(0.5, 0.5,
                 "backend did not stream (SSE)\nTTFT / decode unavailable —\n"
                 "aggregate throughput still valid",
                 ha="center", va="center", fontsize=11, color="#999",
                 transform=axr.transAxes)
        axr.set_xticks([]); axr.set_yticks([])
        for sp in axr.spines.values():
            sp.set_visible(False)

    if node_label:
        fig.suptitle(f"Arc Pro B60 single-node characterization — {node_label}",
                     fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, (axl, axr)


# ── Two-node Ray cluster cost test (notebook 05) ─────────────────────────────

def cluster_throughput(agg, *, node_order=None, figsize=(8, 5)):
    """Stacked bars of system throughput vs per-node concurrency.

    Each bar is the cluster's total tok/s at that concurrency-per-node, split
    into per-node segments so an asymmetric node (e.g. a RAM-starved desktop
    carrying less of the load) is visible rather than averaged away. `agg` is
    bench.cluster.aggregate()'s output.
    """
    agg = sorted(agg, key=lambda r: r["concurrency_per_node"])
    xs = list(range(len(agg)))
    # stable host order = first appearance, or caller-pinned
    hosts = node_order or list(dict.fromkeys(
        h for r in agg for h in r["per_node_tokens_per_s"]))
    palette = [B60_COLOR, "#d62728", "#2ca02c", "#9467bd"]

    _, ax = plt.subplots(figsize=figsize)
    bottoms = [0.0] * len(agg)
    for i, host in enumerate(hosts):
        vals = [r["per_node_tokens_per_s"].get(host, 0.0) for r in agg]
        ax.bar(xs, vals, bottom=bottoms, width=0.6, label=host,
               color=palette[i % len(palette)], edgecolor="white", zorder=3)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, total in zip(xs, bottoms):
        ax.text(x, total, f"{total:.0f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{r['concurrency_per_node']}/node\n({r['total_in_flight']} total)"
                        for r in agg], fontsize=9)
    ax.set_xlabel("in-flight requests")
    ax.set_ylabel("aggregate decode tok/s (system)")
    ax.set_title("Two-node B60 cluster — combined throughput", fontsize=12)
    ax.set_ylim(0, max(bottoms) * 1.18 if bottoms else 1)
    ax.legend(title="node", fontsize=9, title_fontsize=9, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return ax


def cost_comparison(rows, *, figsize=(11, 4.2)):
    """Two panels — tokens/s/$ and tokens/s/W — one bar per system.

    `rows` is bench.cluster.cost_metrics() output. The B60 system is Intel blue,
    the Ada red, so the cost-effectiveness story reads at a glance: the B60 pair
    usually wins tokens/s/$ by a wide margin; tokens/s/W is the closer fight and
    the honest one to show alongside.
    """
    labels = [r["label"] for r in rows]
    colors = [B60_COLOR if "B60" in l else "#d62728" for l in labels]
    fig, (axl, axr) = plt.subplots(1, 2, figsize=figsize)

    for ax, key, title, ylab in [
        (axl, "tokens_per_s_per_usd", "Throughput per dollar", "tok/s per $"),
        (axr, "tokens_per_s_per_w", "Throughput per watt", "tok/s per W"),
    ]:
        vals = [r.get(key) or 0.0 for r in rows]
        xs = range(len(rows))
        ax.bar(xs, vals, color=colors, width=0.55, zorder=3)
        for x, v in zip(xs, vals):
            ax.text(x, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9, zorder=4)
        ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylab); ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(vals) * 1.20 if any(vals) else 1)
        ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Cost-effectiveness — 2× B60 vs 1× RTX 6000 Ada", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, (axl, axr)
