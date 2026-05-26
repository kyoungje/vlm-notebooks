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


def _color(backend: str) -> str:
    return BACKEND_COLORS.get(backend.lower(), "#7f7f7f")


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
