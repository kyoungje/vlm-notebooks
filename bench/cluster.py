"""Ray control + measurement plane for the two-node B60 cost test (notebook 05).

This is **Option B** of the Phase-2 plan: vLLM runs the models — one engine per
B60, launched by the operator and pinned to its card — and Ray runs the
*experiment*. Ray places one load driver on each node so requests originate
**locally** (no cross-network client skew), starts every driver together so the
cards are busy at the same time, and collects each node's results centrally so
we can sum them into the system-level tokens/s the cost comparison needs.

Ray never touches the GPU here: it can't see Intel XPUs as schedulable
resources, and in this design it doesn't need to. Each node advertises a custom
`xpu` resource at `ray start` (see the README). A driver task requesting
`resources={'xpu': 1}` then consumes a node's whole xpu unit, so launching one
task per xpu unit lands exactly one driver per node, each driving that node's
local vLLM on `:8000`.

The per-node sweep logic is reused verbatim from bench.sweep — this module only
fans it out across the cluster and aggregates. The nodes are independent
(separate cards, separate hosts, no shared endpoint), so a node's throughput
doesn't depend on the others; the synchronized launch is what lets us claim the
combined number reflects all cards loaded at once, not one-at-a-time.
"""
from __future__ import annotations

import socket
import time
from typing import Optional

import ray

from . import sweep


def _run_node_sweep(
    *,
    base_url: str,
    model: str,
    concurrency_levels: list[int],
    window_s: float,
    max_tokens: int,
    prompts: list[dict],
    cooldown_s: float = 10.0,
) -> dict:
    """Sweep one node's local vLLM engine and return its per-level summaries +
    identity. Plain function (no Ray) so it's unit-testable; `_drive_node` wraps
    it as a remote task."""
    host = socket.gethostname()
    levels = []
    for i, c in enumerate(concurrency_levels):
        lr = sweep.sweep_level(
            concurrency=c, window_s=window_s,
            base_url=base_url, model=model, prompts=prompts, max_tokens=max_tokens,
        )
        levels.append(lr.summary())
        if cooldown_s and i < len(concurrency_levels) - 1:
            time.sleep(cooldown_s)
    # The sweep already swallows per-request errors; the baseline doesn't, so
    # guard it — a transient hiccup in 5 calls shouldn't discard the whole
    # node's levels.
    try:
        baseline = sweep.single_stream(
            base_url=base_url, model=model, prompts=prompts,
            max_tokens=max_tokens, repeats=5,
        )
    except Exception as e:  # noqa: BLE001
        baseline = {"error": repr(e)}
    return {"host": host, "levels": levels, "baseline": baseline}


# One driver per xpu unit -> one per node. num_cpus keeps it from starving the
# box (and, on the head, from fighting the notebook kernel).
_drive_node = ray.remote(num_cpus=1, resources={"xpu": 1})(_run_node_sweep)


def cluster_xpu_count() -> int:
    """How many `xpu` units the cluster advertises = how many B60 nodes are
    joined. 0 means no node was started with --resources='{"xpu":1}'."""
    return int(ray.cluster_resources().get("xpu", 0))


def cluster_sweep(
    *,
    model: str,
    concurrency_levels: list[int],
    window_s: float,
    max_tokens: int,
    prompts: list[dict],
    base_url: str = "http://127.0.0.1:8000/v1",
    cooldown_s: float = 10.0,
    n_nodes: Optional[int] = None,
) -> dict:
    """Fan the per-node sweep across the cluster, one driver per B60, launched
    together. Returns {nodes: [per-node result, ...], meta: {...}}.

    `base_url` is evaluated **on each driver's own node**, so the default
    localhost endpoint is correct as long as every node serves vLLM on :8000.
    `n_nodes` defaults to the cluster's xpu count.
    """
    n = n_nodes if n_nodes is not None else cluster_xpu_count()
    if n < 1:
        raise RuntimeError(
            "no 'xpu' resources in the cluster — start each node with "
            "ray start ... --resources='{\"xpu\": 1}' (see README)."
        )
    futs = [
        _drive_node.remote(
            base_url=base_url, model=model, concurrency_levels=concurrency_levels,
            window_s=window_s, max_tokens=max_tokens, prompts=prompts,
            cooldown_s=cooldown_s,
        )
        for _ in range(n)
    ]
    nodes = ray.get(futs)
    return {
        "nodes": nodes,
        "meta": {
            "n_nodes": n,
            "model": model,
            "concurrency_levels": concurrency_levels,
            "window_s": window_s,
            "max_tokens": max_tokens,
            "base_url": base_url,
        },
    }


def aggregate(result: dict) -> list[dict]:
    """Sum the cluster into system-level throughput per concurrency level.

    For each concurrency `c`, total tok/s = sum of every node's agg_tokens_per_s
    at `c` (data-parallel: more nodes = more aggregate work). Also surfaces the
    worst node's p95 TTFT at that level, since the system is only as responsive
    as its slowest replica. `per_node` maps host -> that node's tok/s, so a
    laggard (e.g. a RAM-starved desktop) is visible, not averaged away.
    """
    nodes = result["nodes"]
    levels = result["meta"]["concurrency_levels"]
    rows = []
    for c in levels:
        per_node = {}
        ttfts = []
        for nd in nodes:
            lv = next((x for x in nd["levels"] if x["concurrency"] == c), None)
            if lv is None:
                continue
            per_node[nd["host"]] = lv["agg_tokens_per_s"]
            if lv.get("ttft_p95") is not None:
                ttfts.append(lv["ttft_p95"])
        rows.append({
            "concurrency_per_node": c,
            "total_in_flight": c * len(nodes),
            "per_node_tokens_per_s": per_node,
            "total_tokens_per_s": round(sum(per_node.values()), 1),
            "worst_ttft_p95": max(ttfts) if ttfts else None,
        })
    return rows


def cost_metrics(
    *,
    total_tokens_per_s: float,
    n_b60: int,
    b60_price_usd: float,
    b60_board_w: float,
    ada_tokens_per_s: float,
    ada_price_usd: float,
    ada_board_w: float,
) -> list[dict]:
    """Two comparable systems with tokens/s, $/system, W/system, and the two
    headline ratios. The B60 row uses the *measured* aggregate; the Ada row uses
    whatever single-card number you measured or quoted (run notebook 04 against
    the Ada to get an apples-to-apples figure at the same model + context).
    """
    def row(label, tps, price, watts):
        return {
            "label": label,
            "tokens_per_s": round(tps, 1),
            "price_usd": price,
            "board_w": watts,
            "tokens_per_s_per_usd": round(tps / price, 4) if price else None,
            "tokens_per_s_per_w": round(tps / watts, 3) if watts else None,
        }
    return [
        row(f"{n_b60}× B60", total_tokens_per_s, n_b60 * b60_price_usd, n_b60 * b60_board_w),
        row("1× RTX 6000 Ada", ada_tokens_per_s, ada_price_usd, ada_board_w),
    ]


def save_cluster_run(result: dict, agg: list[dict], path, *, timestamp: str,
                     cost: Optional[list[dict]] = None) -> str:
    """Persist the combined run next to the per-node notebook-04 JSONs."""
    import json
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "timestamp": timestamp,
        "meta": result["meta"],
        "nodes": result["nodes"],
        "aggregate": agg,
        "cost": cost or [],
    }
    p.write_text(json.dumps(doc, indent=2))
    return str(p)
