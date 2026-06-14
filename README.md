# vlm-notebooks — benchmarking small VLM/LLM serving on Intel hardware (Panther Lake iGPU/NPU · Arc Pro B60)

Memory footprint, NPU-vs-iGPU concurrency, single-card serving throughput, and a
two-node tokens/s/$ · tokens/s/W cost comparison — for small VLMs/LLMs on Intel
silicon, measured against an OpenAI-compatible backend you bring up yourself.

Five Jupyter notebooks + a tiny Python harness benchmarking small VLMs/LLMs
on Intel hardware. Notebooks 01–03 ask **how much memory each backend's server
process holds** on an Intel Panther Lake laptop (footprint vs model size, not
raw throughput). Notebook 04 is a separate experiment on a different box: the
serving **throughput** of one discrete Arc Pro B60 in a desktop. Notebook 05 is
Phase 2 — two B60 desktops driven together via Ray — producing the headline
**tokens/s/$** and **tokens/s/W** for the two-B60-vs-RTX-6000-Ada comparison.

## Layout

```
bench/                # reusable Python (no notebook code here)
  client.py           # OpenAI client: chat completions (+ per-token timing) and Whisper STT
  load.py             # concurrent LLM+STT load runner + stats (notebook 03)
  sweep.py            # single-node LLM concurrency-sweep runner + stats (notebook 04)
  cluster.py          # Ray distributed load driver + aggregation (notebook 05)
  memprobe.py         # background RSS+iGPU sampler tied to server PID
  plotting.py         # matplotlib helpers used by the notebooks
notebooks/
  01_env.ipynb              # validate the backend on :9000 and the probes
  02_memory_footprint.ipynb # the headline memory bar chart
  03_npu_vs_shared.ipynb    # Whisper on NPU vs sharing the iGPU (concurrency)
  04_b60_throughput.ipynb   # Arc Pro B60 single-node throughput / serving capacity
  05_two_node_cluster.ipynb # two B60s via Ray: aggregate throughput + tokens/s/$ vs Ada
data/
  prompts.json          # reproducible LLM prompts (text + vision)
  audio.json            # STT clips + declared durations (notebook 03)
  audio/                # the audio clips themselves
  memory_footprint.csv  # appended by notebook 02 across runs
  concurrency/          # per-config runs saved by notebook 03
  b60/                  # per-node throughput runs (04) + cluster runs (05)
```

Notebook 02's deliverable is intentionally scoped to **memory footprint**,
not latency. A latency comparison *across backends* would need every
backend to support SSE streaming the same way (so TTFT vs decode can be
separated); `optimum-cli serve` doesn't yet, so a head-to-head latency
chart would be apples-to-oranges. The memory metric, by contrast, is
read at the OS level and is identical across backends.

Notebook 03 *does* measure latency, but it sidesteps that apples-to-oranges
problem: it holds both servers fixed and moves only **where Whisper runs**
(NPU vs iGPU). Same vLLM, same Whisper server, same model — only the
`WHISPER_DEVICE` device changes between the two passes, so the latency delta
is attributable to compute contention, not to backend differences.

## Notebook 03 — Whisper on NPU vs sharing the iGPU

The question behind this one: on Panther Lake, is it better to put Whisper on
the **NPU** (leaving Gemma the iGPU to itself) or to run **both on the iGPU**?
It depends on whether the two workloads overlap, so the notebook targets the
*always-on* case — mic streaming while the model is mid-answer — and drives
both paths concurrently for a fixed window.

Unlike 01/02, this needs **two** servers up at once:

```bash
# LLM/VLM on the iGPU (vLLM-XPU) — same for both configs, :9000
#   bring up any OpenAI-compatible backend (see "How the bench is structured")
docker run --rm -p 9000:8000 --device /dev/dri <intel-vllm-xpu-image> --model <model> --port 8000

# Whisper STT — ONE device per pass, :9010 (any OpenAI-audio-compatible server)
#   Config A (split):  WHISPER_DEVICE=NPU   -> /dev/accel
#   Config B (shared): WHISPER_DEVICE=GPU   -> /dev/dri (the iGPU vLLM is on)
```

Set `CONFIG_LABEL` in the notebook to match the device you brought up, run it
end to end (it warms both paths first — the first NPU call compiles the graph
and must not land in the measured window), then flip `WHISPER_DEVICE`, set the
other label, and run again. The comparison cells load both saved runs from
`data/concurrency/` and draw: the interference timeline (LLM tok/s with
transcriptions shaded — the headline), tail-latency CDFs, combined throughput,
and the Whisper process's iGPU footprint. RTF needs `duration_s` declared in
`data/audio.json`; absolute STT-latency charts work without it.

Caveats worth stating in any writeup: NPU and iGPU share DRAM **bandwidth**
(the isolation shown is on compute, not bandwidth); NPU device memory isn't
exposed via fdinfo, so the split config's Whisper-iGPU bar reads 0 by
construction; and Panther Lake throttles, so cool down between passes.

## Notebook 04 — Arc Pro B60 single-node throughput

A separate experiment on a different machine: one discrete **Arc Pro B60**
(24 GB, Battlemage, `xe` driver) in an ordinary desktop. It is **standalone** —
it does not import or run 01/02/03 and has **no NPU path** (these desktops have
no NPU). It produces the per-node numbers the *two-B60-vs-RTX-6000-Ada* cost
comparison is built from, measuring three things:

1. **Validation** — a backend is up on `:9000` and the B60's memory is readable
   via `xe` fdinfo.
2. **Single-stream baseline** — uncontended TTFT, decode tok/s, peak footprint.
3. **Concurrency sweep** — aggregate throughput and tail latency as in-flight
   requests climb; the level where aggregate tok/s flattens (the *knee*) is this
   B60's serving capacity for the model.

Prerequisites: the B60 visible to the host (`/dev/dri/renderD128`, your user in
the `render` group, kernel 6.12+ with the `xe` driver bound), Docker able to
pass `--device /dev/dri`, and the repo's Python env. Bring up one backend on
`:9000` (vLLM-XPU recommended for the cost story — confirm the current Intel XPU
image tag, the patched "llm-scaler" build is Arc-oriented — or OpenVINO Model
Server); start with `Qwen2.5-VL-3B-Instruct` to prove the path, then swap in the
real model. Edit the `EDIT PER RUN` cell (`NODE`, `MODEL`, `QUANTIZATION`,
`CONCURRENCY_LEVELS`, `WINDOW_S`, …) and run top to bottom. Section 1's asserts
must pass first. The run is saved to
`data/b60/<NODE>_<MODEL>_<timestamp>.json` — the input to the Phase-2 two-node
cost chart (one model instance per B60 behind a load balancer, with
**tokens/s/$** and **tokens/s/W** as the headline metrics vs the single Ada).

Caveats for the writeup: the B60 is Gen5 x8 and a desktop slot may down-train it
(section 1 reads `current_link_speed`/`width` from `/sys`; capture
`sudo lspci -vv | grep -A2 LnkSta` for the authoritative figure); log board
watts per level if you can read them, for tokens/s/W; if the backend doesn't
stream, TTFT/decode read `None` and the latency panel shows a placeholder while
throughput still computes; and Arc XPU serving is younger than the CUDA path, so
note image/driver versions and frame any gap as "already cost-competitive on an
immature stack." The Ada wins single-request latency — expected; the B60 story
is throughput-per-dollar.

## Notebook 05 — Two-node B60 cluster (Ray, Phase 2)

Once both desktops pass notebook 04, this drives **both at once** and turns the
per-node numbers into the headline **tokens/s/$** and **tokens/s/W** vs a single
RTX 6000 Ada. It uses **Option B**: vLLM runs the models (one engine per B60,
pinned to its card), and **Ray runs the experiment** — it places one load driver
on each node (so requests originate locally), launches them together, and sums
the results into system throughput. Ray never touches the GPU: it can't see
Intel XPUs as resources and here doesn't need to — vLLM does the card binding,
and the `xpu` resource below is just a placement label.

Bring-up (run the notebook on the head node):

```bash
# 1. one vLLM engine per node on :8000, pinned to the local B60 — the exact
#    command from notebook 04, on BOTH desktops:
vllm serve <model> --max-num-seqs <KV ceiling> ...     # see notebook 04

# 2. a Ray cluster across both desktops, each tagged with a custom xpu resource.
#    Put the head on the box with more RAM.
#    Same Ray + Python version on both, same LAN, GCS port 6379 reachable.
ray start --head --resources='{"xpu": 1}'                       # desktop1 (head)
ray start --address='<head-ip>:6379' --resources='{"xpu": 1}'   # desktop2 (worker)
```

The notebook connects with `ray.init(address='auto', runtime_env={'working_dir': ...})`
(ships this repo to every worker so the driver code + `data/prompts.json`
match), sweeps `CONCURRENCY_LEVELS` **per node**, and saves the combined run to
`data/b60/cluster_<timestamp>.json`. `CONCURRENCY_LEVELS` must stay at/under each
engine's `--max-num-seqs` KV ceiling — past it requests just queue.

For the cost panel, set `ADA_TOKENS_PER_S` to a figure you **measured the same
way** (run notebook 04 against the Ada at the same model, quantization, context,
and `max_tokens`) — a spec-sheet number isn't apples-to-apples. The combined
throughput is the sum of each node's capacity (the nodes are independent —
separate cards and hosts, no shared endpoint), so an asymmetric node shows up as
a shorter segment in the stacked chart rather than being hidden in an average.
There is no single OpenAI endpoint in this topology; one auto-load-balanced
endpoint is Option A (Ray Serve), a separate setup.

## How the bench is structured

The harness does **not** spawn backend servers. OpenVINO, vLLM, and
llama.cpp have incompatible Python venvs and Intel oneAPI dependencies;
running them in-process from a notebook is fragile. Instead the operator
brings ONE backend up on `:9000` ahead of time. Any container or local
install that exposes the OpenAI Chat Completions API on
`127.0.0.1:9000` works — pick whatever Intel-supported image you prefer.
Sketch commands (substitute your model path / image tag):

```bash
# OpenVINO Model Server (OpenAI-compatible endpoint)
docker run --rm -p 9000:9000 \
    -v "$PWD/models:/models" \
    --device /dev/dri \
    openvino/model_server:latest \
    --model_path /models/Qwen2.5-VL-3B-Instruct \
    --rest_port 9000

# vLLM with Intel XPU support
docker run --rm -p 9000:8000 \
    --device /dev/dri \
    intel/vllm-xpu:latest \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --port 8000

# llama.cpp server (SYCL build, OpenAI-compatible /v1)
./llama-server -m models/qwen2.5-3b.Q4_K_M.gguf --host 0.0.0.0 --port 9000
```

The notebooks then:

1. Find the server PID listening on `:9000` (`ss -ltnp` / `/proc`).
2. Start a background sampler reading `VmRSS` from `/proc/<pid>/status`
   (and from every descendant) plus optional `intel_gpu_top -J` for iGPU
   memory.
3. Send a handful of chat completions through the OpenAI client.
4. Record peak RSS / iGPU into a CSV.
5. Stop that backend, bring up the next one, repeat.

The bar chart in `02_memory_footprint.ipynb` is cumulative — re-run the
plot cell after each capture to refresh it across all backends/models.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Run `01_env.ipynb` first on a fresh Panther Lake box. Then `02` and `03` once
per (backend, model) you want to compare. `04_b60_throughput.ipynb` is a
standalone experiment on the Arc Pro B60 desktop — run it on its own; it needs
the same env and a backend on `:9000` but nothing from 01–03.
`05_two_node_cluster.ipynb` is Phase 2 — run it on the Ray head node once both
B60 desktops pass 04 (needs the `ray[default]` dependency on every node).

## Notes / caveats

- iGPU memory is read from `/proc/<pid>/fdinfo/<fd>` (the `drm-resident-*`
  keys documented at https://docs.kernel.org/gpu/drm-usage-stats.html).
  This works on both `xe` (Panther Lake / Lunar Lake / Battlemage) and
  `i915`, and needs no root or capabilities — just the ability to read
  your own server process's fdinfo. We deliberately do not depend on
  `intel_gpu_top`, which only speaks to `i915`.
- RSS is the server **process tree**, so for vLLM that includes its
  Ray/worker subprocesses. This is intentional — what we care about is
  how much RAM the deployment holds on Panther Lake.
- The vision rows in `data/prompts.json` reference images under
  `data/images/`. Drop a couple of representative JPEGs there before
  exercising the vision path; the bench skips rows whose image is
  missing.

## License

MIT — see [LICENSE](LICENSE).
