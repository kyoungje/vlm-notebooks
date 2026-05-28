# vlm-notebooks — OpenVINO vs vLLM memory footprint on Intel Panther Lake (Core Ultar Series 3)

Three Jupyter notebooks + a tiny Python harness that benchmark **how much
memory each backend's server process holds** for small VLMs/LLMs on an
Intel Panther Lake laptop.
The headline question is RAM/iGPU footprint vs model size, not raw throughput.

## Layout

```
bench/                # reusable Python (no notebook code here)
  client.py           # OpenAI client: chat completions (+ per-token timing) and Whisper STT
  load.py             # concurrent LLM+STT load runner + stats (notebook 03)
  memprobe.py         # background RSS+iGPU sampler tied to server PID
  plotting.py         # matplotlib helpers used by the notebooks
notebooks/
  01_env.ipynb              # validate the backend on :9000 and the probes
  02_memory_footprint.ipynb # the headline memory bar chart
  03_npu_vs_shared.ipynb    # Whisper on NPU vs sharing the iGPU (concurrency)
data/
  prompts.json          # reproducible LLM prompts (text + vision)
  audio.json            # STT clips + declared durations (notebook 03)
  audio/                # the audio clips themselves
  memory_footprint.csv  # appended by notebook 02 across runs
  concurrency/          # per-config runs saved by notebook 03
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
# Gemma on the iGPU (vLLM-XPU) — same for both configs, :9000
cd movensys_vlm/docker && ./vllm-intel-run.sh

# Whisper — ONE device per pass, :9010
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

Run `01_env.ipynb` first on a fresh box. Then `02` and `03` once per
(backend, model) you want to compare.

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
