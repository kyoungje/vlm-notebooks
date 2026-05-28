# Test Report — Panther Lake VLM memory footprint

**Date**: 2026-05-26
**Hardware**: Intel Core Ultra Series 3 (Panther Lake), iGPU on `xe` driver
**Goal**: Compare server-process memory usage of OpenVINO vs vLLM serving
the same model class on Panther Lake. Headline metric: peak system RSS
and peak iGPU resident memory.

## Method

- Bench treats the inference server as an external HTTP service on `:9000`
  (OpenAI-compatible). Operator brings up one backend at a time.
- `bench/memprobe.py` samples every 500 ms:
  - **RSS**: `VmRSS` from `/proc/<pid>/status` for the server process tree.
  - **iGPU resident**: `drm-resident-*` from `/proc/<pid>/fdinfo/<fd>`
    across every DRM fd in the process tree. Works without root on xe and i915.
- Three short text prompts (max 32 / 128 / 256 tokens) are sent
  non-streaming; the probe records peak values over the run.
- Each row in `data/memory_footprint.csv` carries provenance: timestamp,
  backend, backend version, model, params (B), quantization, device.

## What was run

| Backend | Model            | Quant | Device | RSS (GB) | iGPU (GB) |
|---------|------------------|-------|--------|---------:|----------:|
| vLLM 0.20.0 | gemma-4-E4B-it       | FP16  | iGPU   | 8.7      | 21.1      |
| OpenVINO    | gemma-4-E4B-it-int4-ov | INT4 | iGPU   | *TBD*    | *TBD*     |

Only the vLLM row is captured so far. The OpenVINO row needs to be
collected with `optimum-cli serve` (or equivalent) loading the INT4 OV IR
of the same base model.

## Observations

1. **vLLM holds ~21 GB of iGPU memory** for a 4B parameter model in FP16.
   This is the KV-cache pre-allocation behavior — vLLM reserves a large
   block at startup regardless of current load. Matches the original
   concern from the Intel engineer.
2. **iGPU > RSS for vLLM** (21.1 vs 8.7 GB). On a shared-memory iGPU the
   two metrics measure overlapping memory through different lenses;
   vLLM's DRM-mapped buffers don't fully reflect in VmRSS. They should
   not be summed in the writeup.
3. Streaming chat completions returned HTTP 400 against `optimum-cli serve`
   (`streaming not supported by this backend`). `bench.client.call` falls
   back to a blocking call automatically. Latency comparison is out of
   scope for this report — see "Not tested" below.

## Not tested

- **OpenVINO row** — pending. Once captured, the chart in
  `notebooks/02_memory_footprint.ipynb` will refresh automatically.
- **Latency / throughput** — intentionally out of scope. `optimum-cli serve`
  doesn't support SSE streaming, so a fair TTFT-vs-decode comparison
  against vLLM isn't possible without changing the OV serving path.
- **NPU offload of the LLM** — Gemma4 is not yet supported by
  `openvino_genai` or OVMS, so the *LLM* NPU path is not exercised, and
  vLLM has no NPU backend at all. (Whisper on NPU *is* exercised — see the
  concurrency experiment below — via `openvino_genai.WhisperPipeline`.)
- **Vision prompts** — `data/prompts.json` defines vision rows but no
  images have been committed; bench skips them.

## Experiment 2 — Whisper on NPU vs sharing the iGPU (concurrency)

**Status: harness built, results not yet captured** (needs both servers on
the Panther Lake box; numbers below are placeholders).

**Goal**: show whether offloading Whisper to the NPU beats running Whisper
and Gemma together on the iGPU, for an always-on voice workload (mic
streaming while the LLM generates).

**Method** (`notebooks/03_npu_vs_shared.ipynb`, `bench/load.py`):

- Two configs, identical except the Whisper device: **A (split)** = Gemma
  iGPU + Whisper NPU; **B (shared)** = Gemma iGPU + Whisper iGPU. Only
  `WHISPER_DEVICE` (NPU vs GPU) changes — same vLLM, same Whisper server.
- Drive both paths concurrently for a fixed window (default 120 s): LLM
  workers stream chat completions back-to-back; STT workers transcribe a
  fixed clip back-to-back. `STT_WORKERS` sweeps simultaneous audio streams.
- Per-token timestamps on the LLM stream + per-request STT spans, both on
  one perf_counter clock, plus RSS/iGPU sampling of both server PIDs.

**Metrics / charts**: interference timeline (instantaneous LLM tok/s with
transcriptions shaded — the headline), tail-latency CDFs (LLM decode tok/s,
STT latency, STT RTF), combined throughput (LLM tok/s + audio-s/s), and the
Whisper process's peak iGPU-resident memory.

| Metric (concurrent load) | A — split (NPU) | B — shared (iGPU) |
|---|---:|---:|
| LLM decode tok/s, p50 | *TBD* | *TBD* |
| STT latency p95 / p99 (s) | *TBD* | *TBD* |
| STT RTF p50 | *TBD* | *TBD* |
| Combined throughput (LLM tok/s) | *TBD* | *TBD* |
| Whisper peak iGPU (MB) | ~0 (on NPU) | *TBD* |

**Hypothesis**: A holds flat LLM throughput and sub-1.0 STT RTF through
concurrent load while B's both-paths latencies develop a long p95/p99 tail;
A frees the iGPU budget B spends on Whisper weights.

**Caveats** (must accompany any result): NPU and iGPU share DRAM
*bandwidth*, so isolation is on compute only; NPU device memory isn't in
fdinfo, so A's Whisper-iGPU bar is 0 by construction (RAM use shows in RSS);
Panther Lake throttles — cool down between passes.

## How to reproduce

```bash
source .venv/bin/activate
# Experiment 1 — bring up ONE backend on :9000, then:
jupyter lab notebooks/02_memory_footprint.ipynb
# Experiment 2 — bring up vLLM (:9000) AND Whisper (:9010), then:
jupyter lab notebooks/03_npu_vs_shared.ipynb   # run once per WHISPER_DEVICE
```

Edit `BACKEND`, `MODEL`, `QUANTIZATION`, `DEVICE` in the per-run cell,
run end-to-end, repeat for each (backend, model). The bar chart is
cumulative across runs.

## Open questions for the team

- For the OV row, should we measure against the **FP16 HuggingFace
  weights** (apples-to-apples with vLLM's FP16) or the **INT4 OV IR**
  (apples-to-apples with how OV is actually deployed)? Probably both,
  as separate rows.
- Is there a second model size we want on the chart to show the *trend*,
  not just one data point? E.g. a 0.5B or 1B model alongside the 4B.
