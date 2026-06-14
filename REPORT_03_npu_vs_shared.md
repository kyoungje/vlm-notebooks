# Test Report — Whisper on NPU vs sharing the iGPU (notebook 03)

**Date**: 2026-05-28
**Hardware**: Intel Core Ultra Series 3 (Panther Lake) — iGPU on `xe`, NPU on `/dev/accel`
**Notebook**: `notebooks/03_npu_vs_shared.ipynb` · **Harness**: `bench/load.py`, `bench/client.py`
**Data**: `data/concurrency/split.json`, `data/concurrency/shared.json`

## TL;DR

For an always-on voice workload (mic streaming while Gemma is generating), the
two configs trade off rather than one dominating:

- **Offloading Whisper to the NPU (split) made the LLM ~38–41% faster** under
  concurrent load (8.67 vs 6.27 decode tok/s p50). This is the headline win.
- **But the iGPU is a faster Whisper engine than the NPU.** The shared config
  transcribed each clip in **0.38 s vs 0.68 s** and completed **303 vs 190**
  transcriptions in the window. NPU offload protects the LLM at the cost of
  slower-per-clip STT.
- The hypothesized latency-tail blowup in the shared config **did not appear**
  at this load (1 LLM + 1 STT worker). Both latency distributions are tight.
- ⚠️ **The split-config memory numbers are not trustworthy** (probe artifact —
  see Caveats). Do not quote the 23 GB Whisper-iGPU figure.

## Method

- Two configs, identical except where Whisper runs. Same vLLM-XPU serving Gemma
  on the iGPU (`:9000`), same Whisper server (`:9010`); only `WHISPER_DEVICE`
  changes — **split** = Whisper on NPU, **shared** = Whisper on iGPU.
- Both paths driven **concurrently** for a fixed 120 s window: 1 LLM worker
  streaming chat completions back-to-back, 1 STT worker transcribing a fixed
  Korean clip back-to-back (`STT_GAP_S=0`, always-on).
- Per-token LLM timestamps and per-request STT spans are recorded on one
  `perf_counter` clock; `bench/memprobe.py` samples RSS + iGPU-resident memory
  of both server PIDs every 500 ms.

Run config (from `meta`): `llm_workers=1`, `stt_workers=1`, `stt_gap_s=0`,
`llm_max_tokens=null` (per-prompt default, 128). **0 failed requests in either
run.**

## Results

| Metric (120 s concurrent window) | A — split (NPU) | B — shared (iGPU) | Winner |
|---|---:|---:|:--|
| LLM decode tok/s — p50 | **8.67** | 6.27 | split +38% |
| LLM decode tok/s — p95 | 9.96 | 7.39 | split |
| LLM tokens generated (aggregate tok/s) | 1075 (8.96/s) | 762 (6.35/s) | split +41% |
| LLM requests completed | 11 | 8 | split |
| STT latency — p50 (s) | 0.678 | **0.384** | shared −43% |
| STT latency — p95 / p99 (s) | 0.686 / 0.689 | 0.392 / 0.414 | shared |
| STT transcriptions completed | 190 (1.58/s) | **303 (2.52/s)** | shared +59% |
| STT RTF (real-time factor) | n/a | n/a | not computed¹ |
| Whisper peak iGPU resident | 23034 MB ⚠️ | 2038 MB | unreliable² |
| Whisper peak RSS | 21478 MB ⚠️ | 619 MB | unreliable² |
| Gemma peak iGPU / RSS | 22194 / 9797 MB | 22128 / 8825 MB | ~equal |

¹ RTF needs `duration_s` for the clip; `data/audio.json` still has
`duration_s: null` for `korean_10s`, so RTF charts were skipped. Latency in
seconds (above) is unaffected.
² See Caveats — the split-config Whisper figures are a probe artifact.

## Interpretation

1. **NPU offload buys LLM throughput.** Moving Whisper off the iGPU gave Gemma
   ~38% more decode tok/s and ~41% more total tokens over the window. When the
   LLM's responsiveness is the priority, split wins.

2. **The NPU is the slower Whisper engine.** This is the non-obvious result.
   Per-clip STT latency nearly doubled on the NPU (0.68 vs 0.38 s) and the
   shared config cleared 59% more transcriptions. So the choice is genuinely a
   trade, not a free lunch: split optimizes the LLM, shared optimizes STT
   latency/throughput.

3. **No contention tail at this load.** The notebook predicted the shared
   config would grow a long p95/p99 STT tail under overlap. It didn't — both
   distributions are within ~10% of their medians. At 1+1 worker, the iGPU
   time-slices cleanly and neither path starves the other. This needs a heavier
   load to stress (see Next steps).

## Caveats (must accompany any external use of these numbers)

- **⚠️ Split-config memory is a measurement artifact.** The notebook's own logic
  says Whisper-on-NPU should hold ~0 iGPU-resident memory (no DRM fd on
  `/dev/accel`). Instead the probe reported 23034 MB iGPU / 21478 MB RSS for the
  Whisper process — nearly identical to Gemma's own 22194 MB. That strongly
  suggests `find_server_pid(:9010)` resolved into a process tree that overlaps
  vLLM's iGPU buffers (shared parent / container), so the split row double-counts
  Gemma's memory. **Only the shared-config Whisper iGPU figure (~2 GB) is
  credible.** Re-measure before quoting any split memory number.
- **Single load point.** One 120 s window at 1 LLM + 1 STT worker each. No
  worker sweep, no repeats — treat as a directional first result, not a
  characterization.
- **Shared DRAM bandwidth.** NPU and iGPU are separate compute engines but share
  one memory controller; the isolation shown here is on *compute* only.
- **Thermal.** Panther Lake throttles. The two passes were not confirmed to be
  thermally matched, so part of the LLM delta could be heat soak — alternate run
  order on repeats.

## Next steps

1. **Fix the Whisper PID resolution** in the split run, then re-capture memory so
   we can actually show the "iGPU budget freed for KV cache" story (the whole
   point of the memory chart).
2. **Sweep `STT_WORKERS` 1 → 2 → 4** to find the load where the shared config's
   latency tail finally opens up — the hypothesis is untested until it does.
3. **Set `duration_s` in `data/audio.json`** (`ffprobe -show_entries
   format=duration data/audio/korean_audio.mp3`) so RTF is reported — the
   sub-1.0 RTF "keeps up with realtime" bar is what an always-on product cares
   about.
4. Repeat each config 3× with alternating order to separate config effect from
   thermal drift.

## Reproduce

```bash
source .venv/bin/activate
# LLM/VLM on iGPU (vLLM-XPU), same for both passes — any OpenAI-compatible backend:
docker run --rm -p 9000:8000 --device /dev/dri <intel-vllm-xpu-image> --model <model> --port 8000
# Whisper, ONE device per pass:
#   split:  WHISPER_DEVICE=NPU  -> :9010
#   shared: WHISPER_DEVICE=GPU  -> :9010
jupyter lab notebooks/03_npu_vs_shared.ipynb            # set CONFIG_LABEL, run end-to-end, flip device, run again
```
