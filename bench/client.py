"""OpenAI-client wrapper for the OV-vs-vLLM bench.

Both backends speak the OpenAI Chat Completions API on :9000 by default,
so the bench treats them identically — no backend-specific code paths.
The operator brings up exactly one backend at a time (see ../README.md);
this module only measures.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import openai


@dataclass
class CallResult:
    prompt_tokens: int
    completion_tokens: int
    total_latency_s: float
    ttft_s: Optional[float]      # None when stream=False
    decode_tps: Optional[float]  # tokens/sec excluding TTFT, None when stream=False
    text: str
    # Absolute time.perf_counter() of every content chunk arrival, when
    # stream=True. perf_counter is process-wide monotonic, so the concurrent
    # load runner can place these on the same wall-clock axis as STT spans —
    # that's what drives the interference timeline. None for non-streaming.
    token_times: Optional[list[float]] = None
    start_perf: Optional[float] = None  # perf_counter at request submit
    end_perf: Optional[float] = None     # perf_counter at response complete


@dataclass
class TranscribeResult:
    latency_s: float
    audio_dur_s: Optional[float]  # declared/probed clip length; None if unknown
    rtf: Optional[float]          # real-time factor = latency / audio_dur (<1 = faster than realtime)
    text: str
    start_perf: float             # perf_counter at request submit
    end_perf: float               # perf_counter at response complete


def encode_image(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:image/jpeg;base64,{b64}"


def build_messages(prompt: str, image_path: Optional[str | Path] = None) -> list[dict]:
    if image_path is None:
        return [{"role": "user", "content": prompt}]
    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
            {"type": "text", "text": prompt},
        ],
    }]


def call(
    *,
    base_url: str,
    model: str,
    prompt: str,
    image_path: Optional[str | Path] = None,
    max_tokens: int = 64,
    temperature: float = 0.0,
    stream: bool = False,
    timeout_s: float = 300.0,
) -> CallResult:
    """Single blocking call; returns timing + token counts.

    stream=True also measures TTFT and decode tok/s; stream=False is
    cheaper but only gives total wall-clock latency.
    """
    client = openai.OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout_s)
    messages = build_messages(prompt, image_path)

    started = time.perf_counter()

    if not stream:
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, stream=False,
        )
        total = time.perf_counter() - started
        usage = resp.usage
        return CallResult(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_latency_s=total,
            ttft_s=None,
            decode_tps=None,
            text=resp.choices[0].message.content or "",
        )

    first_token_at: Optional[float] = None
    token_times: list[float] = []
    chunks: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0
    try:
        stream_resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, stream=True,
            stream_options={"include_usage": True},
        )
    except openai.BadRequestError as e:
        # Some backends don't implement SSE streaming for chat completions
        # (e.g. `optimum-cli serve` wrapping OVModelForVisualCausalLM
        # returns 400 "streaming not supported by this backend"). Fall back
        # to a blocking call so the bench works uniformly; TTFT and
        # decode_tps come back as None for that row.
        if "streaming" not in str(e).lower():
            raise
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature, stream=False,
        )
        total = time.perf_counter() - started
        usage = resp.usage
        return CallResult(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_latency_s=total,
            ttft_s=None,
            decode_tps=None,
            text=resp.choices[0].message.content or "",
        )
    for chunk in stream_resp:
        if chunk.choices and chunk.choices[0].delta.content:
            now = time.perf_counter()
            if first_token_at is None:
                first_token_at = now
            token_times.append(now)
            chunks.append(chunk.choices[0].delta.content)
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    ended = time.perf_counter()
    total = ended - started
    ttft = (first_token_at - started) if first_token_at else None
    decode_tps: Optional[float] = None
    if completion_tokens and ttft is not None and (total - ttft) > 0:
        decode_tps = completion_tokens / (total - ttft)

    return CallResult(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens or len("".join(chunks).split()),
        total_latency_s=total,
        ttft_s=ttft,
        decode_tps=decode_tps,
        text="".join(chunks),
        token_times=token_times,
        start_perf=started,
        end_perf=ended,
    )


def warmup(*, base_url: str, model: str, prompt: str = "Hi.", **kw) -> CallResult:
    """One throwaway call so first-call cold-compile cost doesn't pollute the bench."""
    return call(base_url=base_url, model=model, prompt=prompt, max_tokens=4, **kw)


# ── Whisper STT path ─────────────────────────────────────────────────────────
# The Whisper server (any OpenAI-audio-compatible STT server) speaks the OpenAI
# audio API: POST /v1/audio/transcriptions with a multipart `file=` field. Same client,
# different sub-API — so NPU (split) and iGPU (shared) deployments are measured
# by moving base_url/device only, exactly like the chat path.

def audio_duration_s(path: str | Path) -> Optional[float]:
    """Best-effort clip length in seconds, no heavy audio deps.

    Reads WAV headers via the stdlib `wave` module. For compressed formats
    (mp3/m4a/…) we can't decode without librosa/soundfile, so we return None
    and rely on the duration the operator declares in data/audio.json — that
    declared value is the source of truth for RTF, this probe is only a
    convenience for WAV clips.
    """
    p = Path(path)
    if p.suffix.lower() == ".wav":
        try:
            import wave
            with wave.open(str(p), "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                if rate:
                    return frames / float(rate)
        except Exception:
            return None
    return None


def transcribe(
    *,
    base_url: str,
    model: str,
    audio_path: str | Path,
    audio_dur_s: Optional[float] = None,
    language: Optional[str] = None,
    timeout_s: float = 300.0,
) -> TranscribeResult:
    """Single blocking transcription; returns wall-clock latency and RTF.

    RTF (real-time factor) = latency / audio_dur_s. <1.0 means the engine
    transcribes faster than realtime — the bar a continuous-listening STT
    path must clear even while the iGPU is busy serving the LLM. If
    `audio_dur_s` is None we try to probe it (WAV only); rtf is None when the
    duration is unknown.
    """
    client = openai.OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout_s)
    dur = audio_dur_s if audio_dur_s is not None else audio_duration_s(audio_path)

    kw: dict = {}
    if language:
        kw["language"] = language

    started = time.perf_counter()
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model=model, file=f, response_format="json", **kw,
        )
    ended = time.perf_counter()

    latency = ended - started
    text = getattr(resp, "text", None) or (resp if isinstance(resp, str) else "")
    rtf = (latency / dur) if (dur and dur > 0) else None
    return TranscribeResult(
        latency_s=latency,
        audio_dur_s=dur,
        rtf=rtf,
        text=text,
        start_perf=started,
        end_perf=ended,
    )


def warmup_stt(*, base_url: str, model: str, audio_path: str | Path, **kw) -> TranscribeResult:
    """Throwaway transcription so NPU cold-compile (~minutes on first call)
    doesn't land inside the measured window."""
    return transcribe(base_url=base_url, model=model, audio_path=audio_path, **kw)
