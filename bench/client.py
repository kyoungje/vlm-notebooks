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
            if first_token_at is None:
                first_token_at = time.perf_counter()
            chunks.append(chunk.choices[0].delta.content)
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

    total = time.perf_counter() - started
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
    )


def warmup(*, base_url: str, model: str, prompt: str = "Hi.", **kw) -> CallResult:
    """One throwaway call so first-call cold-compile cost doesn't pollute the bench."""
    return call(base_url=base_url, model=model, prompt=prompt, max_tokens=4, **kw)
