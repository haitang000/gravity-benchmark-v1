from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from backend.schemas import GenerationConfig, GenerationResult, HealthResult, Message

OnText = Callable[[str], Awaitable[None]]


def generation_speed(output_tokens: int | None, latency_ms: float, ttft_ms: float | None = None) -> float | None:
    if not output_tokens or latency_ms <= 0: return None
    decode_ms = latency_ms - ttft_ms if ttft_ms is not None and latency_ms - ttft_ms > 0 else latency_ms
    return output_tokens / (decode_ms / 1000)


class ModelAdapter(Protocol):
    async def generate(self, messages: list[Message], generation: GenerationConfig, on_text: OnText | None = None) -> GenerationResult: ...
    async def health_check(self) -> HealthResult: ...
    async def close(self) -> None: ...
