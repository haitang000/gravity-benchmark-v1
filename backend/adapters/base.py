from __future__ import annotations

from typing import Protocol

from backend.schemas import GenerationConfig, GenerationResult, HealthResult, Message


class ModelAdapter(Protocol):
    async def generate(self, messages: list[Message], generation: GenerationConfig) -> GenerationResult: ...
    async def health_check(self) -> HealthResult: ...
    async def close(self) -> None: ...
