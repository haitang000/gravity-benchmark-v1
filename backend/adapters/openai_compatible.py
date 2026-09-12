from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import httpx

from backend.hardware.detect import detect_hardware
from backend.schemas import GenerationConfig, GenerationResult, HealthResult, Message


class OpenAICompatibleAdapter:
    def __init__(self, model_ref: str, config: dict[str, Any]):
        self.model_ref, self.config = model_ref, config
        self.base_url = config.get("base_url", "http://127.0.0.1:8000/v1").rstrip("/")
        self.append_path = bool(config.get("append_path", True))
        self.chat_url = f"{self.base_url}/chat/completions" if self.append_path else self.base_url
        self.models_url = self._models_url()
        self.timeout = config.get("timeout", 60)
        self.retries = config.get("retries", 1)

    def _models_url(self) -> str:
        if self.config.get("models_url"): return str(self.config["models_url"]).rstrip("/")
        if self.append_path: return f"{self.base_url}/models"
        suffix = "/chat/completions"
        if self.base_url.endswith(suffix): return f"{self.base_url[:-len(suffix)]}/models"
        return self.base_url

    def _headers(self) -> dict[str, str]:
        headers = dict(self.config.get("headers", {}))
        key = self.config.get("api_key") or os.getenv(self.config.get("api_key_env", ""))
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    async def health_check(self) -> HealthResult:
        probe = not self.append_path and self.models_url == self.base_url
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout, 10)) as client:
                response = await client.get(self.models_url, headers=self._headers())
                ok = response.is_success or (probe and response.status_code < 500)
                message = f"HTTP {response.status_code}" + (" (reachability probe)" if probe else "")
                return HealthResult(ok=ok, message=message, hardware=detect_hardware())
        except httpx.HTTPError as exc:
            return HealthResult(ok=False, message=str(exc), hardware=detect_hardware())

    async def generate(self, messages: list[Message], generation: GenerationConfig) -> GenerationResult:
        payload = {"model": self.model_ref, "messages": [m.model_dump() for m in messages], **generation.model_dump(exclude_none=True)}
        started, retries = time.perf_counter(), 0
        for attempt in range(self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(self.chat_url, json=payload, headers=self._headers())
                    response.raise_for_status()
                    data = response.json()
                elapsed = (time.perf_counter() - started) * 1000
                text = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {})
                output = usage.get("completion_tokens", max(1, len(text.split())))
                total = usage.get("total_tokens", usage.get("prompt_tokens", 0) + output)
                return GenerationResult(text=text, input_tokens=usage.get("prompt_tokens"), output_tokens=output, latency_ms=elapsed, generation_tokens_per_second=output / (elapsed / 1000) if elapsed else None, total_tokens_per_second=total / (elapsed / 1000) if elapsed else None, retries=retries, backend_metadata={"base_url": self.base_url})
            except httpx.HTTPError as exc:
                retries = attempt + 1
                if attempt == self.retries:
                    return GenerationResult(error=str(exc), latency_ms=(time.perf_counter() - started) * 1000, retries=retries)
                await asyncio.sleep(0.25 * (attempt + 1))
        raise AssertionError("unreachable")

    async def close(self) -> None:
        return None
