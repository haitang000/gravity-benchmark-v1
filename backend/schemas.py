from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class BackendType(StrEnum):
    OPENAI = "openai_compatible"
    LLAMA_CPP = "llama_cpp"
    TRANSFORMERS = "transformers"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Message(BaseModel):
    role: str
    content: str


class GenerationConfig(BaseModel):
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = Field(default=1, gt=0, le=1)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    stop: list[str] | None = None
    seed: int | None = 42
    response_format: dict[str, Any] | None = None


class GenerationResult(BaseModel):
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    ttft_ms: float | None = None
    latency_ms: float = 0
    generation_tokens_per_second: float | None = None
    total_tokens_per_second: float | None = None
    error: str | None = None
    retries: int = 0
    backend_metadata: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None


class HealthResult(BaseModel):
    ok: bool
    message: str
    hardware: dict[str, Any] = Field(default_factory=dict)


class ModelProfileIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    backend: BackendType
    model_ref: str
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_config(self):
        import re
        from urllib.parse import urlsplit
        for key in self.config:
            if key.lower() in {"password", "token", "authorization"}:
                raise ValueError("Only api_key may be entered directly; other credentials must use environment references")
        for key in self.config.get("headers", {}):
            if re.search(r"authorization|cookie|key|token|secret", key, re.I):
                raise ValueError("Sensitive headers must use header_env")
        for key in ("base_url", "models_url"):
            if key not in self.config: continue
            url = urlsplit(str(self.config[key]))
            if url.scheme not in {"http", "https"} or url.username or url.password or url.query or url.fragment:
                raise ValueError(f"{key} must be HTTP(S) without credentials, query or fragment")
        if not isinstance(self.config.get("append_path", True), bool):
            raise ValueError("append_path must be a boolean")
        if self.config.get("retries", 1) not in range(0, 6):
            raise ValueError("retries must be 0..5")
        return self


class ModelProfileOut(ModelProfileIn):
    id: int
    created_at: datetime


class RunRequest(BaseModel):
    model_ids: list[int] = Field(min_length=1)
    suite: Literal["quick", "standard", "full", "custom"] = "quick"
    languages: list[Literal["zh", "en"]] = Field(default_factory=lambda: ["zh", "en"], min_length=1)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    concurrency: int = Field(default=1, ge=1, le=16)
    timeout_seconds: int = Field(default=60, ge=1, le=600)
    seed: int = 42
    warmup: int = Field(default=1, ge=0, le=5)
    counts: dict[Literal["instruction", "math", "coding", "performance"], int] = Field(default_factory=dict)
    context_lengths: list[int] = Field(default_factory=lambda: [0], min_length=1, max_length=8)

    @model_validator(mode="after")
    def valid_counts(self):
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("Duplicate model ids")
        if any(n < 0 or n > 10000 for n in self.counts.values()):
            raise ValueError("counts must be 0..10000")
        if any(n < 0 or n > 65536 for n in self.context_lengths):
            raise ValueError("context_lengths are repeat counts in 0..65536")
        return self


class RunProgress(BaseModel):
    id: int
    status: RunStatus
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    current_task: str | None = None
    error: str | None = None
