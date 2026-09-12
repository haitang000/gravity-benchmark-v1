from __future__ import annotations

import asyncio
import time
from typing import Any

from backend.hardware.detect import detect_hardware
from backend.schemas import GenerationConfig, GenerationResult, HealthResult, Message


def _prompt(messages: list[Message]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"


class LlamaCppAdapter:
    def __init__(self, model_ref: str, config: dict[str, Any]):
        self.model_ref, self.config, self._llm = model_ref, config, None

    def _load(self) -> Any:
        if self._llm is None:
            from llama_cpp import Llama
            self._llm = Llama(model_path=self.model_ref, n_ctx=self.config.get("n_ctx", 4096), n_threads=self.config.get("n_threads"), n_gpu_layers=self.config.get("n_gpu_layers", 0), n_batch=self.config.get("n_batch", 512), verbose=False)
        return self._llm

    async def health_check(self) -> HealthResult:
        try:
            await asyncio.to_thread(self._load)
            return HealthResult(ok=True, message="llama.cpp model loaded", hardware=detect_hardware())
        except Exception as exc:
            return HealthResult(ok=False, message=f"llama.cpp unavailable: {exc}", hardware=detect_hardware())

    async def generate(self, messages: list[Message], generation: GenerationConfig) -> GenerationResult:
        started = time.perf_counter()
        try:
            llm = await asyncio.to_thread(self._load)
            output = await asyncio.to_thread(llm.create_chat_completion, messages=[m.model_dump() for m in messages], temperature=generation.temperature, top_p=generation.top_p, max_tokens=generation.max_tokens, stop=generation.stop)
            latency = (time.perf_counter() - started) * 1000
            text = output["choices"][0]["message"]["content"] or ""
            usage = output.get("usage", {})
            generated = usage.get("completion_tokens", max(1, len(text.split())))
            total = usage.get("total_tokens", generated)
            return GenerationResult(text=text, input_tokens=usage.get("prompt_tokens"), output_tokens=generated, latency_ms=latency, generation_tokens_per_second=generated / (latency / 1000), total_tokens_per_second=total / (latency / 1000), backend_metadata={"engine": "llama.cpp", "n_gpu_layers": self.config.get("n_gpu_layers", 0)})
        except Exception as exc:
            return GenerationResult(error=str(exc), latency_ms=(time.perf_counter() - started) * 1000)

    async def close(self) -> None:
        self._llm = None


class TransformersAdapter:
    def __init__(self, model_ref: str, config: dict[str, Any]):
        self.model_ref, self.config, self._model, self._tokenizer, self._device = model_ref, config, None, None, "cpu"

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        requested = self.config.get("device")
        self._device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype_name = self.config.get("torch_dtype", "auto")
        dtype = "auto" if dtype_name == "auto" else getattr(torch, dtype_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_ref, local_files_only=self.config.get("local_files_only", True))
        kwargs: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": self.config.get("local_files_only", True)}
        if self._device != "cpu": kwargs["device_map"] = self.config.get("device_map", "auto")
        self._model = AutoModelForCausalLM.from_pretrained(self.model_ref, **kwargs)
        if self._device == "cpu": self._model.to("cpu")

    async def health_check(self) -> HealthResult:
        try:
            if self._model is None: await asyncio.to_thread(self._load)
            return HealthResult(ok=True, message=f"Transformers model loaded on {self._device}", hardware=detect_hardware())
        except Exception as exc:
            return HealthResult(ok=False, message=f"Transformers unavailable: {exc}", hardware=detect_hardware())

    async def generate(self, messages: list[Message], generation: GenerationConfig) -> GenerationResult:
        started = time.perf_counter()
        try:
            if self._model is None: await asyncio.to_thread(self._load)
            import torch
            prompt = self._tokenizer.apply_chat_template([m.model_dump() for m in messages], tokenize=False, add_generation_prompt=True) if getattr(self._tokenizer, "chat_template", None) else _prompt(messages)
            def infer() -> tuple[str, int, int]:
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
                with torch.inference_mode():
                    ids = self._model.generate(**inputs, max_new_tokens=generation.max_tokens, do_sample=generation.temperature > 0, temperature=max(generation.temperature, 1e-5), top_p=generation.top_p, pad_token_id=self._tokenizer.eos_token_id)
                new_ids = ids[0][inputs.input_ids.shape[1]:]
                return self._tokenizer.decode(new_ids, skip_special_tokens=True), int(inputs.input_ids.shape[1]), int(new_ids.shape[0])
            text, input_tokens, output_tokens = await asyncio.to_thread(infer)
            latency = (time.perf_counter() - started) * 1000
            return GenerationResult(text=text, input_tokens=input_tokens, output_tokens=output_tokens, latency_ms=latency, generation_tokens_per_second=output_tokens / (latency / 1000), total_tokens_per_second=(input_tokens + output_tokens) / (latency / 1000), backend_metadata={"engine": "transformers", "device": self._device})
        except Exception as exc:
            return GenerationResult(error=str(exc), latency_ms=(time.perf_counter() - started) * 1000)

    async def close(self) -> None:
        self._model, self._tokenizer = None, None
