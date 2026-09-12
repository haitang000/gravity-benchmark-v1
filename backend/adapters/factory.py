from backend.adapters.local import LlamaCppAdapter, TransformersAdapter
from backend.adapters.openai_compatible import OpenAICompatibleAdapter
from backend.schemas import BackendType


def create_adapter(backend: BackendType, model_ref: str, config: dict):
    if backend == BackendType.OPENAI: return OpenAICompatibleAdapter(model_ref, config)
    if backend == BackendType.LLAMA_CPP: return LlamaCppAdapter(model_ref, config)
    if backend == BackendType.TRANSFORMERS: return TransformersAdapter(model_ref, config)
    raise ValueError(f"Unsupported backend {backend}")
