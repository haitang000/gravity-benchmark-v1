from backend.adapters.local import LlamaCppAdapter
from backend.schemas import GenerationConfig, Message


class FakeLlama:
    def __init__(self): self.kwargs = None
    def create_chat_completion(self, **kwargs):
        self.kwargs = kwargs
        if kwargs.get("stream"):
            def stream():
                yield {"choices": [{"delta": {"content": "The "}}]}
                yield {"choices": [{"delta": {"content": "answer is 12."}}]}
                yield {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}}
            return stream()
        return {"choices": [{"message": {"content": "The answer is 12."}}], "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}}


def messages(): return [Message(role="user", content="1+11?")]

async def test_llama_cpp_streaming():
    item = LlamaCppAdapter("m", {}); fake = FakeLlama(); item._llm = fake
    seen = []
    async def on_text(text): seen.append(text)
    result = await item.generate(messages(), GenerationConfig(), on_text)
    assert result.text == "The answer is 12." and seen == ["The ", "The answer is 12."]
    assert result.ttft_ms is not None and result.output_tokens == 4
    assert result.backend_metadata["stream"] is True and fake.kwargs["stream"] is True
    assert result.generation_tokens_per_second and result.generation_tokens_per_second > 0

async def test_llama_cpp_non_streaming():
    item = LlamaCppAdapter("m", {}); fake = FakeLlama(); item._llm = fake
    result = await item.generate(messages(), GenerationConfig())
    assert result.text == "The answer is 12." and result.ttft_ms is None
    assert result.backend_metadata["stream"] is False and fake.kwargs.get("stream") is None
