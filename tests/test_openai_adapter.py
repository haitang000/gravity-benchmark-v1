import json

import httpx
import pytest
from pydantic import ValidationError

from backend.adapters.base import generation_speed
from backend.adapters.openai_compatible import OpenAICompatibleAdapter
from backend.schemas import GenerationConfig, Message, ModelProfileIn

def adapter(config): return OpenAICompatibleAdapter("m", config)

def test_generation_speed_uses_decode_window():
    assert generation_speed(4, 150, 100) == 80
    assert generation_speed(4, 150) == pytest.approx(4 / 0.15)
    assert generation_speed(2, 100, 100) == 20
    assert generation_speed(0, 100) is None
    assert generation_speed(5, 0) is None

def test_append_path_default():
    item = adapter({"base_url": "http://h/v1"})
    assert item.chat_url == "http://h/v1/chat/completions"
    assert item.models_url == "http://h/v1/models"

def test_append_path_disabled_full_url():
    item = adapter({"base_url": "http://h/v1/chat/completions", "append_path": False})
    assert item.chat_url == "http://h/v1/chat/completions"
    assert item.models_url == "http://h/v1/models"

def test_append_path_disabled_custom_url():
    item = adapter({"base_url": "http://h/custom", "append_path": False})
    assert item.chat_url == "http://h/custom"
    assert item.models_url == "http://h/custom"

def test_models_url_override():
    item = adapter({"base_url": "http://h/custom", "append_path": False, "models_url": "http://h/health/"})
    assert item.models_url == "http://h/health"

async def test_health_probe_custom_endpoint(respx_mock):
    respx_mock.get("http://h/custom").mock(return_value=httpx.Response(405))
    result = await adapter({"base_url": "http://h/custom", "append_path": False}).health_check()
    assert result.ok is True

async def test_health_models_url_strict(respx_mock):
    respx_mock.get("http://h/health").mock(return_value=httpx.Response(404))
    result = await adapter({"base_url": "http://h/custom", "append_path": False, "models_url": "http://h/health"}).health_check()
    assert result.ok is False

def test_profile_accepts_append_path():
    profile = ModelProfileIn(name="m", backend="openai_compatible", model_ref="m", config={"base_url": "http://h/v1", "append_path": False, "models_url": "http://h/health"})
    assert profile.config["append_path"] is False

def test_profile_rejects_bad_values():
    with pytest.raises(ValidationError): ModelProfileIn(name="m", backend="openai_compatible", model_ref="m", config={"base_url": "http://h/v1", "append_path": "yes"})
    with pytest.raises(ValidationError): ModelProfileIn(name="m", backend="openai_compatible", model_ref="m", config={"base_url": "http://h/v1", "models_url": "ftp://h/health"})

async def test_streaming_parses_sse(respx_mock):
    body = 'data: {"choices":[{"delta":{"content":"The "}}]}\n\ndata: {"choices":[{"delta":{"content":"answer is 12."}}]}\n\ndata: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":4,"total_tokens":11}}\n\ndata: [DONE]\n\n'
    route = respx_mock.post("http://h/v1/chat/completions").mock(return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}))
    seen = []
    async def on_text(text): seen.append(text)
    result = await adapter({"base_url": "http://h/v1"}).generate([Message(role="user", content="1+11?")], GenerationConfig(), on_text)
    assert result.text == "The answer is 12." and result.output_tokens == 4 and result.ttft_ms is not None
    assert result.generation_tokens_per_second and result.generation_tokens_per_second > 0
    assert seen == ["The ", "The answer is 12."]
    assert json.loads(route.calls.last.request.content)["stream"] is True

async def test_streaming_falls_back_on_400(respx_mock):
    body = {"choices": [{"message": {"content": "The answer is 12."}}], "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}}
    route = respx_mock.post("http://h/v1/chat/completions").mock(side_effect=[httpx.Response(400, json={"error": "stream unsupported"}), httpx.Response(200, json=body)])
    seen = []
    async def on_text(text): seen.append(text)
    result = await adapter({"base_url": "http://h/v1"}).generate([Message(role="user", content="1+11?")], GenerationConfig(), on_text)
    assert result.text == "The answer is 12." and not result.error
    assert result.backend_metadata["stream"] is False and result.output_tokens == 4
    assert seen == ["The answer is 12."]
    assert json.loads(route.calls[0].request.content)["stream"] is True
    assert "stream" not in json.loads(route.calls[1].request.content)
