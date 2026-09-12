import httpx
import pytest
from pydantic import ValidationError

from backend.adapters.openai_compatible import OpenAICompatibleAdapter
from backend.schemas import ModelProfileIn

def adapter(config): return OpenAICompatibleAdapter("m", config)

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
