from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

import backend.storage.database as database
from backend.main import app
from backend.schemas import ModelProfileIn

def fresh_db(monkeypatch):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    monkeypatch.setattr(database, "engine", engine)
    SQLModel.metadata.create_all(engine)

def profile(config): return ModelProfileIn(name="m", backend="openai_compatible", model_ref="m", config=config)

def test_update_profile_merges_config_and_keeps_key(monkeypatch):
    fresh_db(monkeypatch)
    item = database.create_profile(profile({"base_url": "http://h/v1", "api_key": "secret", "timeout": 30}))
    updated = database.update_profile(item.id, profile({"base_url": "http://h/v2", "api_key": ""}))
    assert updated.name == "m" and updated.config["base_url"] == "http://h/v2"
    assert updated.config["api_key"] == "secret" and updated.config["timeout"] == 30

def test_update_profile_sets_new_key(monkeypatch):
    fresh_db(monkeypatch)
    item = database.create_profile(profile({"base_url": "http://h/v1", "api_key": "old"}))
    updated = database.update_profile(item.id, profile({"base_url": "http://h/v1", "api_key": "new"}))
    assert updated.config["api_key"] == "new"

def test_update_profile_backend_switch_drops_api_config(monkeypatch):
    fresh_db(monkeypatch)
    item = database.create_profile(profile({"base_url": "http://h/v1", "api_key": "secret"}))
    data = ModelProfileIn(name="gguf", backend="llama_cpp", model_ref="m.gguf", config={"device": "cpu", "n_gpu_layers": 10})
    updated = database.update_profile(item.id, data)
    assert updated.backend == "llama_cpp" and updated.name == "gguf"
    assert updated.config == {"device": "cpu", "n_gpu_layers": 10}

def test_update_profile_missing(monkeypatch):
    fresh_db(monkeypatch)
    assert database.update_profile(123, profile({"base_url": "http://h/v1"})) is None

def test_api_update_model(monkeypatch):
    fresh_db(monkeypatch)
    with TestClient(app) as client:
        created = client.post("/api/models", json={"name": "m", "backend": "openai_compatible", "model_ref": "ref", "config": {"base_url": "http://h/v1", "api_key": "secret"}}).json()
        response = client.put(f"/api/models/{created['id']}", json={"name": "m2", "backend": "openai_compatible", "model_ref": "ref2", "config": {"base_url": "http://h/v2", "api_key": ""}})
        assert response.status_code == 200
        body = response.json()
        assert body["name"] == "m2" and body["model_ref"] == "ref2" and body["config"]["base_url"] == "http://h/v2" and body["config"]["api_key"] == "••••••"
        missing = client.put("/api/models/999", json={"name": "x", "backend": "llama_cpp", "model_ref": "x", "config": {}})
        assert missing.status_code == 404
