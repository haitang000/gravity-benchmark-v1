from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

import backend.execution.runner as runner
import backend.storage.database as database
from backend.benchmarks.tasks import BenchmarkTask
from backend.schemas import GenerationConfig, GenerationResult, HealthResult, ModelProfileIn, RunStatus


class FakeAdapter:
    def __init__(self, failures=0):
        self.failures, self.calls = failures, 0
    async def health_check(self): return HealthResult(ok=True, message="ok")
    async def generate(self, messages, generation):
        self.calls += 1
        if self.calls <= self.failures: return GenerationResult(error="boom")
        return GenerationResult(text="The answer is 12.", input_tokens=5, output_tokens=3)
    async def close(self): return None


class SequenceAdapter:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0
    async def health_check(self): return HealthResult(ok=True, message="ok")
    async def generate(self, messages, generation):
        text = self.outputs[min(self.calls, len(self.outputs) - 1)]; self.calls += 1
        return GenerationResult(text=text)
    async def close(self): return None


class ExplodingAdapter:
    def __init__(self, failures):
        self.failures, self.calls = failures, 0
    async def health_check(self): return HealthResult(ok=True, message="ok")
    async def generate(self, messages, generation):
        self.calls += 1
        if self.calls <= self.failures: raise RuntimeError("network down")
        return GenerationResult(text="The answer is 12.")
    async def close(self): return None


def math_task(): return BenchmarkTask("math-1", "math", "en", "easy", "1+11?", {"answer": "12"})

def setup(monkeypatch, adapter):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    monkeypatch.setattr(database, "engine", engine)
    SQLModel.metadata.create_all(engine)
    profile = database.create_profile(ModelProfileIn(name="m", backend="openai_compatible", model_ref="m", config={"base_url": "http://h/v1"}))
    run = database.create_run("quick", {}, {}, 1)
    monkeypatch.setattr(runner, "create_adapter", lambda *args: adapter)
    monkeypatch.setattr(runner, "load_tasks", lambda *args: [math_task()])
    return profile, run

def test_should_retry_rules():
    assert runner.should_retry(GenerationResult(error="x")) is True
    assert runner.should_retry(GenerationResult(text="wrong")) is False
    assert runner.should_retry(GenerationResult(text="")) is False

async def test_execute_retries_generation_errors(monkeypatch):
    fake = FakeAdapter(failures=2)
    profile, run = setup(monkeypatch, fake)
    await runner.execute(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1, 2)
    rows = database.results_for_run(run.id)
    assert fake.calls == 3 and len(rows) == 1
    assert rows[0].passed and rows[0].error is None and rows[0].metrics["attempts"] == 3
    assert database.get_run(run.id).status == RunStatus.COMPLETED

async def test_execute_does_not_retry_failed_scores(monkeypatch):
    fake = SequenceAdapter(["The answer is 99", "The answer is 12"])
    profile, run = setup(monkeypatch, fake)
    await runner.execute(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1, 1)
    rows = database.results_for_run(run.id)
    assert fake.calls == 1 and not rows[0].passed and rows[0].metrics["attempts"] == 1

async def test_execute_retries_exceptions(monkeypatch):
    fake = ExplodingAdapter(failures=1)
    profile, run = setup(monkeypatch, fake)
    await runner.execute(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1, 1)
    rows = database.results_for_run(run.id)
    assert fake.calls == 2 and rows[0].passed and rows[0].metrics["attempts"] == 2

async def test_execute_stops_after_retries(monkeypatch):
    fake = FakeAdapter(failures=99)
    profile, run = setup(monkeypatch, fake)
    await runner.execute(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1, 1)
    assert fake.calls == 2
    row = database.results_for_run(run.id)[0]
    assert row.error == "boom" and row.metrics["attempts"] == 2
    assert database.get_run(run.id).failed_tasks == 1
