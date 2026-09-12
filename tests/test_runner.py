import asyncio

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine

import backend.execution.runner as runner
import backend.storage.database as database
from backend.benchmarks.tasks import BenchmarkTask
from backend.metrics.aggregate import aggregate
from backend.schemas import GenerationConfig, GenerationResult, HealthResult, ModelProfileIn, RunStatus


class FakeAdapter:
    def __init__(self, failures=0):
        self.failures, self.calls = failures, 0
    async def health_check(self): return HealthResult(ok=True, message="ok")
    async def generate(self, messages, generation, on_text=None):
        self.calls += 1
        if self.calls <= self.failures: return GenerationResult(error="boom")
        return GenerationResult(text="The answer is 12.", input_tokens=5, output_tokens=3)
    async def close(self): return None


class SequenceAdapter:
    def __init__(self, outputs):
        self.outputs, self.calls = list(outputs), 0
    async def health_check(self): return HealthResult(ok=True, message="ok")
    async def generate(self, messages, generation, on_text=None):
        text = self.outputs[min(self.calls, len(self.outputs) - 1)]; self.calls += 1
        return GenerationResult(text=text)
    async def close(self): return None


class ExplodingAdapter:
    def __init__(self, failures):
        self.failures, self.calls = failures, 0
    async def health_check(self): return HealthResult(ok=True, message="ok")
    async def generate(self, messages, generation, on_text=None):
        self.calls += 1
        if self.calls <= self.failures: raise RuntimeError("network down")
        return GenerationResult(text="The answer is 12.")
    async def close(self): return None


class StreamingAdapter:
    def __init__(self):
        self.calls, self.run_id, self.seen = 0, 0, None
    async def health_check(self): return HealthResult(ok=True, message="ok")
    async def generate(self, messages, generation, on_text=None):
        self.calls += 1
        if on_text: await on_text("The answer is 12.")
        self.seen = runner.streaming_outputs(self.run_id)
        return GenerationResult(text="The answer is 12.")
    async def close(self): return None


class SelfPausingAdapter:
    def __init__(self):
        self.calls, self.run_id = 0, 0
    async def health_check(self): return HealthResult(ok=True, message="ok")
    async def generate(self, messages, generation, on_text=None):
        self.calls += 1
        if self.calls == 1: runner.pause(self.run_id)
        return GenerationResult(text="The answer is 12.")
    async def close(self): return None


def math_task(task_id="math-1"): return BenchmarkTask(task_id, "math", "en", "easy", "1+11?", {"answer": "12"})

def setup(monkeypatch, adapter, tasks=1):
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    monkeypatch.setattr(database, "engine", engine)
    SQLModel.metadata.create_all(engine)
    profile = database.create_profile(ModelProfileIn(name="m", backend="openai_compatible", model_ref="m", config={"base_url": "http://h/v1"}))
    run = database.create_run("quick", {}, {}, tasks)
    monkeypatch.setattr(runner, "create_adapter", lambda *args: adapter)
    monkeypatch.setattr(runner, "load_tasks", lambda *args: [math_task(f"math-{i}") for i in range(tasks)])
    return profile, run

async def wait_for(condition, timeout=2.0):
    for _ in range(int(timeout / 0.01)):
        if condition(): return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met in time")

def test_retry_kind_rules():
    assert runner.retry_kind(GenerationResult(error="x"), 0, {}) == "error"
    assert runner.retry_kind(GenerationResult(text="wrong"), 0, {"kind": "test_failure"}) == "score"
    assert runner.retry_kind(GenerationResult(text=""), 0, {"kind": "sandbox_unavailable"}) is None
    assert runner.retry_kind(GenerationResult(text="ok"), 1, {}) is None

async def test_execute_retries_generation_errors(monkeypatch):
    fake = FakeAdapter(failures=2)
    profile, run = setup(monkeypatch, fake)
    await runner.execute(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1, 2)
    rows = database.results_for_run(run.id)
    assert fake.calls == 3 and len(rows) == 1
    assert rows[0].passed and rows[0].error is None and rows[0].metrics["attempts"] == 3
    assert rows[0].metrics["error_retries"] == 2
    assert database.get_run(run.id).status == RunStatus.COMPLETED

async def test_execute_does_not_retry_failed_scores(monkeypatch):
    fake = SequenceAdapter(["The answer is 99", "The answer is 12"])
    profile, run = setup(monkeypatch, fake)
    await runner.execute(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1, 1)
    rows = database.results_for_run(run.id)
    assert fake.calls == 1 and not rows[0].passed and rows[0].metrics["attempts"] == 1

async def test_execute_retries_wrong_answers(monkeypatch):
    fake = SequenceAdapter(["The answer is 99", "The answer is 12"])
    profile, run = setup(monkeypatch, fake)
    await runner.execute(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1, 0, 1, 1)
    rows = database.results_for_run(run.id)
    assert fake.calls == 2 and rows[0].passed and rows[0].metrics["attempts"] == 2
    assert rows[0].metrics["score_retries"] == 1

async def test_execute_stops_after_score_retries(monkeypatch):
    fake = SequenceAdapter(["The answer is 99"])
    profile, run = setup(monkeypatch, fake)
    await runner.execute(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1, 0, 1, 2)
    rows = database.results_for_run(run.id)
    assert fake.calls == 3 and not rows[0].passed and rows[0].metrics["score_retries"] == 2

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

async def test_execute_reports_streaming_outputs(monkeypatch):
    fake = StreamingAdapter()
    profile, run = setup(monkeypatch, fake)
    fake.run_id = run.id
    await runner.execute(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1, 0)
    assert fake.seen == {f"{profile.id}:math-0": "The answer is 12."}
    assert runner.streaming_outputs(run.id) == {}

async def test_execute_repeats_suite_and_averages(monkeypatch):
    fake = SequenceAdapter(["The answer is 99", "The answer is 12"])
    profile, run = setup(monkeypatch, fake)
    await runner.execute(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1, 0, 2)
    rows = database.results_for_run(run.id)
    summary = aggregate(rows)
    assert fake.calls == 2 and len(rows) == 2
    assert {row.metrics["repeat"] for row in rows} == {1, 2}
    assert database.get_run(run.id).settings["repeats"] == 2
    assert summary["total"]["score"] == 0.5 and summary["total"]["pass_rate"] == 0.5

def test_pause_resume_require_active_run():
    assert runner.pause(999) is False and runner.resume(999) is False

async def test_execute_pause_and_resume(monkeypatch):
    fake = SelfPausingAdapter()
    profile, run = setup(monkeypatch, fake, tasks=2)
    fake.run_id = run.id
    job = asyncio.create_task(runner.execute(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1))
    await wait_for(lambda: database.get_run(run.id).status == RunStatus.PAUSED)
    await asyncio.sleep(0.05)
    assert fake.calls == 1
    assert runner.resume(run.id) is True
    await job
    assert fake.calls == 2
    assert database.get_run(run.id).status == RunStatus.COMPLETED

async def test_cancel_paused_run(monkeypatch):
    fake = SelfPausingAdapter()
    profile, run = setup(monkeypatch, fake, tasks=2)
    fake.run_id = run.id
    runner.start(run.id, [profile.id], "quick", ["en"], GenerationConfig(), {}, 42, 1)
    job = runner.jobs[run.id]
    await wait_for(lambda: database.get_run(run.id).status == RunStatus.PAUSED)
    assert runner.cancel(run.id) is True
    with pytest.raises(asyncio.CancelledError): await job
    assert database.get_run(run.id).status == RunStatus.CANCELLED
