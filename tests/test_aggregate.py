import pytest
from types import SimpleNamespace

from backend.metrics.aggregate import aggregate, token_total

def row(category, score, passed=True, task_id="t", language="en", difficulty="easy", model_id=1, **metrics): return SimpleNamespace(category=category, score=score, passed=passed, task_id=task_id, language=language, difficulty=difficulty, model_id=model_id, metrics=metrics)

def test_token_total():
    assert token_total({"input_tokens": 10, "output_tokens": 20}) == 30
    assert token_total({"input_tokens": 10}) == 10
    assert token_total({"output_tokens": 20}) == 20
    assert token_total({}) is None

def test_avg_tokens_per_task():
    summary = aggregate([row("math", 1, input_tokens=10, output_tokens=20, latency_ms=100), row("math", 0, passed=False, input_tokens=30, latency_ms=300), row("coding", 1, output_tokens=50, latency_ms=200)])
    assert summary["total"]["avg_input_tokens"] == 20
    assert summary["total"]["avg_output_tokens"] == 35
    assert summary["total"]["avg_total_tokens"] == pytest.approx(110 / 3)
    assert summary["math"]["avg_total_tokens"] == 30
    assert summary["coding"]["avg_total_tokens"] == 50

def test_avg_tokens_missing():
    summary = aggregate([row("math", 1, latency_ms=100), row("math", 1)])
    assert summary["total"]["avg_input_tokens"] is None
    assert summary["total"]["avg_output_tokens"] is None
    assert summary["total"]["avg_total_tokens"] is None

def test_aggregate_per_model():
    summary = aggregate([row("math", 1, task_id="a", model_id=1), row("math", 0, passed=False, task_id="a", model_id=1), row("math", 1, task_id="b", model_id=2), row("coding", 1, task_id="c", model_id=2)])
    assert summary["models"]["1"]["total"]["score"] == 0.5
    assert summary["models"]["1"]["math"]["count"] == 2
    assert "coding" not in summary["models"]["1"]
    assert summary["models"]["2"]["total"]["score"] == 1
    assert summary["models"]["2"]["coding"]["count"] == 1

def test_task_language_and_difficulty_token_breakdowns():
    summary = aggregate([
        row("math", 1, task_id="math-a", language="zh", difficulty="hard", input_tokens=10, output_tokens=5),
        row("math", 0, passed=False, task_id="math-a", language="zh", difficulty="hard", input_tokens=20, output_tokens=7),
        row("coding", 1, task_id="code-a", language="en", difficulty="medium", input_tokens=30, output_tokens=9),
    ])
    assert summary["task_breakdown"]["math-a"]["avg_total_tokens"] == 21
    assert summary["language_breakdown"]["zh"]["avg_output_tokens"] == 6
    assert summary["difficulty_breakdown"]["hard"]["avg_input_tokens"] == 15
