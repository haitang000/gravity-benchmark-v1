import pytest
from types import SimpleNamespace

from backend.metrics.aggregate import aggregate, token_total

def row(category, score, passed=True, **metrics): return SimpleNamespace(category=category, score=score, passed=passed, metrics=metrics)

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
