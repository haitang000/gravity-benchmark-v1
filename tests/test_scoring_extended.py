import json

from backend.benchmarks.scoring import score
from backend.benchmarks.tasks import BenchmarkTask


def make(expected, category="instruction"):
    return BenchmarkTask("extended", category, "en", "easy", "", expected)


def test_boolean_json_schema():
    assert score(make({"json_schema": {"enabled": "boolean"}}), json.dumps({"enabled": True}))[0] == 1


def test_forbidden_text():
    value, _ = score(make({"contains": ["please"], "forbid": ["!"]}), "please save")
    assert value == 1


def test_fraction_math():
    value, _ = score(make({"answer": "1/2"}, "math"), "The answer is 1/2")
    assert value == 1
