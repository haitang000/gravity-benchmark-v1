from backend.benchmarks.scoring import score
from backend.benchmarks.tasks import BenchmarkTask


def test_hard_line_prefixes_are_scored_per_line():
    task = BenchmarkTask("prefix", "instruction", "en", "hard", "", {"line_count": 3, "line_prefixes": ["PLAN:", "RISK:", "TEST:"], "contains": ["model"], "forbid": ["?"]})
    value, details = score(task, "PLAN: model\nRISK: model\nTEST: model")
    assert value == 1
    assert all(details["checks"])
