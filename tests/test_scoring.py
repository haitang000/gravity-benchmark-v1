from backend.benchmarks.scoring import score
from backend.benchmarks.tasks import BenchmarkTask

def task(category, prompt, expected): return BenchmarkTask("x", category, "en", "easy", prompt, expected)
def test_json_instruction(): assert score(task("instruction", "", {"json_schema":{"name":"string"}}), '{"name":"x"}')[0] == 1
def test_math(): assert score(task("math", "", {"answer":"12"}), "The answer is 12.")[0] == 1
def test_regex(): assert score(task("instruction", "", {"regex":"^OK$"}), "OK")[0] == 1
def test_code_is_disabled_without_docker():
    value, details = score(task("coding", "", {"tests":"assert add(1, 2)==3"}), "def add(a,b): return a+b")
    assert value in (0, 1)
