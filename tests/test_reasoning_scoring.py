from backend.benchmarks.scoring import score
from backend.benchmarks.tasks import BenchmarkTask

OPEN = "<" + "think" + ">"
CLOSE = "<" + "/think" + ">"

def task(expected, category="instruction"): return BenchmarkTask("r", category, "en", "easy", "", expected)

def test_think_block_is_ignored():
    output = f"{OPEN}Maybe the answer is NO, but I should say the final answer.{CLOSE}OK"
    assert score(task({"regex": "^OK$"}), output)[0] == 1

def test_block_with_newlines_is_ignored():
    output = f"{OPEN}\nstep one\nstep two\n{CLOSE}\nOK"
    assert score(task({"exact": "OK"}), output)[0] == 1

def test_json_after_reasoning_block():
    output = f'{OPEN}{{"name": 5}} wrong try{CLOSE}{{"name": "x"}}'
    assert score(task({"json_schema": {"name": "string"}}), output)[0] == 1

def test_last_json_in_tagless_cot_wins():
    assert score(task({"json_schema": {"name": "string"}}), 'Example: {"name": 5}. Final: {"name": "x"}')[0] == 1

def test_json_with_prose_and_fence():
    assert score(task({"json_keys": ["name"]}), 'Here you go:\n```json\n{"name": "x", "count": 2}\n```')[0] == 1

def test_tagless_cot_uses_answer_tail():
    assert score(task({"exact": "OK", "max_chars": 2}), "The instruction demands a short reply.\nOK")[0] == 1

def test_line_prefixes_after_tagless_cot():
    assert score(task({"line_prefixes": ["PLAN:", "RISK:", "TEST:"], "line_count": 3}), "Let me plan this.\nPLAN: ship\nRISK: bugs\nTEST: run")[0] == 1

def test_forbid_only_in_reasoning_block():
    output = f"{OPEN}this is danger? no, safe to ignore{CLOSE}SAFE"
    assert score(task({"contains": ["SAFE"], "forbid": ["danger"]}), output)[0] == 1

def test_math_reads_final_number():
    assert score(task({"answer": "12"}, "math"), "First I thought 7, then 3 + 9, so the answer is 12")[0] == 1

def test_code_fence_inside_reasoning_is_ignored():
    output = f"{OPEN}```python\ndef ok(): return False\n```{CLOSE}```python\ndef ok():\n    return True\n```"
    value, details = score(task({"tests": "assert ok()"}, "coding"), output)
    assert value in (0, 1) and details.get("kind") != "syntax_error"
