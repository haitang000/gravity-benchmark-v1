from pathlib import Path

from backend.reports import _html_results, export_report
from backend.storage.database import BenchmarkRun, TaskResult

def test_html_results_escape_output():
    html = _html_results([{"task_id": "math-1", "category": "math", "language": "en", "score": 1.0, "output": "<script>alert(1)</script>", "error": None}])
    assert "math-1" in html and "&lt;script&gt;" in html and "<script>" not in html

def test_html_results_include_error():
    html = _html_results([{"task_id": "x", "category": "math", "language": "en", "score": 0, "output": "", "error": "boom"}])
    assert "boom" in html and "class='error'" in html

def test_export_html_records_outputs(tmp_path, monkeypatch):
    import backend.reports as reports
    row = TaskResult(run_id=1, model_id=1, task_id="math-1", category="math", language="en", difficulty="easy", prompt="1+1?", output="The answer is 2.", score=1.0, passed=True)
    monkeypatch.setattr(reports, "ROOT", tmp_path)
    monkeypatch.setattr(reports, "results_for_run", lambda run_id: [row])
    path, _ = export_report(BenchmarkRun(id=1, suite="quick"), "html")
    assert "The answer is 2." in Path(path).read_text(encoding="utf-8")
