from pathlib import Path

from backend.reports import _html_models, _html_results, export_report
from backend.storage.database import BenchmarkRun, TaskResult

def test_html_results_escape_output():
    html = _html_results([{"task_id": "math-1", "category": "math", "language": "en", "score": 1.0, "output": "<script>alert(1)</script>", "error": None}], {})
    assert "math-1" in html and "&lt;script&gt;" in html and "<script>" not in html

def test_html_results_include_error():
    html = _html_results([{"task_id": "x", "category": "math", "language": "en", "score": 0, "output": "", "error": "boom"}], {})
    assert "boom" in html and "class='error'" in html

def test_html_models_lists_breakdown():
    html = _html_models({"models": {"7": {"total": {"score": 0.5, "count": 2}, "math": {"score": 1.0, "count": 1}}}}, {"7": "model-a"})
    assert "model-a" in html and "math" in html and "50.0%" in html

def test_export_html_records_outputs(tmp_path, monkeypatch):
    import backend.reports as reports
    reports.invalidate(1)
    row = TaskResult(run_id=1, model_id=1, task_id="math-1", category="math", language="en", difficulty="easy", prompt="1+1?", output="The answer is 2.", score=1.0, passed=True)
    monkeypatch.setattr(reports, "ROOT", tmp_path)
    monkeypatch.setattr(reports, "results_for_run", lambda run_id: [row])
    monkeypatch.setattr(reports, "list_profiles", lambda: [])
    path, _ = export_report(BenchmarkRun(id=1, suite="quick"), "html")
    html = Path(path).read_text(encoding="utf-8")
    assert "The answer is 2." in html and "Models" in html

def test_report_cache(monkeypatch):
    import backend.reports as reports
    reports.invalidate(4242)
    calls = {"n": 0}
    def fake_results(run_id): calls["n"] += 1; return []
    monkeypatch.setattr(reports, "results_for_run", fake_results)
    monkeypatch.setattr(reports, "list_profiles", lambda: [])
    run = BenchmarkRun(id=4242, suite="quick")
    first = reports.report(run)
    assert reports.report(run) is first and calls["n"] == 1
    assert "results" not in reports.report(run, include_results=False)
    reports.invalidate(4242)
    assert reports.report(run) is not first and calls["n"] == 2
