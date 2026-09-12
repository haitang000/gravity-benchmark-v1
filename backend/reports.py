from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from backend.metrics.aggregate import aggregate
from backend.storage.database import BenchmarkRun, results_for_run

ROOT = Path(__file__).resolve().parents[1]

def _num(value: float | None, digits: int = 1) -> str: return f"{value:.{digits}f}" if value is not None else "-"

def _html_summary(summary: dict[str, Any]) -> str:
    head = "<tr><th>Category</th><th>Score</th><th>Avg Input Tokens/Task</th><th>Avg Output Tokens/Task</th><th>Avg Total Tokens/Task</th><th>Tokens/s</th></tr>"
    cards = "".join(f"<tr><td>{escape(k)}</td><td>{v.get('score', 0):.1%}</td><td>{_num(v.get('avg_input_tokens'))}</td><td>{_num(v.get('avg_output_tokens'))}</td><td>{_num(v.get('avg_total_tokens'))}</td><td>{_num(v.get('avg_generation_tokens_per_second'))}</td></tr>" for k, v in summary.items())
    return f"<table>{head}{cards}</table>"

def _html_error(error: str | None) -> str: return f"<p class='error'>{escape(error)}</p>" if error else ""

def _html_results(results: list[dict[str, Any]]) -> str:
    head = "<tr><th>Task</th><th>Category</th><th>Language</th><th>Score</th><th>Output</th></tr>"
    cards = "".join(f"<tr><td>{escape(str(r.get('task_id')))}</td><td>{escape(str(r.get('category')))}</td><td>{escape(str(r.get('language')))}</td><td>{r.get('score', 0):.1%}</td><td>{_html_error(r.get('error'))}<pre>{escape(r.get('output') or '')}</pre></td></tr>" for r in results)
    return f"<table>{head}{cards}</table>"

def report(run: BenchmarkRun) -> dict[str, Any]:
    rows = results_for_run(run.id)
    return {"run": {k: v for k, v in run.model_dump().items() if k != "settings"}, "settings": run.settings, "summary": aggregate(rows), "results": [r.model_dump() for r in rows]}

def export_report(run: BenchmarkRun, fmt: str) -> tuple[str, str]:
    data = report(run); stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S"); base = ROOT / "reports"; base.mkdir(exist_ok=True)
    if fmt == "json": path = base / f"run-{run.id}-{stamp}.json"; path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    elif fmt == "csv":
        path = base / f"run-{run.id}-{stamp}.csv"; buf = io.StringIO(); rows = data["results"]; writer = csv.DictWriter(buf, fieldnames=["task_id", "category", "language", "score", "passed", "output", "metrics"]); writer.writeheader()
        for row in rows: writer.writerow({k: json.dumps(row[k], ensure_ascii=False) if isinstance(row[k], dict) else row[k] for k in writer.fieldnames})
        path.write_text(buf.getvalue(), encoding="utf-8")
    elif fmt == "html":
        path = base / f"run-{run.id}-{stamp}.html"; style = "body{font-family:system-ui;margin:24px;color:#192033}h1{font-size:24px}h2{font-size:18px;margin-top:28px}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #e8ebf3;padding:8px;text-align:left;vertical-align:top}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;margin:0}.error{color:#c73333}"
        path.write_text(f"<!doctype html><meta charset='utf-8'><title>GravityBench Run {run.id}</title><style>{style}</style><h1>GravityBench Run {run.id}</h1><h2>Summary</h2>{_html_summary(data['summary'])}<h2>Model Outputs ({len(data['results'])})</h2>{_html_results(data['results'])}", encoding="utf-8")
    else: raise ValueError("format must be json, csv or html")
    return str(path), path.name
