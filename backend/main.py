from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.adapters.factory import create_adapter
from backend.benchmarks.tasks import load_tasks
from backend.execution.runner import cancel, start
from backend.hardware.detect import detect_hardware
from backend.reports import export_report, report
from backend.schemas import ModelProfileIn, RunProgress, RunRequest
from backend.storage.database import create_profile, create_run, get_profile, get_run, init_db, list_profiles, list_runs, results_for_run

app = FastAPI(title="GravityBench", version="0.1.0")
ROOT = Path(__file__).resolve().parents[1]

@app.on_event("startup")
def startup(): init_db()

@app.get("/api/health")
def health(): return {"ok": True, "hardware": detect_hardware()}

@app.get("/api/models")
def models():
    return [_public_model(item) for item in list_profiles()]

def _public_model(item):
    data = item.model_dump()
    config = dict(data.get("config") or {})
    if config.get("api_key"):
        config["api_key"] = "••••••"
    data["config"] = config
    return data

@app.post("/api/models")
def add_model(data: ModelProfileIn): return _public_model(create_profile(data))

@app.post("/api/models/{model_id}/health")
async def model_health(model_id: int):
    item = get_profile(model_id)
    if not item: raise HTTPException(404, "Model not found")
    result = await create_adapter(item.backend, item.model_ref, item.config).health_check()
    return result

@app.get("/api/benchmarks/tasks")
def tasks(suite: str = "quick", languages: str = "zh,en"):
    return {"suite": suite, "count": len(load_tasks(suite, languages.split(","), 42)), "categories": ["instruction", "math", "coding", "performance"]}

@app.post("/api/runs", response_model=RunProgress)
async def run_benchmark(request: RunRequest):
    task_count = len(load_tasks(request.suite, request.languages, request.seed, request.counts)) * len(request.model_ids)
    item = create_run(request.suite, request.model_dump(), detect_hardware(), task_count)
    start(item.id, request.model_ids, request.suite, request.languages, request.generation, request.counts, request.seed, request.concurrency)
    return item

@app.get("/api/runs")
def runs(): return list_runs()

@app.get("/api/runs/{run_id}")
def run_detail(run_id: int):
    item = get_run(run_id)
    if not item: raise HTTPException(404, "Run not found")
    return {"run": item, "results": results_for_run(run_id), "report": report(item)}

@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: int): return {"cancelled": cancel(run_id)}

@app.get("/api/runs/{run_id}/export/{fmt}")
def export_run(run_id: int, fmt: str):
    item = get_run(run_id)
    if not item: raise HTTPException(404, "Run not found")
    path, name = export_report(item, fmt)
    return FileResponse(path, filename=name)

FRONTEND = ROOT / "frontend" / "dist"
if FRONTEND.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")
    @app.get("/{path:path}")
    def spa(path: str): return FileResponse(FRONTEND / "index.html")

def run():
    import uvicorn; uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
