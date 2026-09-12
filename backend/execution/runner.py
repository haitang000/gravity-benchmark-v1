from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from backend.adapters.factory import create_adapter
from backend.benchmarks.scoring import score
from backend.benchmarks.tasks import dataset_hash, load_tasks
from backend.hardware.detect import detect_hardware
from backend.schemas import GenerationConfig, GenerationResult, RunStatus, Message
from backend.storage.database import TaskResult, add_result, get_profile, get_run, update_run

jobs: dict[int, asyncio.Task] = {}

def should_retry(result: GenerationResult) -> bool:
    return bool(result.error)

async def execute(run_id: int, model_ids: list[int], suite: str, languages: list[str], generation: GenerationConfig, counts: dict[str,int], seed: int, concurrency: int, retries: int = 0) -> None:
    run = get_run(run_id); tasks = load_tasks(suite, languages, seed, counts); update_run(run_id, status=RunStatus.RUNNING, total_tasks=len(tasks) * len(model_ids), settings={"model_ids": model_ids, "suite": suite, "languages": languages, "generation": generation.model_dump(), "seed": seed, "retries": retries, "dataset_hash": dataset_hash()})
    total_done = total_failed = 0
    try:
        for model_id in model_ids:
            profile = get_profile(model_id)
            if not profile: raise ValueError(f"Unknown model id {model_id}")
            adapter = create_adapter(profile.backend, profile.model_ref, profile.config)
            health = await adapter.health_check()
            if not health.ok: raise RuntimeError(f"{profile.name}: {health.message}")
            sem = asyncio.Semaphore(concurrency)
            async def one(task):
                nonlocal total_done, total_failed
                async with sem:
                    update_run(run_id, current_task=f"{profile.name}: {task.id}")
                    attempts = 0
                    while True:
                        attempts += 1
                        try:
                            result = await adapter.generate([Message(role="user", content=task.prompt)], generation)
                            output = result.text; value, details = score(task, output) if not result.error else (0, {"kind":"generation_error"})
                        except Exception as exc:
                            result, output, value, details = GenerationResult(error=str(exc)), "", 0, {"kind": "exception", "message": str(exc)}
                        if attempts > retries or not should_retry(result): break
                    metrics = result.model_dump(exclude={"text", "error", "backend_metadata"}) | result.backend_metadata | {"attempts": attempts}
                    add_result(TaskResult(run_id=run_id, model_id=model_id, task_id=task.id, category=task.category, language=task.language, difficulty=task.difficulty, prompt=task.prompt, output=output, score=value, passed=value >= 1, details=details, metrics=metrics, error=result.error))
                    total_done += 1; total_failed += bool(result.error); update_run(run_id, completed_tasks=total_done, failed_tasks=total_failed)
            await asyncio.gather(*(one(task) for task in tasks))
            await adapter.close()
        update_run(run_id, status=RunStatus.COMPLETED, finished_at=datetime.utcnow(), current_task=None)
    except asyncio.CancelledError:
        update_run(run_id, status=RunStatus.CANCELLED, finished_at=datetime.utcnow()); raise
    except Exception as exc:
        update_run(run_id, status=RunStatus.FAILED, error=str(exc), finished_at=datetime.utcnow())

def start(*args: Any) -> int:
    run_id = args[0]; jobs[run_id] = asyncio.create_task(execute(*args)); return run_id

def cancel(run_id: int) -> bool:
    task = jobs.get(run_id)
    return bool(task and not task.done() and task.cancel())
