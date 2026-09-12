from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Column, JSON
from sqlmodel import Field, SQLModel, Session, create_engine, select

from backend.schemas import BackendType, ModelProfileIn, RunStatus

ROOT = Path(__file__).resolve().parents[2]
engine = create_engine(f"sqlite:///{ROOT / 'gravity_bench.db'}", connect_args={"check_same_thread": False})


class ModelProfile(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    backend: BackendType
    model_ref: str
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)


class BenchmarkRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    status: RunStatus = RunStatus.PENDING
    suite: str
    settings: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    hardware: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    current_task: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None


class TaskResult(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(index=True)
    model_id: int = Field(index=True)
    task_id: str
    category: str
    language: str
    difficulty: str
    prompt: str
    output: str = ""
    score: float = 0
    passed: bool = False
    details: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    error: str | None = None


def init_db() -> None: SQLModel.metadata.create_all(engine)
def session() -> Session: return Session(engine)

def create_profile(data: ModelProfileIn) -> ModelProfile:
    with session() as db:
        item = ModelProfile(**data.model_dump()); db.add(item); db.commit(); db.refresh(item); return item

def list_profiles() -> list[ModelProfile]:
    with session() as db: return list(db.exec(select(ModelProfile).order_by(ModelProfile.id.desc())))

def get_profile(profile_id: int) -> ModelProfile | None:
    with session() as db: return db.get(ModelProfile, profile_id)

def update_profile(profile_id: int, data: ModelProfileIn) -> ModelProfile | None:
    with session() as db:
        item = db.get(ModelProfile, profile_id)
        if not item: return None
        config = {**(item.config or {}), **data.config} if item.backend == data.backend else dict(data.config)
        if data.backend == BackendType.OPENAI:
            key = data.config.get("api_key") or (item.config or {}).get("api_key")
            if key: config["api_key"] = key
            else: config.pop("api_key", None)
        item.name, item.backend, item.model_ref, item.config = data.name, data.backend, data.model_ref, config
        db.add(item); db.commit(); db.refresh(item); return item

def create_run(suite: str, settings: dict, hardware: dict, total: int) -> BenchmarkRun:
    with session() as db:
        item = BenchmarkRun(suite=suite, settings=settings, hardware=hardware, total_tasks=total); db.add(item); db.commit(); db.refresh(item); return item

def get_run(run_id: int) -> BenchmarkRun | None:
    with session() as db: return db.get(BenchmarkRun, run_id)

def list_runs(limit: int | None = None, offset: int = 0) -> list[BenchmarkRun]:
    with session() as db:
        query = select(BenchmarkRun).order_by(BenchmarkRun.id.desc()).offset(offset)
        if limit is not None: query = query.limit(limit)
        return list(db.exec(query))

def delete_run(run_id: int) -> bool:
    with session() as db:
        item = db.get(BenchmarkRun, run_id)
        if not item: return False
        for row in db.exec(select(TaskResult).where(TaskResult.run_id == run_id)): db.delete(row)
        db.delete(item); db.commit(); return True

def update_run(run_id: int, **values: Any) -> None:
    with session() as db:
        item = db.get(BenchmarkRun, run_id)
        for key, value in values.items(): setattr(item, key, value)
        db.add(item); db.commit()

def add_result(result: TaskResult) -> None:
    with session() as db: db.add(result); db.commit()

def results_for_run(run_id: int) -> list[TaskResult]:
    with session() as db: return list(db.exec(select(TaskResult).where(TaskResult.run_id == run_id)))
