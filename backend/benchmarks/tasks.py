from __future__ import annotations

import json
import random
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = ROOT / "datasets" / "fixtures" / "core_tasks.json"
SUITE_COUNTS = {"quick": {"instruction": 20, "math": 20, "coding": 10, "performance": 10}, "standard": {"instruction": 80, "math": 80, "coding": 50, "performance": 30}, "full": {"instruction": None, "math": None, "coding": None, "performance": None}, "custom": {}}


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    category: str
    language: str
    difficulty: str
    prompt: str
    expected: dict[str, Any]
    skill: str = "general"


def dataset_hash() -> str: return sha256(TASK_PATH.read_bytes()).hexdigest()

def load_tasks(suite: str, languages: list[str], seed: int, counts: dict[str, int] | None = None) -> list[BenchmarkTask]:
    if suite not in SUITE_COUNTS: raise ValueError(f"Unknown suite: {suite}")
    raw = [BenchmarkTask(**item) for item in json.loads(TASK_PATH.read_text(encoding="utf-8")) if item["language"] in languages]
    rng, selected = random.Random(seed), []
    for category, limit in SUITE_COUNTS[suite].items():
        pool = [task for task in raw if task.category == category]
        rng.shuffle(pool)
        if not pool: continue
        if limit is None: selected.extend(pool)
        else: selected.extend(pool[:limit])
    if suite == "custom" and counts is not None:
        selected = []
        for category, limit in counts.items():
            pool = [task for task in raw if task.category == category]
            rng.shuffle(pool); selected.extend(pool[:limit])
    return selected
