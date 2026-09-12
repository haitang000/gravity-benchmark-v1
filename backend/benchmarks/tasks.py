from __future__ import annotations

import json
import random
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TASK_PATH = ROOT / "datasets" / "fixtures" / "core_tasks.json"
EXTENDED_TASK_PATH = ROOT / "datasets" / "fixtures" / "extended_tasks.json"

# Quotas keep quick runs useful on 2B-4B models while standard/full expose
# harder failure modes without duplicating a task to inflate the score.
SUITE_CONFIG = {
    "quick": {"difficulties": {"easy", "medium"}, "counts": {"instruction": 20, "math": 20, "coding": 10, "performance": 10}},
    "standard": {"difficulties": {"easy", "medium"}, "counts": {"instruction": 24, "math": 24, "coding": 18, "performance": 10}},
    "full": {"difficulties": {"easy", "medium", "hard"}, "counts": {"instruction": None, "math": None, "coding": None, "performance": None}},
    "custom": {"difficulties": {"easy", "medium", "hard"}, "counts": {}},
}


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    category: str
    language: str
    difficulty: str
    prompt: str
    expected: dict[str, Any]
    skill: str = "general"


def _task_paths() -> list[Path]:
    return [path for path in (TASK_PATH, EXTENDED_TASK_PATH) if path.exists()]


def dataset_hash() -> str:
    digest = sha256()
    for path in _task_paths():
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def task_catalog(languages: list[str] | None = None) -> list[BenchmarkTask]:
    allowed = set(languages or ["zh", "en"])
    return [
        BenchmarkTask(**item)
        for path in _task_paths()
        for item in json.loads(path.read_text(encoding="utf-8"))
        if item["language"] in allowed
    ]

def load_tasks(suite: str, languages: list[str], seed: int, counts: dict[str, int] | None = None) -> list[BenchmarkTask]:
    if suite not in SUITE_CONFIG: raise ValueError(f"Unknown suite: {suite}")
    config = SUITE_CONFIG[suite]
    raw = [task for task in task_catalog(languages) if task.difficulty in config["difficulties"]]
    rng, selected = random.Random(seed), []
    suite_counts = counts if suite == "custom" and counts is not None else config["counts"]
    for category, limit in suite_counts.items():
        pool = [task for task in raw if task.category == category]
        if suite == "standard":
            med_pool = [t for t in pool if t.difficulty == "medium"]
            easy_pool = [t for t in pool if t.difficulty == "easy"]
            rng.shuffle(med_pool)
            rng.shuffle(easy_pool)
            pool = med_pool + easy_pool
        else:
            rng.shuffle(pool)
        if not pool: continue
        if limit is None: selected.extend(pool)
        else: selected.extend(pool[:limit])
    return selected
