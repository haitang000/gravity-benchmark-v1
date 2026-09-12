from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any


def token_total(metrics: dict[str, Any]) -> int | None:
    input_tokens, output_tokens = metrics.get("input_tokens"), metrics.get("output_tokens")
    if input_tokens is None and output_tokens is None: return None
    return (input_tokens or 0) + (output_tokens or 0)

def aggregate(results: list[Any]) -> dict[str, Any]:
    groups: dict[str, list[Any]] = defaultdict(list)
    for result in results: groups[result.category].append(result)
    out: dict[str, Any] = {"total": {}}
    for name, rows in [("total", results), *groups.items()]:
        scores = [r.score for r in rows]
        latencies = [r.metrics.get("latency_ms") for r in rows if r.metrics.get("latency_ms") is not None]
        input_tokens = [r.metrics.get("input_tokens") for r in rows if r.metrics.get("input_tokens") is not None]
        output_tokens = [r.metrics.get("output_tokens") for r in rows if r.metrics.get("output_tokens") is not None]
        totals = [total for r in rows if (total := token_total(r.metrics)) is not None]
        speeds = [r.metrics.get("generation_tokens_per_second") for r in rows if r.metrics.get("generation_tokens_per_second")]
        out[name] = {"count": len(rows), "score": mean(scores) if scores else 0, "pass_rate": sum(r.passed for r in rows) / len(rows) if rows else 0, "avg_latency_ms": mean(latencies) if latencies else None, "p50_latency_ms": median(latencies) if latencies else None, "avg_input_tokens": mean(input_tokens) if input_tokens else None, "avg_output_tokens": mean(output_tokens) if output_tokens else None, "avg_total_tokens": mean(totals) if totals else None, "avg_generation_tokens_per_second": mean(speeds) if speeds else None}
    return out
