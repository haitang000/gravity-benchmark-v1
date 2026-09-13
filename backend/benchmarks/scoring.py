from __future__ import annotations

import ast
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from fractions import Fraction

from backend.benchmarks.tasks import BenchmarkTask


_THINK_TAGS = r"(?:think|thinking|thought|reasoning|analysis)"
_THINK_BLOCK = re.compile(rf"<\s*{_THINK_TAGS}\s*>.*?<\s*/\s*{_THINK_TAGS}\s*>", re.S | re.I)
_THINK_CLOSE = re.compile(rf"<\s*/\s*{_THINK_TAGS}\s*>", re.I)


def _strip_reasoning(text: str) -> str:
    cleaned = _THINK_BLOCK.sub("", text)
    matches = list(_THINK_CLOSE.finditer(cleaned))
    if matches: cleaned = cleaned[matches[-1].end():]
    return cleaned.strip()


def _strip_code(text: str) -> str:
    matches = list(re.finditer(r"```(?:python|json)?\s*(.*?)```", text, re.S | re.I))
    return matches[-1].group(1).strip() if matches else text.strip()


def _parse_json(text: str) -> Any:
    candidate = _strip_code(text)
    try: return json.loads(candidate)
    except json.JSONDecodeError: pass
    decoder = json.JSONDecoder(); found = None; position = 0
    while position < len(candidate):
        match = re.search(r"[{\[]", candidate[position:])
        if not match: break
        start = position + match.start()
        try:
            value, end = decoder.raw_decode(candidate[start:])
            found = value; position = start + end
        except json.JSONDecodeError: position = start + 1
    if found is not None: return found
    raise ValueError("no JSON value found")


def _number(text: str) -> str | None:
    boxed = re.findall(r"(?:\\boxed\{|答案[:：]?\s*)([-+]?\d+(?:\.\d+)?)", text)
    values = boxed or re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text)
    return values[-1] if values else None


def _instruction_checks(text: str, expected: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    checks: list[bool] = []
    if "json_schema" in expected:
        try:
            parsed = _parse_json(text)
            def valid_schema(value: Any, schema: Any) -> bool:
                if isinstance(schema, str):
                    if schema == "string": return isinstance(value, str)
                    if schema == "integer": return isinstance(value, int) and not isinstance(value, bool)
                    if schema == "boolean": return isinstance(value, bool)
                    if schema == "number": return isinstance(value, (int, float)) and not isinstance(value, bool)
                    if schema == "array": return isinstance(value, list)
                    if schema == "object": return isinstance(value, dict)
                    return False
                if not isinstance(schema, dict) or not isinstance(value, (dict, list)) and schema.get("type") in {"object", "array"}:
                    return False
                kind = schema.get("type")
                if kind and not valid_schema(value, kind): return False
                if kind == "object":
                    properties = schema.get("properties", {})
                    return all(key in value and valid_schema(value[key], child) for key, child in properties.items())
                if kind == "array":
                    return all(valid_schema(item, schema["items"]) for item in value) if "items" in schema else True
                return True
            schema = expected["json_schema"]
            if isinstance(schema, dict) and schema.get("type"):
                checks.append(valid_schema(parsed, schema))
            else:
                checks.extend(valid_schema(parsed.get(k), typ) for k, typ in schema.items())
        except (json.JSONDecodeError, ValueError, AttributeError, TypeError): checks.append(False)
    if "line_count" in expected: checks.append(len([x for x in text.splitlines() if x.strip()]) == expected["line_count"])
    if expected.get("lowercase"): checks.append(text == text.lower())
    if "regex" in expected: checks.append(bool(re.fullmatch(expected["regex"], text.strip())))
    if "contains" in expected: checks.extend(item in text for item in expected["contains"])
    if "forbid" in expected: checks.extend(item not in text for item in expected["forbid"])
    if "max_chars" in expected: checks.append(len(text.strip()) <= expected["max_chars"])
    if "min_chars" in expected: checks.append(len(text.strip()) >= expected["min_chars"])
    if "max_lines" in expected: checks.append(len([x for x in text.splitlines() if x.strip()]) <= expected["max_lines"])
    if "exact" in expected: checks.append(text.strip() == expected["exact"])
    if "exact_lines" in expected: checks.append([x.strip() for x in text.splitlines() if x.strip()] == expected["exact_lines"])
    if "line_prefixes" in expected:
        lines = [x for x in text.splitlines() if x.strip()]
        prefixes = expected["line_prefixes"]
        checks.append(len(lines) == len(prefixes))
        checks.extend(line.startswith(prefix) for line, prefix in zip(lines, prefixes))
    if "prefix" in expected: checks.extend(x.startswith(expected["prefix"]) for x in text.splitlines() if x.strip())
    if "json_keys" in expected:
        try: checks.extend(key in _parse_json(text) for key in expected["json_keys"])
        except (json.JSONDecodeError, ValueError, TypeError): checks.append(False)
    if "json_array" in expected:
        try:
            parsed = _parse_json(text); spec = expected["json_array"]
            checks.append(isinstance(parsed, list) and len(parsed) == spec["length"])
            for item in parsed:
                checks.extend(
                    isinstance(item.get(k), str) if typ == "string" else
                    isinstance(item.get(k), int) and not isinstance(item.get(k), bool) if typ == "integer" else
                    isinstance(item.get(k), bool) if typ == "boolean" else False
                    for k, typ in spec["fields"].items()
                )
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError): checks.append(False)
    return (sum(checks) / len(checks) if checks else 0), {"checks": checks}


def _answer_tail(text: str, expected: dict[str, Any]) -> str:
    lines = [x for x in text.splitlines() if x.strip()]
    if not lines: return ""
    width = expected.get("line_count") or (len(expected["exact_lines"]) if "exact_lines" in expected else 0) or (len(expected["line_prefixes"]) if "line_prefixes" in expected else 0) or 1
    return "\n".join(lines[-width:])


def _score_instruction(output: str, expected: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    cleaned = _strip_reasoning(output)
    value, details = _instruction_checks(cleaned, expected)
    if value >= 1: return value, details
    tail = _answer_tail(cleaned, expected)
    if tail and tail != cleaned:
        alt_value, alt_details = _instruction_checks(tail, expected)
        if alt_value > value: return alt_value, {"checks": alt_details["checks"], "answer_tail": True}
    return value, details


def _score_math(output: str, expected: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    found, answer = _number(output), expected["answer"]
    try:
        correct = found is not None and math.isclose(float(Fraction(found)), float(Fraction(answer)), rel_tol=1e-9, abs_tol=1e-9)
    except (ValueError, ZeroDivisionError): correct = found == answer
    return float(correct), {"found": found, "answer": answer, "topic": expected.get("topic")}


def _score_code(output: str, expected: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    code = _strip_code(_strip_reasoning(output))
    try: ast.parse(code)
    except SyntaxError as exc: return 0, {"kind": "syntax_error", "message": str(exc)}
    # Docker is the default isolation boundary. Refuse host execution for model-generated code.
    with tempfile.TemporaryDirectory(prefix="gravitybench-") as tmp:
        path = Path(tmp) / "solution.py"; path.write_text(code + "\n" + expected["tests"], encoding="utf-8")
        try:
            result = subprocess.run(["docker", "run", "--rm", "--network=none", "--cpus=0.5", "--memory=256m", "--pids-limit=64", "-v", f"{tmp}:/work:ro", "python:3.12-slim", "python", "/work/solution.py"], text=True, capture_output=True, timeout=8)
        except FileNotFoundError: return 0, {"kind": "sandbox_unavailable", "message": "Docker is required for coding evaluation"}
        except subprocess.TimeoutExpired: return 0, {"kind": "timeout"}
    if result.returncode == 0: return 1, {"kind": "passed"}
    return 0, {"kind": "test_failure", "stderr": result.stderr[-1000:]}


def score(task: BenchmarkTask, output: str) -> tuple[float, dict[str, Any]]:
    if task.category in {"instruction", "performance"}: return _score_instruction(output, task.expected)
    if task.category == "math": return _score_math(output, task.expected)
    if task.category == "coding": return _score_code(output, task.expected)
    return 0, {"kind": "unknown_category"}
