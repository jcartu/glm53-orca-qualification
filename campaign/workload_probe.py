#!/usr/bin/env python3
"""Matched capability, OpenAI-API, and bounded soak probes."""

from __future__ import annotations

import argparse
import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

import httpx


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
FIXTURE_PATH = REPO / "fixtures/workload_fixtures.json"
CAPABILITY_TASKS_PATH = REPO / "fixtures/capability/tasks.json"
CAPABILITY_TASKS_SHA256 = (
    "d6ac27099c621d9c82e95a125bf595e7fdf27f181ebe0088a6dccbbdd1becf10"
)
SANDBOX_IMAGE = (
    "sha256:f41ca8bb10bb3a125a50340d70d39ad4b7f5605f3fcb661bc992ed0bc4701a00"
)
SANDBOX_LABEL = "orcarouter.workload-code-sandbox=1"
CHAT_PATH = "/v1/chat/completions"
MAX_RAW_RESPONSE_BYTES = 16 * 1024 * 1024
SANDBOX_SEMAPHORE = threading.Semaphore(2)

SANDBOX_PROGRAM = r"""
import ast, builtins, contextlib, copy, glob, io, json, signal, sys
if glob.glob('/dev/nvidia*'):
    raise RuntimeError('code-scoring sandbox unexpectedly exposes GPU devices')
signal.alarm(8)
data = json.load(sys.stdin)
source = data['source']
if len(source) > 100000:
    raise ValueError('source exceeds 100000 characters')
tree = ast.parse(source)
allowed_imports = {'math','collections','heapq','bisect','itertools','functools','re','json','string','typing','posixpath','__future__'}
for node in ast.walk(tree):
    if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
        raise ValueError('dunder introspection is outside the task contract')
    if isinstance(node, ast.Name) and node.id.startswith('__'):
        raise ValueError('dunder names are outside the task contract')
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module or '']
        if any(name.split('.')[0] not in allowed_imports for name in names):
            raise ValueError('import outside the non-I/O allowlist')
for node in tree.body:
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        continue
    if not isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
        raise ValueError('module may contain functions, permitted imports, and assignments only')
def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split('.')[0] not in allowed_imports:
        raise ImportError('import outside allowlist')
    return builtins.__import__(name, globals, locals, fromlist, level)
names = 'abs all any bin bool bytearray bytes chr complex dict divmod enumerate filter float format frozenset hash hex int isinstance issubclass iter len list map max min next object oct ord pow print range repr reversed round set slice sorted str sum tuple type zip Exception ArithmeticError AssertionError IndexError KeyError OverflowError RuntimeError StopIteration TypeError ValueError ZeroDivisionError'.split()
safe = {name:getattr(builtins,name) for name in names}
safe['__import__'] = restricted_import
scope = {'__builtins__':safe, '__name__':'candidate'}
def equal(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    exec(compile(tree, 'candidate.py', 'exec'), scope)
    fn = scope.get('f')
    if not callable(fn):
        raise ValueError('no callable f was defined')
    rows = []
    for index, case in enumerate(data['tests']):
        try:
            actual = fn(*copy.deepcopy(case['args']))
            encoded = json.dumps(actual, allow_nan=False)
            if len(encoded) > 1000000:
                raise ValueError('result exceeds 1000000 encoded characters')
            actual = json.loads(encoded)
            rows.append({'case':index, 'passed':equal(actual, case['expected']), 'actual':actual})
        except Exception as error:
            rows.append({'case':index, 'passed':False, 'error':type(error).__name__ + ': ' + str(error)})
print(json.dumps({'cases':rows, 'passed':all(row['passed'] for row in rows), 'fraction':sum(row['passed'] for row in rows) / len(rows)}, allow_nan=False))
"""

REFUSAL_RE = re.compile(
    r"\b(?:i\s+(?:cannot|can't|won't)\s+(?:assist|help|provide|comply)|"
    r"i(?:'m|\s+am)\s+unable\s+to\s+(?:assist|help|provide|comply)|"
    r"cannot\s+(?:assist|help)\s+with\s+that)\b",
    re.IGNORECASE,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not name:
        raise ValueError("identifier has no safe filename characters")
    return name[:160]


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to rewrite historical result: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(value, output, indent=2, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(canonical(value) + "\n")
        output.flush()


def prepare_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False)
    (path / "records").mkdir()
    (path / "progress.jsonl").touch(exist_ok=False)
    return path


def validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("--base-url must be an absolute HTTP(S) origin")
    if parsed.username or parsed.password:
        raise ValueError("credentials are forbidden in --base-url")
    if parsed.query or parsed.fragment or parsed.path not in ("", "/"):
        raise ValueError(
            "--base-url must be an origin without a path, query, or fragment"
        )
    return value.rstrip("/")


def load_inputs() -> tuple[dict[str, Any], str, dict[str, Any], str]:
    fixture_bytes = FIXTURE_PATH.read_bytes()
    fixture = json.loads(fixture_bytes)
    if fixture.get("schema") != "orcarouter-workload-fixtures/v1":
        raise ValueError("unexpected workload fixture schema")
    tasks_bytes = CAPABILITY_TASKS_PATH.read_bytes()
    tasks_hash = sha256_bytes(tasks_bytes)
    if tasks_hash != CAPABILITY_TASKS_SHA256:
        raise ValueError(f"frozen capability task hash changed: {tasks_hash}")
    capability = json.loads(tasks_bytes)
    tasks = capability.get("tasks")
    if (
        capability.get("seed") != 20260915
        or not isinstance(tasks, list)
        or len(tasks) != 60
    ):
        raise ValueError("frozen capability suite is not the expected 60-task seed")
    ids = [task.get("id") for task in tasks]
    if len(set(ids)) != 60:
        raise ValueError("frozen capability suite has duplicate task ids")
    return fixture, sha256_bytes(fixture_bytes), capability, tasks_hash


def response_headers(response: httpx.Response) -> dict[str, str]:
    return {str(key): str(value) for key, value in response.headers.items()}


def decode_json(raw: str) -> tuple[object | None, str | None]:
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as error:
        return None, f"JSONDecodeError: {error}"


def post_json(
    client: httpx.Client,
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
    request_id: str,
) -> dict[str, Any]:
    url = base_url + CHAT_PATH
    started_wall = utc_now()
    started = time.monotonic()
    try:
        response = client.post(
            url,
            json=payload,
            timeout=timeout,
            headers={"x-request-id": safe_name(request_id)},
        )
        raw = response.content.decode("utf-8", errors="replace")
        parsed, decode_error = decode_json(raw)
        status = int(response.status_code)
        return {
            "schema": "orcarouter-workload-http-call/v1",
            "stream": False,
            "url": url,
            "request_id": request_id,
            "request": payload,
            "request_sha256": sha256_bytes(canonical(payload).encode()),
            "started_at": started_wall,
            "status": status,
            "response_headers": response_headers(response),
            "raw_body": raw,
            "response": parsed,
            "elapsed_seconds": time.monotonic() - started,
            "error": decode_error
            if 200 <= status < 300
            else f"HTTP {status}" + (f"; {decode_error}" if decode_error else ""),
            "ok": 200 <= status < 300 and decode_error is None,
        }
    except Exception as error:
        return {
            "schema": "orcarouter-workload-http-call/v1",
            "stream": False,
            "url": url,
            "request_id": request_id,
            "request": payload,
            "request_sha256": sha256_bytes(canonical(payload).encode()),
            "started_at": started_wall,
            "status": None,
            "response_headers": {},
            "raw_body": "",
            "response": None,
            "elapsed_seconds": time.monotonic() - started,
            "error": f"{type(error).__name__}: {error}",
            "ok": False,
        }


def append_fragment(target: dict[str, Any], key: str, value: object) -> None:
    if isinstance(value, str):
        target[key] = str(target.get(key) or "") + value


def stream_json(
    client: httpx.Client,
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
    request_id: str,
) -> dict[str, Any]:
    url = base_url + CHAT_PATH
    started_wall = utc_now()
    started = time.monotonic()
    raw_lines: list[str] = []
    raw_bytes = 0
    events: list[dict[str, Any]] = []
    malformed: list[dict[str, str]] = []
    response_ids: set[str] = set()
    finish_reasons: list[object] = []
    content: list[str] = []
    reasoning: list[str] = []
    tool_slots: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None
    done = False
    first_event_seconds: float | None = None
    first_token_seconds: float | None = None
    status: int | None = None
    headers: dict[str, str] = {}
    error_text: str | None = None
    parsed_error: object | None = None

    def consume(payload_text: str) -> None:
        nonlocal done, first_event_seconds, first_token_seconds, usage, error_text
        if payload_text == "[DONE]":
            done = True
            return
        try:
            event = json.loads(payload_text)
        except json.JSONDecodeError as error:
            malformed.append({"payload": payload_text[:2000], "error": str(error)})
            error_text = f"malformed SSE JSON: {error}"
            return
        if not isinstance(event, dict):
            malformed.append(
                {"payload": payload_text[:2000], "error": "event is not an object"}
            )
            error_text = "SSE event is not a JSON object"
            return
        events.append(event)
        if first_event_seconds is None:
            first_event_seconds = time.monotonic() - started
        event_id = event.get("id")
        if isinstance(event_id, str) and event_id:
            response_ids.add(event_id)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                continue
            if choice.get("finish_reason") is not None:
                finish_reasons.append(choice["finish_reason"])
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            visible = delta.get("content")
            thought = delta.get("reasoning_content")
            if thought is None:
                thought = delta.get("reasoning")
            if isinstance(visible, str):
                content.append(visible)
            if isinstance(thought, str):
                reasoning.append(thought)
            if first_token_seconds is None and any(
                value for value in (visible, thought, delta.get("tool_calls"))
            ):
                first_token_seconds = time.monotonic() - started
            calls = delta.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for position, fragment in enumerate(calls):
                if not isinstance(fragment, dict):
                    continue
                index = fragment.get("index", position)
                if not isinstance(index, int):
                    index = position
                slot = tool_slots.setdefault(
                    index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                append_fragment(slot, "id", fragment.get("id"))
                if isinstance(fragment.get("type"), str):
                    slot["type"] = fragment["type"]
                function = fragment.get("function")
                if isinstance(function, dict):
                    append_fragment(slot["function"], "name", function.get("name"))
                    append_fragment(
                        slot["function"], "arguments", function.get("arguments")
                    )

    try:
        with client.stream(
            "POST",
            url,
            json=payload,
            timeout=timeout,
            headers={
                "x-request-id": safe_name(request_id),
                "accept": "text/event-stream",
            },
        ) as response:
            status = int(response.status_code)
            headers = response_headers(response)
            if status != 200:
                body = response.read().decode("utf-8", errors="replace")
                raw_lines.append(body)
                parsed_error, decode_error = decode_json(body)
                error_text = f"HTTP {status}" + (
                    f"; {decode_error}" if decode_error else ""
                )
            else:
                for line in response.iter_lines():
                    raw_bytes += len(line.encode("utf-8", errors="replace")) + 1
                    if raw_bytes > MAX_RAW_RESPONSE_BYTES:
                        error_text = (
                            f"SSE response exceeded {MAX_RAW_RESPONSE_BYTES} bytes"
                        )
                        break
                    raw_lines.append(line)
                    if line.startswith("data:"):
                        consume(line[5:].lstrip())
                    if done or error_text:
                        break
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
    tool_calls = [tool_slots[index] for index in sorted(tool_slots)]
    visible_text = "".join(content)
    reasoning_text = "".join(reasoning)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": visible_text if visible_text or not tool_calls else None,
    }
    if reasoning_text:
        message["reasoning_content"] = reasoning_text
    if tool_calls:
        message["tool_calls"] = tool_calls
    reconstructed = {
        "id": sorted(response_ids)[0] if len(response_ids) == 1 else None,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reasons[-1] if finish_reasons else None,
            }
        ],
        "usage": usage,
    }
    return {
        "schema": "orcarouter-workload-http-call/v1",
        "stream": True,
        "url": url,
        "request_id": request_id,
        "request": payload,
        "request_sha256": sha256_bytes(canonical(payload).encode()),
        "started_at": started_wall,
        "status": status,
        "response_headers": headers,
        "raw_body": "\n".join(raw_lines),
        "raw_events": events,
        "malformed_events": malformed,
        "response_ids": sorted(response_ids),
        "done_received": done,
        "first_event_seconds": first_event_seconds,
        "first_token_seconds": first_token_seconds,
        "response": reconstructed if status == 200 else parsed_error,
        "elapsed_seconds": time.monotonic() - started,
        "error": error_text,
        "ok": status == 200 and error_text is None and done,
    }


def response_message(call: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if not call.get("ok"):
        return None, str(call.get("error") or f"HTTP {call.get('status')}")
    response = call.get("response")
    if not isinstance(response, dict):
        return None, "response JSON is not an object"
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None, "response must contain exactly one choice"
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        return None, "first choice has no message object"
    return choice["message"], None


def message_text(message: dict[str, Any]) -> tuple[str, str]:
    content = message.get("content")
    reasoning = message.get("reasoning")
    if reasoning is None:
        reasoning = message.get("reasoning_content")
    return (
        content if isinstance(content, str) else "",
        reasoning if isinstance(reasoning, str) else "",
    )


def finish_reason(call: dict[str, Any]) -> object | None:
    response = call.get("response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("finish_reason")


def response_usage(call: dict[str, Any]) -> dict[str, Any] | None:
    response = call.get("response")
    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
        return response["usage"]
    return None


def usage_issues(call: dict[str, Any]) -> list[str]:
    usage = response_usage(call)
    if usage is None:
        return ["usage unavailable"]
    issues: list[str] = []
    values: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            issues.append(f"{key} is not a nonnegative integer")
        else:
            values[key] = value
    if (
        len(values) == 3
        and values["total_tokens"]
        != values["prompt_tokens"] + values["completion_tokens"]
    ):
        issues.append("total_tokens does not equal prompt_tokens + completion_tokens")
    if values.get("prompt_tokens", 0) <= 0:
        issues.append("prompt_tokens is not positive")
    if values.get("completion_tokens", 0) <= 0:
        issues.append("completion_tokens is not positive")
    return issues


def repetition_evidence(text: str) -> dict[str, Any] | None:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    shingles: dict[str, int] = {}
    for offset in range(0, max(len(normalized) - 60, 0), 30):
        shingle = normalized[offset : offset + 60]
        shingles[shingle] = shingles.get(shingle, 0) + 1
        if shingles[shingle] >= 6:
            return {
                "detector": "60-character shingle at 30-character stride",
                "threshold": 6,
                "count": shingles[shingle],
                "shingle": shingle,
            }
    lines: dict[str, int] = {}
    for line in text.splitlines():
        normalized_line = re.sub(r"\s+", " ", line).strip().lower()
        if len(normalized_line) < 24:
            continue
        lines[normalized_line] = lines.get(normalized_line, 0) + 1
        if lines[normalized_line] >= 5:
            return {
                "detector": "normalized repeated line",
                "threshold": 5,
                "count": lines[normalized_line],
                "line": normalized_line[:240],
            }
    return None


def equal_json(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            equal_json(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            equal_json(a, b) for a, b in zip(left, right)
        )
    return left == right


def parse_json_answer(text: str) -> object:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*\n?", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as original:
        decoder = json.JSONDecoder()
        positions = [index for index, char in enumerate(stripped) if char in "[{"]
        for position in reversed(positions):
            try:
                value, end = decoder.raw_decode(stripped[position:])
            except json.JSONDecodeError:
                continue
            trailing = stripped[position + end :].strip()
            if not trailing or trailing == "```":
                return value
        raise original


def extract_int_candidates(text: str) -> list[int]:
    patterns = [
        r"(?:Final\s+answer|Answer)\s*:\s*(-?\d[\d,]*)",
        r"\\boxed\{\s*(-?\d[\d,]*)\s*\}",
        r"\*\*(-?\d[\d,]*)\*\*",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return [int(matches[-1].replace(",", ""))]
    lines = [line for line in text.strip().splitlines() if line.strip()][-3:]
    candidates: list[int] = []
    for line in lines:
        for match in re.findall(
            r"(?<![\d-])(-?(?:\d{1,3}(?:,\d{3})+|\d+))(?![\d-])", line
        ):
            candidates.append(int(match.replace(",", "")))
    return candidates


def extract_string_answer(text: str) -> str | None:
    matches = re.findall(
        r"(?:Final\s+answer|Answer)\s*:\s*([A-Za-z][A-Za-z0-9_-]*)", text, re.IGNORECASE
    )
    if matches:
        return matches[-1]
    stripped = text.strip().strip("`*_.,;: ")
    return stripped if re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", stripped) else None


def extract_code(text: str) -> str:
    blocks = re.findall(
        r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE
    )
    return (blocks[0] if blocks else text).strip()


def fibonacci_mod(number: int, modulus: int = 998244353) -> int:
    def pair(value: int) -> tuple[int, int]:
        if value == 0:
            return 0, 1
        a, b = pair(value // 2)
        c = a * ((2 * b - a) % modulus) % modulus
        d = (a * a + b * b) % modulus
        return (d, (c + d) % modulus) if value % 2 else (c, d)

    return pair(number)[0]


def effective_code_tests(
    task_id: str, verifier: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    corrections: list[str] = []
    if task_id == "C01":
        corrections.append(
            "Replaced the original non-assertion for f(10**18) with an exact independently computed modular oracle."
        )
        return [
            {"args": [0], "expected": 0},
            {"args": [1], "expected": 1},
            {"args": [10], "expected": 55},
            {"args": [1000], "expected": 548571675},
            {"args": [10**18], "expected": fibonacci_mod(10**18)},
        ], corrections
    return copy.deepcopy(verifier.get("tests") or []), corrections


def cleanup_owned_container(name: str) -> str | None:
    try:
        inspected = subprocess.run(
            ["docker", "inspect", name], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"container inspection failed: {type(error).__name__}: {error}"
    if inspected.returncode:
        try:
            remaining = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-a",
                    "--filter",
                    f"name=^{name}$",
                    "--format",
                    "{{.ID}}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return f"could not establish container absence: {type(error).__name__}: {error}"
        if not remaining.stdout.strip():
            return None
        return "container inspection failed while the named container still exists"
    try:
        container = json.loads(inspected.stdout)[0]
    except (json.JSONDecodeError, IndexError, TypeError) as error:
        return f"container inspection was not parseable: {error}"
    labels = (container.get("Config") or {}).get("Labels") or {}
    if labels.get("orcarouter.workload-code-sandbox") != "1":
        return "refused cleanup because the container ownership label did not match"
    try:
        removed = subprocess.run(
            ["docker", "rm", "-f", str(container.get("Id") or name)],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"owned container cleanup failed: {type(error).__name__}: {error}"
    if removed.returncode:
        return "owned container cleanup returned nonzero: " + removed.stderr[-1000:]
    return None


def score_code(text: str, tests: list[dict[str, Any]]) -> dict[str, Any]:
    source = extract_code(text)
    request_bytes = canonical({"source": source, "tests": tests}).encode()
    scope = (
        "Generated Python executes only inside a pinned-image runc container with no network, no mounts, "
        "no GPU runtime/devices, a read-only root, dropped capabilities, no-new-privileges, an unprivileged user, "
        "bounded CPU/PIDs/memory/time/output, and an AST/builtin/import allowlist. There is no host-execution fallback."
    )
    if not source:
        return {
            "passed": False,
            "fraction": 0.0,
            "error": "no code found",
            "sandbox_scope": scope,
        }
    if len(source) > 100000 or len(request_bytes) > 2 * 1024 * 1024:
        return {
            "passed": False,
            "fraction": 0.0,
            "error": "code-scoring input limit exceeded",
            "sandbox_scope": scope,
        }
    name = "orca-workload-code-" + uuid.uuid4().hex[:16]
    command = [
        "docker",
        "run",
        "--rm",
        "--init",
        "--name",
        name,
        "--pull=never",
        "--log-driver",
        "none",
        "--label",
        SANDBOX_LABEL,
        "--runtime",
        "runc",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "65534:65534",
        "--pids-limit",
        "32",
        "--memory",
        "256m",
        "--memory-swap",
        "256m",
        "--cpus",
        "1",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "-i",
        "--env",
        "NVIDIA_VISIBLE_DEVICES=void",
        "--entrypoint",
        "python3",
        SANDBOX_IMAGE,
        "-c",
        SANDBOX_PROGRAM,
    ]
    process: subprocess.Popen[bytes] | None = None
    result: dict[str, Any]
    with SANDBOX_SEMAPHORE:
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = process.communicate(request_bytes, timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate(timeout=5)
                result = {
                    "passed": False,
                    "fraction": 0.0,
                    "limit_reason": "wall_time",
                    "exit_code": process.returncode,
                    "infrastructure_error": False,
                }
            else:
                if len(stdout) + len(stderr) > 2 * 1024 * 1024:
                    result = {
                        "passed": False,
                        "fraction": 0.0,
                        "limit_reason": "captured_output_bytes",
                        "exit_code": process.returncode,
                    }
                elif process.returncode:
                    stderr_text = stderr.decode(errors="replace")[-4000:]
                    result = {
                        "passed": False,
                        "fraction": 0.0,
                        "error": stderr_text,
                        "exit_code": process.returncode,
                        "infrastructure_error": (
                            process.returncode in (125, 126, 127)
                            or "unexpectedly exposes GPU devices" in stderr_text
                        ),
                    }
                else:
                    try:
                        parsed = json.loads(stdout)
                    except json.JSONDecodeError as error:
                        result = {
                            "passed": False,
                            "fraction": 0.0,
                            "error": f"sandbox JSONDecodeError: {error}",
                            "infrastructure_error": True,
                        }
                    else:
                        result = (
                            parsed
                            if isinstance(parsed, dict)
                            else {
                                "passed": False,
                                "fraction": 0.0,
                                "error": "sandbox result was not an object",
                                "infrastructure_error": True,
                            }
                        )
        except (OSError, subprocess.SubprocessError) as error:
            result = {
                "passed": False,
                "fraction": 0.0,
                "error": f"{type(error).__name__}: {error}",
                "infrastructure_error": True,
            }
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            cleanup_error = cleanup_owned_container(name)
    if cleanup_error:
        result = {
            **result,
            "passed": False,
            "infrastructure_error": True,
            "cleanup_error": cleanup_error,
        }
    result["sandbox_scope"] = scope
    result["sandbox_image"] = SANDBOX_IMAGE
    result["source_sha256"] = sha256_bytes(source.encode())
    return result


def score_answer(
    text: str, verifier: dict[str, Any], task_id: str = ""
) -> dict[str, Any]:
    verifier_type = verifier.get("type")
    if verifier_type == "int":
        candidates = extract_int_candidates(text)
        answer = verifier.get("answer")
        return {
            "passed": answer in candidates,
            "candidates": candidates,
            "expected": answer,
        }
    if verifier_type == "string":
        actual = extract_string_answer(text)
        expected = str(verifier.get("answer"))
        return {
            "passed": actual is not None and actual.casefold() == expected.casefold(),
            "actual": actual,
            "expected": expected,
        }
    if verifier_type == "json":
        try:
            actual = parse_json_answer(text)
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            return {
                "passed": False,
                "expected": verifier.get("answer"),
                "parse_error": str(error),
            }
        return {
            "passed": equal_json(actual, verifier.get("answer")),
            "actual": actual,
            "expected": verifier.get("answer"),
        }
    if verifier_type == "exact_text":
        actual = text.strip()
        expected = str(verifier.get("answer"))
        return {"passed": actual == expected, "actual": actual, "expected": expected}
    if verifier_type == "code":
        tests, corrections = effective_code_tests(task_id, verifier)
        result = score_code(text, tests)
        result["effective_tests"] = tests
        result["oracle_corrections"] = corrections
        return result
    return {"passed": False, "error": f"unsupported verifier type {verifier_type!r}"}


def schema_errors(value: object, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(expected_type, True)
    if not type_ok:
        return [f"{path}: expected {expected_type}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value is outside enum")
    if expected_type == "object" and isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                errors.append(f"{path}: missing required key {key!r}")
        if schema.get("additionalProperties") is False:
            for key in value.keys() - properties.keys():
                errors.append(f"{path}: unexpected key {key!r}")
        for key, child in properties.items():
            if key in value and isinstance(child, dict):
                errors.extend(schema_errors(value[key], child, f"{path}.{key}"))
    if expected_type == "array" and isinstance(value, list):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems")
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(schema_errors(item, item_schema, f"{path}[{index}]"))
    return errors


def generation_body(
    model: str,
    generation: dict[str, Any],
    messages: list[dict[str, Any]],
    max_tokens: int | None = None,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": copy.deepcopy(messages),
        **copy.deepcopy(generation),
    }
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    return body


def normalized_tool_arguments(
    entry: dict[str, Any], arguments: dict[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(arguments)
    for field in entry.get("case_insensitive_fields") or []:
        if isinstance(result.get(field), str):
            result[field] = result[field].casefold()
    return result


def tool_key(
    name: str, arguments: dict[str, Any], entry: dict[str, Any] | None = None
) -> str:
    normalized = normalized_tool_arguments(entry or {}, arguments)
    return name + ":" + canonical(normalized)


def resolve_tool(
    task: dict[str, Any], call: dict[str, Any]
) -> tuple[dict[str, Any], str | None, str | None]:
    if not isinstance(call, dict) or call.get("type") != "function":
        return (
            {"error": "invalid tool call object"},
            None,
            "tool call is not a function object",
        )
    call_id = call.get("id")
    function = call.get("function")
    if not isinstance(call_id, str) or not call_id or not isinstance(function, dict):
        return (
            {"error": "invalid tool call id/function"},
            None,
            "tool call lacks a valid id/function",
        )
    name = function.get("name")
    raw_arguments = function.get("arguments")
    if not isinstance(name, str) or not isinstance(raw_arguments, str):
        return (
            {"error": "invalid function name/arguments"},
            None,
            "tool name or arguments is not a string",
        )
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        return (
            {"error": "arguments are not valid JSON"},
            None,
            f"invalid tool argument JSON: {error}",
        )
    if not isinstance(arguments, dict):
        return (
            {"error": "arguments must be an object"},
            None,
            "tool arguments are not an object",
        )
    for entry in task.get("tool_responses") or []:
        if entry.get("name") != name:
            continue
        expected = normalized_tool_arguments(entry, entry.get("arguments") or {})
        actual = normalized_tool_arguments(entry, arguments)
        if equal_json(actual, expected):
            return (
                copy.deepcopy(entry.get("result")),
                tool_key(name, arguments, entry),
                None,
            )
    return (
        {"error": "unknown deterministic fixture or invalid arguments"},
        tool_key(name, arguments),
        "tool call did not match a fixture",
    )


def run_tool_workflow(
    client: httpx.Client,
    base_url: str,
    model: str,
    generation: dict[str, Any],
    task: dict[str, Any],
    timeout: float,
    request_prefix: str,
) -> dict[str, Any]:
    messages = copy.deepcopy(task["messages"])
    calls: list[dict[str, Any]] = []
    issues: list[str] = []
    observed: list[str] = []
    final_message: dict[str, Any] | None = None
    final_finish: object | None = None
    for turn in range(int(task.get("max_turns") or 6)):
        body = generation_body(
            model,
            generation,
            messages,
            int(task.get("max_tokens") or generation["max_tokens"]),
        )
        body.update({"tools": copy.deepcopy(task["tools"]), "tool_choice": "auto"})
        if task.get("stream"):
            body.update({"stream": True, "stream_options": {"include_usage": True}})
            call = stream_json(
                client, base_url, body, timeout, f"{request_prefix}-turn-{turn}"
            )
        else:
            call = post_json(
                client, base_url, body, timeout, f"{request_prefix}-turn-{turn}"
            )
        calls.append(call)
        if task.get("stream"):
            if len(call.get("response_ids") or []) != 1:
                issues.append(f"turn {turn}: SSE response id was absent or changed")
            if not call.get("raw_events"):
                issues.append(f"turn {turn}: SSE stream contained no JSON data events")
            if call.get("malformed_events"):
                issues.append(f"turn {turn}: SSE stream contained malformed events")
        message, error = response_message(call)
        if error:
            issues.append(error)
            break
        assert message is not None
        turn_finish = finish_reason(call)
        if turn_finish == "length":
            issues.append("generation truncated during tool workflow")
            break
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            final_message = message
            final_finish = turn_finish
            messages.append(copy.deepcopy(message))
            break
        if not isinstance(tool_calls, list):
            issues.append("tool_calls is not a list")
            break
        if turn_finish != "tool_calls":
            issues.append(f"tool calls were emitted with finish_reason={turn_finish!r}")
        messages.append(copy.deepcopy(message))
        for tool_call in tool_calls:
            result, key, issue = resolve_tool(task, tool_call)
            if key is not None:
                observed.append(key)
            if issue:
                issues.append(issue)
            call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
            if not isinstance(call_id, str) or not call_id:
                issues.append("cannot submit a result for a tool call without an id")
                continue
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": canonical(result),
                }
            )
    if final_message is None and not any("truncated" in issue for issue in issues):
        issues.append("no final answer within the bounded tool-turn limit")
    required = sorted(task.get("required_calls") or [])
    if sorted(observed) != required:
        issues.append(
            f"tool calls differed: expected {required!r}, observed {sorted(observed)!r}"
        )
    return {
        "calls": calls,
        "transcript": messages,
        "protocol_issues": issues,
        "observed_tool_calls": observed,
        "final_message": final_message,
        "finish_reason": final_finish,
    }


def classify_task(record: dict[str, Any]) -> str:
    if record.get("runtime_error"):
        return "runtime_error"
    if record.get("truncated"):
        return "truncation"
    if record.get("protocol_issues"):
        return "protocol_error"
    if record.get("overrefusal_candidate"):
        return "over_refusal"
    if not record.get("semantic_passed"):
        return "knowledge_or_task_failure"
    return "pass"


def capability_task(
    args: argparse.Namespace,
    client: httpx.Client,
    generation: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(task["id"])
    record: dict[str, Any] = {
        "schema": "orcarouter-workload-capability-record/v1",
        "id": task_id,
        "category": task.get("cat") or task.get("category"),
        "source": task.get("source"),
        "started_at": utc_now(),
        "semantic_passed": False,
        "runtime_error": False,
        "truncated": False,
        "protocol_issues": [],
        "calls": [],
    }
    kind = task.get("kind", "standard")
    final_message: dict[str, Any] | None = None
    final_finish: object | None = None
    if kind == "tool_workflow":
        workflow = run_tool_workflow(
            client,
            args.base_url,
            args.model,
            generation,
            task,
            float(args.fixture["capability"]["request_timeout_seconds"]),
            f"cap-{task_id}",
        )
        record.update(
            {
                key: value
                for key, value in workflow.items()
                if key not in ("final_message",)
            }
        )
        final_message = workflow["final_message"]
        final_finish = workflow["finish_reason"]
    else:
        messages = task.get("messages") or [{"role": "user", "content": task["prompt"]}]
        body = generation_body(
            args.model,
            generation,
            messages,
            int(task.get("max_tokens") or generation["max_tokens"]),
        )
        if task.get("reasoning_effort"):
            body["reasoning_effort"] = task["reasoning_effort"]
        if kind == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": copy.deepcopy(task["json_schema"]),
            }
        call = post_json(
            client,
            args.base_url,
            body,
            float(args.fixture["capability"]["request_timeout_seconds"]),
            f"cap-{task_id}",
        )
        record["calls"] = [call]
        final_message, error = response_message(call)
        final_finish = finish_reason(call)
        if error:
            record["runtime_error"] = True
            record["error"] = error
        if final_finish == "length":
            record["truncated"] = True
        elif final_finish not in (None, "stop"):
            record["protocol_issues"].append(
                f"unexpected final finish_reason={final_finish!r}"
            )
        if final_finish is None and final_message is not None:
            record["protocol_issues"].append("final finish_reason is unavailable")
    if kind == "tool_workflow":
        record["runtime_error"] = any(not call.get("ok") for call in record["calls"])
        record["truncated"] = any(
            finish_reason(call) == "length" for call in record["calls"]
        )
        if final_finish is None and final_message is not None:
            record["protocol_issues"].append("final finish_reason is unavailable")
        elif final_finish not in (None, "stop"):
            record["protocol_issues"].append(
                f"unexpected final finish_reason={final_finish!r}"
            )
    if final_message is not None:
        visible, reasoning = message_text(final_message)
        record["visible_answer"] = visible
        record["reasoning"] = reasoning
        record["finish_reason"] = final_finish
        record["usage"] = response_usage(record["calls"][-1])
        record["repetition_candidate"] = repetition_evidence(visible + "\n" + reasoning)
        if final_message.get("tool_calls"):
            record["protocol_issues"].append("final answer still contains tool calls")
        score = score_answer(visible, task["verifier"], task_id)
        record["verifier"] = score
        record["semantic_passed"] = score.get("passed") is True
        if score.get("infrastructure_error"):
            record["runtime_error"] = True
        if task.get("benign_overrefusal"):
            record["overrefusal_candidate"] = not record["semantic_passed"] and bool(
                REFUSAL_RE.search(visible)
            )
    record["failure_kind"] = classify_task(record)
    record["task_passed"] = (
        record["semantic_passed"]
        and not record["runtime_error"]
        and not record["truncated"]
        and not record["protocol_issues"]
        and not record.get("overrefusal_candidate")
    )
    record["finished_at"] = utc_now()
    return record


def run_capability(args: argparse.Namespace, capability: dict[str, Any]) -> int:
    section = args.fixture["capability"]
    tasks: list[dict[str, Any]] = []
    for original in capability["tasks"]:
        task = copy.deepcopy(original)
        task.update(
            {"kind": "standard", "source": "fixtures/capability/tasks.json"}
        )
        tasks.append(task)
    for original in section["extra_tasks"]:
        task = copy.deepcopy(original)
        task["source"] = "fixtures/workload_fixtures.json"
        tasks.append(task)
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("capability task ids are not unique")
    save_json(
        args.out_dir / "input-contract.json",
        {
            "schema": "orcarouter-workload-input-contract/v1",
            "mode": "capability",
            "label": args.label,
            "model": args.model,
            "base_url": args.base_url,
            "fixture_sha256": args.fixture_sha256,
            "capability_tasks_sha256": args.capability_tasks_sha256,
            "task_count": len(tasks),
            "generation": section["generation"],
            "concurrency": section["concurrency"],
            "sandbox": {
                "image": SANDBOX_IMAGE,
                "network": "none",
                "runtime": "runc",
                "gpu_devices": "void",
                "host_execution_fallback": False,
            },
            "oracle_audit": [
                "The original verify_eval.py executes generated Python directly on the host; this probe instead uses the bounded Docker sandbox and has no host fallback.",
                "C01's original f(10**18) check asserted only that abs(result)>=0; this probe uses an exact fast-doubling modular oracle.",
                "The public capability fixture contains the audited in-domain C05 input 2147483646 and corrected C09 keys 3/10/21; historical as-run inputs and response records remain in published evidence.",
                "Python, py, bare fenced, and unfenced code are accepted; answer markers, boxed/bold answers, and terminal numeric lines are tolerated without accepting arbitrary earlier numbers.",
                "Missing usage is retained as unavailable rather than rewritten as zero; missing/duplicate tasks and malformed responses fail closed.",
            ],
        },
    )
    started = utc_now()
    records: list[dict[str, Any]] = []
    limits = httpx.Limits(
        max_connections=int(section["concurrency"]),
        max_keepalive_connections=int(section["concurrency"]),
    )
    with httpx.Client(trust_env=False, limits=limits) as client:
        with ThreadPoolExecutor(max_workers=int(section["concurrency"])) as pool:
            futures = {
                pool.submit(
                    capability_task, args, client, section["generation"], task
                ): task["id"]
                for task in tasks
            }
            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = {
                        "schema": "orcarouter-workload-capability-record/v1",
                        "id": task_id,
                        "semantic_passed": False,
                        "runtime_error": True,
                        "truncated": False,
                        "protocol_issues": [],
                        "failure_kind": "runtime_error",
                        "task_passed": False,
                        "error": f"harness exception: {type(error).__name__}: {error}",
                        "calls": [],
                        "finished_at": utc_now(),
                    }
                records.append(record)
                record_path = args.out_dir / "records" / (safe_name(task_id) + ".json")
                save_json(record_path, record)
                append_jsonl(
                    args.out_dir / "progress.jsonl",
                    {
                        "completed": len(records),
                        "expected": len(tasks),
                        "id": task_id,
                        "task_passed": record.get("task_passed"),
                        "failure_kind": record.get("failure_kind"),
                        "record": str(record_path),
                    },
                )
                print(
                    canonical(
                        {
                            "completed": len(records),
                            "expected": len(tasks),
                            "id": task_id,
                            "passed": record.get("task_passed"),
                        }
                    ),
                    flush=True,
                )
    by_category: dict[str, dict[str, int]] = {}
    failure_kinds: dict[str, int] = {}
    for record in records:
        category = str(record.get("category") or "unknown")
        bucket = by_category.setdefault(category, {"passed": 0, "failed": 0})
        bucket["passed" if record.get("task_passed") else "failed"] += 1
        kind = str(record.get("failure_kind"))
        failure_kinds[kind] = failure_kinds.get(kind, 0) + 1
    passed = len(records) == len(tasks) and all(
        record.get("task_passed") is True for record in records
    )
    usage_available = sum(
        bool(record.get("calls")) and response_usage(record["calls"][-1]) is not None
        for record in records
    )
    usage_unavailable = len(records) - usage_available
    status = (
        "fail"
        if not passed
        else "pass_with_usage_unavailable"
        if usage_unavailable
        else "pass"
    )
    summary = {
        "schema": "orcarouter-workload-capability-summary/v1",
        "status": status,
        "label": args.label,
        "model": args.model,
        "started_at": started,
        "finished_at": utc_now(),
        "expected": len(tasks),
        "completed": len(records),
        "passed_tasks": sum(record.get("task_passed") is True for record in records),
        "failed_tasks": sum(
            record.get("task_passed") is not True for record in records
        ),
        "by_category": by_category,
        "failure_kinds": failure_kinds,
        "usage_available": usage_available,
        "usage_unavailable": usage_unavailable,
        "passed": passed,
        "results": [
            {
                "id": record.get("id"),
                "category": record.get("category"),
                "task_passed": record.get("task_passed"),
                "failure_kind": record.get("failure_kind"),
                "record": str(
                    args.out_dir
                    / "records"
                    / (safe_name(str(record.get("id"))) + ".json")
                ),
            }
            for record in sorted(records, key=lambda row: str(row.get("id")))
        ],
        "scope": "The frozen 60-task deterministic suite plus explicit benign reasoning/tool/schema/language/over-refusal fixtures. No language-model judge is used.",
    }
    save_json(args.out_dir / "summary.json", summary)
    print(
        canonical({key: value for key, value in summary.items() if key != "results"}),
        flush=True,
    )
    return 0 if passed else 1


def validate_completion_case(
    call: dict[str, Any], case: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    message, error = response_message(call)
    visible = reasoning = ""
    score: dict[str, Any] = {"passed": False, "error": error or "no response"}
    if error:
        issues.append(error)
    else:
        assert message is not None
        visible, reasoning = message_text(message)
        if message.get("tool_calls"):
            issues.append("unexpected tool calls in final completion")
        if case.get("kind") == "stop":
            stops = case.get("stop") or []
            actual_stop = call["response"]["choices"][0].get("stop_reason")
            score = {
                "passed": actual_stop in stops,
                "actual_stop_reason": actual_stop,
                "expected_stop_sequences": stops,
                "channels_with_output": [
                    name
                    for name, text in (("content", visible), ("reasoning", reasoning))
                    if text
                ],
            }
            for stop in stops:
                if stop in visible or stop in reasoning:
                    issues.append("stop sequence leaked into returned output")
                    score["passed"] = False
        else:
            score = score_answer(visible, case["verifier"], str(case["id"]))
        if case.get("kind") == "json_schema" and score.get("actual") is not None:
            validation = schema_errors(score["actual"], case["json_schema"]["schema"])
            if validation:
                issues.extend(validation)
                score["passed"] = False
                score["schema_errors"] = validation
    expected_finish = case.get("require_finish")
    actual_finish = finish_reason(call)
    if expected_finish is not None and actual_finish != expected_finish:
        issues.append(
            f"finish_reason expected {expected_finish!r}, got {actual_finish!r}"
        )
    if case.get("require_usage"):
        issues.extend(usage_issues(call))
    if call.get("stream"):
        if not call.get("done_received"):
            issues.append("SSE stream ended without [DONE]")
        if len(call.get("response_ids") or []) != 1:
            issues.append("SSE response id was absent or changed")
        if call.get("malformed_events"):
            issues.append("SSE stream contained malformed events")
        if not call.get("raw_events"):
            issues.append("SSE stream contained no JSON data events")
    return {
        "visible_answer": visible,
        "reasoning": reasoning,
        "finish_reason": actual_finish,
        "usage": response_usage(call),
        "verifier": score,
        "repetition_candidate": repetition_evidence(visible + "\n" + reasoning),
    }, issues


def api_case(
    args: argparse.Namespace, client: httpx.Client, case: dict[str, Any]
) -> dict[str, Any]:
    section = args.fixture["api"]
    record: dict[str, Any] = {
        "schema": "orcarouter-workload-api-record/v1",
        "id": case["id"],
        "kind": case["kind"],
        "started_at": utc_now(),
        "calls": [],
        "issues": [],
        "passed": False,
    }
    timeout = float(section["request_timeout_seconds"])
    kind = case["kind"]
    if kind == "malformed":
        body = {"model": args.model, **copy.deepcopy(case["body"])}
        call = post_json(client, args.base_url, body, timeout, f"api-{case['id']}")
        record["calls"] = [call]
        response = call.get("response")
        content_type = str(
            (call.get("response_headers") or {}).get("content-type") or ""
        )
        error_object = response.get("error") if isinstance(response, dict) else None
        checks = {
            "expected_4xx": call.get("status") in case["expected_statuses"],
            "json_content_type": "json" in content_type.casefold(),
            "json_error_object": (
                isinstance(error_object, dict)
                and isinstance(error_object.get("message"), str)
                and bool(error_object["message"].strip())
            ),
            "no_transport_error": call.get("status") is not None,
        }
        record["checks"] = checks
        record["passed"] = all(checks.values())
        if not record["passed"]:
            record["issues"].append(
                "malformed input did not produce the expected structured 4xx error"
            )
    elif kind == "identity":
        nonces = list(case["request_ids"])
        results: list[dict[str, Any]] = []

        def one(nonce: str) -> dict[str, Any]:
            body = generation_body(
                args.model,
                section["generation"],
                [
                    {
                        "role": "user",
                        "content": f"Return only JSON with request_id exactly `{nonce}`.",
                    }
                ],
            )
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": copy.deepcopy(case["json_schema"]),
            }
            call = post_json(
                client, args.base_url, body, timeout, f"api-identity-{nonce}"
            )
            message, error = response_message(call)
            visible = reasoning = ""
            parsed: object | None = None
            issues: list[str] = []
            if error:
                issues.append(error)
            else:
                assert message is not None
                visible, reasoning = message_text(message)
                try:
                    parsed = parse_json_answer(visible)
                except (json.JSONDecodeError, ValueError, TypeError) as parse_error:
                    issues.append(f"JSON parse error: {parse_error}")
            expected = {"request_id": nonce}
            if not equal_json(parsed, expected):
                issues.append(
                    f"identity mismatch: expected {expected!r}, got {parsed!r}"
                )
            if finish_reason(call) != case["require_finish"]:
                issues.append(f"finish_reason={finish_reason(call)!r}")
            if case.get("require_usage"):
                issues.extend(usage_issues(call))
            foreign = sorted(
                other
                for other in nonces
                if other != nonce and other in (visible + "\n" + reasoning)
            )
            if foreign:
                issues.append(f"foreign request identities leaked: {foreign!r}")
            return {
                "nonce": nonce,
                "call": call,
                "visible_answer": visible,
                "reasoning": reasoning,
                "parsed": parsed,
                "issues": issues,
                "passed": not issues,
            }

        with ThreadPoolExecutor(max_workers=int(case["concurrency"])) as pool:
            futures = {pool.submit(one, nonce): nonce for nonce in nonces}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append(
                        {
                            "nonce": futures[future],
                            "issues": [
                                f"harness exception: {type(error).__name__}: {error}"
                            ],
                            "passed": False,
                            "call": None,
                        }
                    )
        results.sort(key=lambda row: row["nonce"])
        record["identity_results"] = results
        record["calls"] = [
            row["call"] for row in results if isinstance(row.get("call"), dict)
        ]
        record["passed"] = len(results) == len(nonces) and all(
            row["passed"] for row in results
        )
        record["issues"] = [issue for row in results for issue in row["issues"]]
    elif kind == "tool_workflow":
        workflow = run_tool_workflow(
            client,
            args.base_url,
            args.model,
            section["generation"],
            case,
            timeout,
            f"api-{case['id']}",
        )
        record["calls"] = workflow["calls"]
        record["transcript"] = workflow["transcript"]
        record["observed_tool_calls"] = workflow["observed_tool_calls"]
        record["issues"].extend(workflow["protocol_issues"])
        final_message = workflow["final_message"]
        if final_message is None:
            record["issues"].append("tool workflow has no final message")
        else:
            visible, reasoning = message_text(final_message)
            score = score_answer(visible, case["verifier"], str(case["id"]))
            record.update(
                {
                    "visible_answer": visible,
                    "reasoning": reasoning,
                    "verifier": score,
                    "finish_reason": workflow["finish_reason"],
                    "usage": response_usage(record["calls"][-1]),
                }
            )
            if not score.get("passed"):
                record["issues"].append(
                    "final tool-workflow answer failed its deterministic oracle"
                )
            if workflow["finish_reason"] != case.get("require_finish"):
                record["issues"].append(
                    f"final finish_reason={workflow['finish_reason']!r}"
                )
        if case.get("require_usage"):
            for index, call in enumerate(record["calls"]):
                record["issues"].extend(
                    f"call {index}: {issue}" for issue in usage_issues(call)
                )
        record["passed"] = not record["issues"]
    else:
        body = generation_body(args.model, section["generation"], case["messages"])
        if case.get("tools"):
            body.update({"tools": copy.deepcopy(case["tools"]), "tool_choice": "none"})
        if kind == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": copy.deepcopy(case["json_schema"]),
            }
        if kind == "stop":
            body["stop"] = copy.deepcopy(case["stop"])
        if case.get("stream"):
            body.update({"stream": True, "stream_options": {"include_usage": True}})
            call = stream_json(
                client, args.base_url, body, timeout, f"api-{case['id']}"
            )
        else:
            call = post_json(client, args.base_url, body, timeout, f"api-{case['id']}")
        record["calls"] = [call]
        detail, issues = validate_completion_case(call, case)
        record.update(detail)
        record["issues"].extend(issues)
        record["passed"] = (
            detail["verifier"].get("passed") is True and not record["issues"]
        )
    record["finished_at"] = utc_now()
    return record


def run_api(args: argparse.Namespace) -> int:
    section = args.fixture["api"]
    cases = section["cases"]
    save_json(
        args.out_dir / "input-contract.json",
        {
            "schema": "orcarouter-workload-input-contract/v1",
            "mode": "api",
            "label": args.label,
            "model": args.model,
            "base_url": args.base_url,
            "fixture_sha256": args.fixture_sha256,
            "case_ids": [case["id"] for case in cases],
            "generation": section["generation"],
            "no_silent_retry": True,
        },
    )
    started = utc_now()
    records: list[dict[str, Any]] = []
    limits = httpx.Limits(max_connections=16, max_keepalive_connections=16)
    with httpx.Client(trust_env=False, limits=limits) as client:
        for case in cases:
            try:
                record = api_case(args, client, case)
            except Exception as error:
                record = {
                    "schema": "orcarouter-workload-api-record/v1",
                    "id": case["id"],
                    "kind": case["kind"],
                    "calls": [],
                    "issues": [f"harness exception: {type(error).__name__}: {error}"],
                    "passed": False,
                    "finished_at": utc_now(),
                }
            records.append(record)
            record_path = (
                args.out_dir / "records" / (safe_name(str(case["id"])) + ".json")
            )
            save_json(record_path, record)
            append_jsonl(
                args.out_dir / "progress.jsonl",
                {
                    "completed": len(records),
                    "expected": len(cases),
                    "id": case["id"],
                    "passed": record["passed"],
                    "issues": record.get("issues"),
                    "record": str(record_path),
                },
            )
            print(
                canonical(
                    {
                        "completed": len(records),
                        "expected": len(cases),
                        "id": case["id"],
                        "passed": record["passed"],
                    }
                ),
                flush=True,
            )
    passed = len(records) == len(cases) and all(
        record.get("passed") is True for record in records
    )
    summary = {
        "schema": "orcarouter-workload-api-summary/v1",
        "label": args.label,
        "model": args.model,
        "started_at": started,
        "finished_at": utc_now(),
        "expected": len(cases),
        "completed": len(records),
        "passed_cases": sum(record.get("passed") is True for record in records),
        "failed_cases": sum(record.get("passed") is not True for record in records),
        "passed": passed,
        "results": [
            {
                "id": record["id"],
                "kind": record["kind"],
                "passed": record["passed"],
                "issues": record.get("issues"),
                "record": str(
                    args.out_dir / "records" / (safe_name(str(record["id"])) + ".json")
                ),
            }
            for record in records
        ],
        "scope": "Non-stream and SSE reconstruction, tool continuations, empty/null tool results, JSON schema, stop/usage, malformed requests, and concurrent identity isolation. Every fixture is attempted once; there are no retries.",
    }
    save_json(args.out_dir / "summary.json", summary)
    print(
        canonical({key: value for key, value in summary.items() if key != "results"}),
        flush=True,
    )
    return 0 if passed else 1


async def async_post_json(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
    request_id: str,
) -> dict[str, Any]:
    url = base_url + CHAT_PATH
    started_wall = utc_now()
    started = time.monotonic()
    try:
        async with asyncio.timeout(timeout):
            response = await client.post(
                url,
                json=payload,
                headers={"x-request-id": safe_name(request_id)},
            )
        raw = response.content.decode("utf-8", errors="replace")
        parsed, decode_error = decode_json(raw)
        status = int(response.status_code)
        return {
            "schema": "orcarouter-workload-http-call/v1",
            "stream": False,
            "url": url,
            "request_id": request_id,
            "request": payload,
            "request_sha256": sha256_bytes(canonical(payload).encode()),
            "started_at": started_wall,
            "status": status,
            "response_headers": response_headers(response),
            "raw_body": raw,
            "response": parsed,
            "elapsed_seconds": time.monotonic() - started,
            "error": decode_error
            if 200 <= status < 300
            else f"HTTP {status}" + (f"; {decode_error}" if decode_error else ""),
            "ok": 200 <= status < 300 and decode_error is None,
        }
    except TimeoutError:
        error: BaseException = TimeoutError("request timeout")
    except asyncio.CancelledError:
        raise
    except Exception as caught:
        error = caught
    return {
        "schema": "orcarouter-workload-http-call/v1",
        "stream": False,
        "url": url,
        "request_id": request_id,
        "request": payload,
        "request_sha256": sha256_bytes(canonical(payload).encode()),
        "started_at": started_wall,
        "status": None,
        "response_headers": {},
        "raw_body": "",
        "response": None,
        "elapsed_seconds": time.monotonic() - started,
        "error": f"{type(error).__name__}: {error}",
        "ok": False,
    }


async def async_cancel_stream(
    client: httpx.AsyncClient,
    base_url: str,
    payload: dict[str, Any],
    timeout: float,
    request_id: str,
) -> dict[str, Any]:
    url = base_url + CHAT_PATH
    started_wall = utc_now()
    started = time.monotonic()
    raw_lines: list[str] = []
    events: list[object] = []
    status: int | None = None
    headers: dict[str, str] = {}
    first_generated = False
    done = False
    terminal_finish: object | None = None
    error_text: str | None = None
    try:
        async with asyncio.timeout(timeout):
            async with client.stream(
                "POST",
                url,
                json=payload,
                headers={
                    "x-request-id": safe_name(request_id),
                    "accept": "text/event-stream",
                    "connection": "close",
                },
            ) as response:
                status = int(response.status_code)
                headers = response_headers(response)
                if status != 200:
                    raw_lines.append((await response.aread()).decode(errors="replace"))
                    error_text = f"HTTP {status}"
                else:
                    async for line in response.aiter_lines():
                        raw_lines.append(line)
                        if sum(len(item) + 1 for item in raw_lines) > 1024 * 1024:
                            error_text = "cancellation capture exceeded 1 MiB"
                            break
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].lstrip()
                        if data == "[DONE]":
                            done = True
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError as error:
                            error_text = f"malformed SSE JSON: {error}"
                            break
                        events.append(event)
                        if not isinstance(event, dict):
                            continue
                        for choice in event.get("choices") or []:
                            if not isinstance(choice, dict):
                                continue
                            if choice.get("finish_reason") is not None:
                                terminal_finish = choice["finish_reason"]
                            delta = choice.get("delta")
                            if isinstance(delta, dict) and any(
                                delta.get(key)
                                for key in (
                                    "content",
                                    "reasoning",
                                    "reasoning_content",
                                    "tool_calls",
                                )
                            ):
                                first_generated = True
                                break
                        if first_generated or terminal_finish is not None:
                            break
    except TimeoutError:
        error_text = "TimeoutError: cancellation stream timeout"
    except asyncio.CancelledError:
        raise
    except Exception as error:
        error_text = f"{type(error).__name__}: {error}"
    return {
        "schema": "orcarouter-workload-cancellation-call/v1",
        "url": url,
        "request_id": request_id,
        "request": payload,
        "request_sha256": sha256_bytes(canonical(payload).encode()),
        "started_at": started_wall,
        "status": status,
        "response_headers": headers,
        "raw_body": "\n".join(raw_lines),
        "events": events,
        "first_generated_data_event_observed": first_generated,
        "done_observed": done,
        "terminal_finish_reason": terminal_finish,
        "client_closed_early": first_generated and not done and terminal_finish is None,
        "elapsed_seconds_before_close": time.monotonic() - started,
        "error": error_text,
    }


async def models_metadata(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    url = base_url + "/v1/models"
    started = time.monotonic()
    try:
        async with asyncio.timeout(60):
            response = await client.get(url)
        raw = response.content.decode(errors="replace")
        parsed, decode_error = decode_json(raw)
        return {
            "url": url,
            "status": int(response.status_code),
            "response_headers": response_headers(response),
            "raw_body": raw,
            "response": parsed,
            "error": decode_error
            if response.status_code == 200
            else f"HTTP {response.status_code}"
            + (f"; {decode_error}" if decode_error else ""),
            "elapsed_seconds": time.monotonic() - started,
        }
    except Exception as error:
        return {
            "url": url,
            "status": None,
            "response": None,
            "raw_body": "",
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.monotonic() - started,
        }


def context_candidates(value: object) -> list[int]:
    found: list[int] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key.casefold() in {
                "max_model_len",
                "max_context_length",
                "context_length",
                "max_position_embeddings",
            }:
                if isinstance(child, int) and not isinstance(child, bool) and child > 0:
                    found.append(child)
                elif isinstance(child, str) and child.isdigit():
                    found.append(int(child))
            found.extend(context_candidates(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(context_candidates(child))
    return found


def context_limit(metadata: dict[str, Any], model: str) -> tuple[int | None, list[int]]:
    response = metadata.get("response")
    if not isinstance(response, dict):
        return None, []
    rows = response.get("data")
    if isinstance(rows, list):
        exact = [
            row for row in rows if isinstance(row, dict) and row.get("id") == model
        ]
        candidates = context_candidates(exact if exact else rows)
    else:
        candidates = context_candidates(response)
    return (max(candidates) if candidates else None), sorted(set(candidates))


def json_schema_for(value: object) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, list):
        item_schemas = [json_schema_for(item) for item in value]
        item = (
            item_schemas[0]
            if item_schemas
            and all(schema == item_schemas[0] for schema in item_schemas)
            else {}
        )
        return {
            "type": "array",
            "items": item,
            "minItems": len(value),
            "maxItems": len(value),
        }
    if isinstance(value, dict):
        return {
            "type": "object",
            "properties": {key: json_schema_for(child) for key, child in value.items()},
            "required": list(value),
            "additionalProperties": False,
        }
    return {"type": "null"}


def context_error(call: dict[str, Any]) -> bool:
    if call.get("status") not in (400, 413, 422):
        return False
    text = (
        str(call.get("raw_body") or "") + " " + str(call.get("error") or "")
    ).casefold()
    return any(
        phrase in text
        for phrase in (
            "context length",
            "maximum context",
            "max model len",
            "too many tokens",
            "prompt is too long",
        )
    )


def prefill_text(nominal_tokens: int) -> str:
    # A leading-space common word is one token in the shared GLM tokenizer;
    # server-reported prompt_tokens remains authoritative.
    return (" the" * nominal_tokens).lstrip()


def stable_seed(name: str, base: int = 260919) -> int:
    return base + int(hashlib.sha256(name.encode()).hexdigest()[:6], 16) % 1000000


class SoakState:
    def __init__(
        self,
        args: argparse.Namespace,
        section: dict[str, Any],
        known_context: int | None,
        prefill_prompts: dict[str, str],
    ):
        self.args = args
        self.section = section
        self.known_context = known_context
        self.prefill_prompts = prefill_prompts
        self.sequence = 0
        self.rows: list[dict[str, Any]] = []

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def emit(self, record: dict[str, Any]) -> None:
        sequence = int(record["sequence"])
        path = (
            self.args.out_dir
            / "records"
            / f"{sequence:08d}-{safe_name(str(record.get('kind') or 'record'))}.json"
        )
        save_json(path, record)
        projection = {
            "sequence": sequence,
            "phase_concurrency": record.get("phase_concurrency"),
            "worker": record.get("worker"),
            "kind": record.get("kind"),
            "fixture_id": record.get("fixture_id"),
            "outcome": record.get("outcome"),
            "passed": record.get("passed"),
            "runtime_error": record.get("runtime_error"),
            "truncated": record.get("truncated"),
            "elapsed_seconds": record.get("elapsed_seconds"),
            "prompt_tokens": record.get("prompt_tokens"),
            "completion_tokens": record.get("completion_tokens"),
            "repetition_candidate": record.get("repetition_candidate"),
            "record": str(path),
        }
        self.rows.append(projection)
        append_jsonl(self.args.out_dir / "progress.jsonl", projection)
        print(canonical(projection), flush=True)


def soak_body(
    args: argparse.Namespace,
    section: dict[str, Any],
    messages: list[dict[str, Any]],
    fixture_id: str,
    expected: object | None = None,
) -> dict[str, Any]:
    generation = copy.deepcopy(section["generation"])
    generation["seed"] = stable_seed(fixture_id)
    body = generation_body(args.model, generation, messages)
    if expected is not None:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "soak_" + safe_name(fixture_id).replace("-", "_")[:48],
                "strict": True,
                "schema": json_schema_for(expected),
            },
        }
    return body


async def normal_soak_attempt(
    state: SoakState,
    client: httpx.AsyncClient,
    concurrency: int,
    worker: int,
    fixture_id: str,
    kind: str,
    messages: list[dict[str, Any]],
    expected: object,
    timeout: float,
    body_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sequence = state.next_sequence()
    body = soak_body(
        state.args,
        state.section,
        messages,
        fixture_id,
        expected if kind not in ("prefill", "reasoning", "history") else None,
    )
    if body_extra:
        body.update(copy.deepcopy(body_extra))
    call = await async_post_json(
        client,
        state.args.base_url,
        body,
        timeout,
        f"soak-{concurrency}-{worker}-{sequence}",
    )
    record: dict[str, Any] = {
        "schema": "orcarouter-workload-soak-record/v1",
        "sequence": sequence,
        "phase_concurrency": concurrency,
        "worker": worker,
        "kind": kind,
        "fixture_id": fixture_id,
        "started_at": call.get("started_at"),
        "calls": [call],
        "passed": False,
        "runtime_error": False,
        "truncated": False,
    }
    if kind == "prefill" and context_error(call):
        record.update(
            {
                "outcome": "unsupported_context_length",
                "passed": None,
                "runtime_error": False,
                "error": call.get("error"),
                "elapsed_seconds": call.get("elapsed_seconds"),
                "finished_at": utc_now(),
            }
        )
        return record
    message, error = response_message(call)
    if error:
        record.update(
            {"outcome": "runtime_error", "runtime_error": True, "error": error}
        )
    else:
        assert message is not None
        visible, reasoning = message_text(message)
        finish = finish_reason(call)
        record.update(
            {
                "visible_answer": visible,
                "reasoning": reasoning,
                "finish_reason": finish,
                "usage": response_usage(call),
                "repetition_candidate": repetition_evidence(visible + "\n" + reasoning),
            }
        )
        record["truncated"] = finish == "length"
        if kind == "prefill":
            semantic_passed = visible.strip() == str(expected)
            record["verifier"] = {
                "passed": semantic_passed,
                "actual": visible.strip(),
                "expected": expected,
            }
        else:
            try:
                actual = parse_json_answer(visible)
            except (json.JSONDecodeError, ValueError, TypeError) as parse_error:
                actual = None
                record["verifier"] = {
                    "passed": False,
                    "parse_error": str(parse_error),
                    "expected": expected,
                }
            else:
                semantic_passed = equal_json(actual, expected)
                record["verifier"] = {
                    "passed": semantic_passed,
                    "actual": actual,
                    "expected": expected,
                }
            semantic_passed = record["verifier"]["passed"]
        record["passed"] = bool(
            semantic_passed
            and not record["truncated"]
            and finish == "stop"
            and not message.get("tool_calls")
        )
        record["outcome"] = (
            "pass"
            if record["passed"]
            else ("truncation" if record["truncated"] else "deterministic_failure")
        )
    usage = response_usage(call)
    record["prompt_tokens"] = (
        usage.get("prompt_tokens")
        if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int)
        else None
    )
    record["completion_tokens"] = (
        usage.get("completion_tokens")
        if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int)
        else None
    )
    record["elapsed_seconds"] = call.get("elapsed_seconds")
    record["finished_at"] = utc_now()
    return record


async def cancellation_recovery_attempt(
    state: SoakState,
    client: httpx.AsyncClient,
    concurrency: int,
    worker: int,
    attempt: int,
    timeout: float,
) -> dict[str, Any]:
    sequence = state.next_sequence()
    fixture = state.section["cancellation"]
    identity = f"c{concurrency}-w{worker}-a{attempt}"
    cancellation_payload = generation_body(
        state.args.model,
        state.section["generation"],
        [{"role": "user", "content": fixture["prompt"]}],
        int(fixture["max_tokens"]),
    )
    cancellation_payload.update(
        {
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": True,
            "min_tokens": int(fixture["max_tokens"]),
            "seed": stable_seed("cancel-recovery"),
            "cache_salt": "orca-soak-cancel-" + identity,
        }
    )
    cancellation = await async_cancel_stream(
        client,
        state.args.base_url,
        cancellation_payload,
        timeout,
        f"soak-cancel-{identity}",
    )
    marker = str(fixture["recovery_prefix"]) + identity.upper().replace("-", "_")
    recovery_payload = generation_body(
        state.args.model,
        state.section["generation"],
        [
            {
                "role": "user",
                "content": f"Return exactly this token and nothing else: {marker}",
            }
        ],
        512,
    )
    recovery_payload["seed"] = stable_seed("recovery-" + marker)
    recovery = await async_post_json(
        client,
        state.args.base_url,
        recovery_payload,
        timeout,
        f"soak-recovery-{identity}",
    )
    message, error = response_message(recovery)
    visible = reasoning = ""
    if message is not None:
        visible, reasoning = message_text(message)
    cancelled = cancellation.get(
        "client_closed_early"
    ) is True and not cancellation.get("error")
    recovered = (
        error is None
        and visible.strip() == marker
        and finish_reason(recovery) == "stop"
        and not message.get("tool_calls")
    )
    usage = response_usage(recovery)
    cancellation_elapsed = cancellation.get("elapsed_seconds_before_close")
    recovery_elapsed = recovery.get("elapsed_seconds")
    elapsed = (
        cancellation_elapsed + recovery_elapsed
        if isinstance(cancellation_elapsed, (int, float))
        and not isinstance(cancellation_elapsed, bool)
        and isinstance(recovery_elapsed, (int, float))
        and not isinstance(recovery_elapsed, bool)
        else None
    )
    runtime_error = (
        cancellation.get("status") != 200
        or bool(cancellation.get("error"))
        or recovery.get("ok") is not True
    )
    return {
        "schema": "orcarouter-workload-soak-record/v1",
        "sequence": sequence,
        "phase_concurrency": concurrency,
        "worker": worker,
        "kind": "cancel_recovery",
        "fixture_id": "cancel-recovery",
        "started_at": cancellation.get("started_at"),
        "calls": [cancellation, recovery],
        "cancellation_observed": cancelled,
        "recovery_useful": recovered,
        "visible_answer": visible,
        "reasoning": reasoning,
        "expected_recovery": marker,
        "finish_reason": finish_reason(recovery),
        "usage": usage,
        "prompt_tokens": usage.get("prompt_tokens")
        if isinstance(usage, dict) and isinstance(usage.get("prompt_tokens"), int)
        else None,
        "completion_tokens": usage.get("completion_tokens")
        if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int)
        else None,
        "elapsed_seconds": elapsed,
        "runtime_error": runtime_error,
        "truncated": False,
        "passed": cancelled and recovered,
        "outcome": "pass"
        if cancelled and recovered
        else "cancellation_or_recovery_failure",
        "finished_at": utc_now(),
    }


def reset_history(section: dict[str, Any], session: int) -> dict[str, Any]:
    return {
        "session": session,
        "next_turn": -1,
        "messages": [{"role": "system", "content": section["history"]["system"]}],
    }


async def soak_worker(
    state: SoakState,
    client: httpx.AsyncClient,
    concurrency: int,
    worker: int,
    deadline: float,
) -> None:
    mix = state.section["mix"]
    attempt = 0
    history = reset_history(state.section, 0)
    current_kind = "not_started"
    try:
        while time.monotonic() < deadline:
            current_kind = str(mix[(attempt + worker) % len(mix)])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            timeout = float(state.section["request_timeout_seconds"])
            if current_kind == "canary":
                fixture = state.section["canaries"][
                    (attempt + worker) % len(state.section["canaries"])
                ]
                record = await normal_soak_attempt(
                    state,
                    client,
                    concurrency,
                    worker,
                    fixture["id"],
                    "canary",
                    [{"role": "user", "content": fixture["prompt"]}],
                    fixture["expected"],
                    timeout,
                )
            elif current_kind == "reasoning":
                fixture = state.section["reasoning_tasks"][
                    (attempt + worker) % len(state.section["reasoning_tasks"])
                ]
                record = await normal_soak_attempt(
                    state,
                    client,
                    concurrency,
                    worker,
                    fixture["id"],
                    "reasoning",
                    [{"role": "user", "content": fixture["prompt"]}],
                    fixture["expected"],
                    timeout,
                    {"reasoning_effort": "high"},
                )
            elif current_kind == "history":
                history_fixture = state.section["history"]
                if history["next_turn"] == -1:
                    prompt = history_fixture["initial_user"]
                    expected = history_fixture["initial_expected"]
                    fixture_id = f"history-s{history['session']}-initial"
                else:
                    turn = history_fixture["turns"][history["next_turn"]]
                    prompt = turn["prompt"]
                    expected = turn["expected"]
                    fixture_id = (
                        f"history-s{history['session']}-turn{history['next_turn'] + 1}"
                    )
                history["messages"].append({"role": "user", "content": prompt})
                record = await normal_soak_attempt(
                    state,
                    client,
                    concurrency,
                    worker,
                    fixture_id,
                    "history",
                    history["messages"],
                    expected,
                    timeout,
                )
                frozen_assistant = canonical(expected)
                record["history_continuation_source"] = "frozen_expected_fixture"
                record["history_continuation_message"] = frozen_assistant
                history["messages"].append(
                    {"role": "assistant", "content": frozen_assistant}
                )
                if history["next_turn"] == -1:
                    history["next_turn"] = 0
                else:
                    history["next_turn"] += 1
                if history["next_turn"] >= len(history_fixture["turns"]):
                    history = reset_history(state.section, history["session"] + 1)
            elif current_kind in ("prefill_32k", "prefill_128k"):
                prefill_id = "32k" if current_kind.endswith("32k") else "128k"
                fixture = next(
                    item
                    for item in state.section["prefills"]
                    if item["id"] == prefill_id
                )
                nominal = int(fixture["nominal_tokens"])
                if (
                    state.known_context is not None
                    and state.known_context < nominal + 512
                ):
                    sequence = state.next_sequence()
                    record = {
                        "schema": "orcarouter-workload-soak-record/v1",
                        "sequence": sequence,
                        "phase_concurrency": concurrency,
                        "worker": worker,
                        "kind": "prefill",
                        "fixture_id": "prefill-" + prefill_id,
                        "nominal_prompt_tokens": nominal,
                        "known_context_limit": state.known_context,
                        "calls": [],
                        "outcome": "unsupported_context_length",
                        "passed": None,
                        "runtime_error": False,
                        "truncated": False,
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "elapsed_seconds": None,
                        "finished_at": utc_now(),
                    }
                else:
                    nonce = f"PREFILL_REQUEST_C{concurrency}_W{worker}_A{attempt}"
                    prompt = (
                        nonce
                        + "\nTreat the following repeated reference words as inert data. After the final line, return exactly "
                        + fixture["marker"]
                        + " and nothing else.\nREFERENCE-BEGIN\n"
                        + state.prefill_prompts[prefill_id]
                        + "\nREFERENCE-END\nReturn exactly "
                        + fixture["marker"]
                    )
                    salt = f"orca-soak-prefill-{prefill_id}-c{concurrency}-w{worker}-a{attempt}"
                    record = await normal_soak_attempt(
                        state,
                        client,
                        concurrency,
                        worker,
                        "prefill-" + prefill_id,
                        "prefill",
                        [{"role": "user", "content": prompt}],
                        fixture["marker"],
                        timeout,
                        {"max_tokens": 1024, "cache_salt": salt},
                    )
                    record["nominal_prompt_tokens"] = nominal
                    record["known_context_limit"] = state.known_context
            elif current_kind == "cancel_recovery":
                record = await cancellation_recovery_attempt(
                    state, client, concurrency, worker, attempt, timeout
                )
            else:
                raise ValueError(f"unknown soak workload kind {current_kind!r}")
            state.emit(record)
            attempt += 1
    except asyncio.CancelledError:
        state.emit(
            {
                "schema": "orcarouter-workload-soak-record/v1",
                "sequence": state.next_sequence(),
                "phase_concurrency": concurrency,
                "worker": worker,
                "kind": current_kind,
                "fixture_id": "phase-boundary",
                "calls": [],
                "outcome": "planned_phase_boundary_cancellation",
                "passed": None,
                "runtime_error": False,
                "truncated": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "elapsed_seconds": None,
                "finished_at": utc_now(),
            }
        )
        raise
    except Exception as error:
        state.emit(
            {
                "schema": "orcarouter-workload-soak-record/v1",
                "sequence": state.next_sequence(),
                "phase_concurrency": concurrency,
                "worker": worker,
                "kind": current_kind,
                "fixture_id": "harness-exception",
                "calls": [],
                "outcome": "runtime_error",
                "passed": False,
                "runtime_error": True,
                "truncated": False,
                "error": f"harness exception: {type(error).__name__}: {error}",
                "prompt_tokens": None,
                "completion_tokens": None,
                "elapsed_seconds": None,
                "finished_at": utc_now(),
            }
        )


async def run_soak_async(args: argparse.Namespace) -> int:
    section = args.fixture["soak"]
    limits = httpx.Limits(
        max_connections=max(section["concurrencies"]) + 4,
        max_keepalive_connections=max(section["concurrencies"]) + 4,
    )
    async with httpx.AsyncClient(
        trust_env=False, limits=limits, timeout=None
    ) as client:
        metadata = await models_metadata(client, args.base_url)
        known_context, candidates = context_limit(metadata, args.model)
        save_json(
            args.out_dir / "models-metadata.json",
            {
                "schema": "orcarouter-workload-model-metadata/v1",
                "call": metadata,
                "model": args.model,
                "context_limit": known_context,
                "context_limit_candidates": candidates,
                "context_observability": "observed"
                if known_context is not None
                else "unobservable",
            },
        )
        prefills = {
            item["id"]: prefill_text(int(item["nominal_tokens"]))
            for item in section["prefills"]
        }
        state = SoakState(args, section, known_context, prefills)
        save_json(
            args.out_dir / "input-contract.json",
            {
                "schema": "orcarouter-workload-input-contract/v1",
                "mode": "soak",
                "label": args.label,
                "model": args.model,
                "base_url": args.base_url,
                "fixture_sha256": args.fixture_sha256,
                "duration_seconds": args.duration_seconds,
                "concurrencies": section["concurrencies"],
                "generation": section["generation"],
                "mix": section["mix"],
                "nominal_prefills": section["prefills"],
                "context_limit": known_context,
                "actual_prompt_tokens_authoritative": True,
                "history_continuations": "frozen expected assistant turns keep subsequent requests identical across models; each actual response remains scored and preserved",
                "no_silent_retry": True,
                "degeneration_heuristics_are_diagnostic_only": True,
            },
        )
        started_at = utc_now()
        origin = time.monotonic()
        phase_rows: list[dict[str, Any]] = []
        concurrencies = section["concurrencies"]
        for phase_index, concurrency in enumerate(concurrencies):
            phase_start = time.monotonic()
            phase_started_at = utc_now()
            phase_deadline = origin + float(args.duration_seconds) * (
                phase_index + 1
            ) / len(concurrencies)
            before = len(state.rows)
            workers = [
                asyncio.create_task(
                    soak_worker(state, client, int(concurrency), worker, phase_deadline)
                )
                for worker in range(int(concurrency))
            ]
            remaining = max(0.0, phase_deadline - time.monotonic())
            _, pending = await asyncio.wait(workers, timeout=remaining)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            completed = workers[:]
            for task in completed:
                if task.done() and not task.cancelled():
                    exception = task.exception()
                    if exception:
                        state.emit(
                            {
                                "schema": "orcarouter-workload-soak-record/v1",
                                "sequence": state.next_sequence(),
                                "phase_concurrency": concurrency,
                                "worker": None,
                                "kind": "phase",
                                "fixture_id": "worker-exception",
                                "calls": [],
                                "outcome": "runtime_error",
                                "passed": False,
                                "runtime_error": True,
                                "truncated": False,
                                "error": f"worker exception: {type(exception).__name__}: {exception}",
                                "finished_at": utc_now(),
                            }
                        )
            phase_projection = state.rows[before:]
            phase_rows.append(
                {
                    "concurrency": concurrency,
                    "started_at": phase_started_at,
                    "started_offset_seconds": phase_start - origin,
                    "elapsed_seconds": time.monotonic() - phase_start,
                    "records": len(phase_projection),
                    "completed_attempts": sum(
                        row.get("outcome") != "planned_phase_boundary_cancellation"
                        for row in phase_projection
                    ),
                    "runtime_errors": sum(
                        row.get("runtime_error") is True for row in phase_projection
                    ),
                    "failed_attempts": sum(
                        row.get("passed") is False for row in phase_projection
                    ),
                    "unsupported": sum(
                        row.get("outcome") == "unsupported_context_length"
                        for row in phase_projection
                    ),
                }
            )
    rows = state.rows
    runtime_errors = sum(row.get("runtime_error") is True for row in rows)
    deterministic_failures = sum(row.get("passed") is False for row in rows)
    completed_by_concurrency = {
        str(concurrency): sum(
            row.get("phase_concurrency") == concurrency
            and row.get("outcome")
            not in ("planned_phase_boundary_cancellation", "unsupported_context_length")
            for row in rows
        )
        for concurrency in section["concurrencies"]
    }
    required_concurrency_covered = all(
        count > 0 for count in completed_by_concurrency.values()
    )
    attempt_rows = [
        row
        for row in rows
        if row.get("outcome") != "planned_phase_boundary_cancellation"
    ]
    cancellation_attempts = [
        row for row in attempt_rows if row.get("kind") == "cancel_recovery"
    ]
    kind_counts: dict[str, int] = {}
    for row in rows:
        kind = str(row.get("kind"))
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    required_kinds = ("canary", "reasoning", "history", "prefill", "cancel_recovery")
    workload_coverage = {
        kind: sum(row.get("kind") == kind for row in attempt_rows)
        for kind in required_kinds
    }
    history_multiturn_completed = any(
        row.get("kind") == "history"
        and str(row.get("fixture_id") or "").endswith("-turn3")
        for row in attempt_rows
    )
    prefill_coverage = {
        prefill_id: sum(
            row.get("fixture_id") == "prefill-" + prefill_id for row in attempt_rows
        )
        for prefill_id in ("32k", "128k")
    }
    required_workloads_covered = (
        all(count > 0 for count in workload_coverage.values())
        and history_multiturn_completed
        and all(count > 0 for count in prefill_coverage.values())
    )
    runtime_stability_passed = (
        runtime_errors == 0
        and required_concurrency_covered
        and required_workloads_covered
    )
    model_checks_passed = deterministic_failures == 0
    passed = runtime_stability_passed and model_checks_passed
    unsupported_context_attempts = sum(
        row.get("outcome") == "unsupported_context_length" for row in rows
    )
    unobservable_usage_attempts = sum(
        row.get("kind") in required_kinds
        and row.get("outcome") != "unsupported_context_length"
        and row.get("prompt_tokens") is None
        for row in attempt_rows
    )
    qualifiers: list[str] = []
    if unsupported_context_attempts:
        qualifiers.append("unsupported_context")
    if known_context is None:
        qualifiers.append("context_limit_unobservable")
    if unobservable_usage_attempts:
        qualifiers.append("usage_unobservable")
    status = (
        "fail"
        if not passed
        else "pass"
        if not qualifiers
        else "pass_with_" + "_and_".join(qualifiers)
    )

    def numeric_stats(field: str) -> dict[str, Any]:
        values = [
            float(row[field])
            for row in rows
            if isinstance(row.get(field), (int, float))
            and not isinstance(row.get(field), bool)
        ]
        return {
            "available": len(values),
            "unavailable": len(rows) - len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
        }

    summary = {
        "schema": "orcarouter-workload-soak-summary/v1",
        "status": status,
        "label": args.label,
        "model": args.model,
        "started_at": started_at,
        "finished_at": utc_now(),
        "requested_duration_seconds": args.duration_seconds,
        "actual_timed_phase_seconds": time.monotonic() - origin,
        "phase_summaries": phase_rows,
        "records": len(rows),
        "kind_counts": kind_counts,
        "completed_by_concurrency": completed_by_concurrency,
        "required_concurrency_covered": required_concurrency_covered,
        "runtime_errors": runtime_errors,
        "deterministic_failures": deterministic_failures,
        "runtime_stability_passed": runtime_stability_passed,
        "model_checks_passed": model_checks_passed,
        "unsupported_context_attempts": unsupported_context_attempts,
        "unobservable_usage_attempts": unobservable_usage_attempts,
        "planned_phase_boundary_cancellations": sum(
            row.get("outcome") == "planned_phase_boundary_cancellation" for row in rows
        ),
        "cancellation_recovery_attempts": len(cancellation_attempts),
        "workload_coverage": workload_coverage,
        "prefill_coverage": prefill_coverage,
        "history_multiturn_completed": history_multiturn_completed,
        "required_workloads_covered": required_workloads_covered,
        "degeneration_heuristic_candidates": [
            row["record"] for row in rows if row.get("repetition_candidate")
        ],
        "latency_seconds": numeric_stats("elapsed_seconds"),
        "prompt_tokens": numeric_stats("prompt_tokens"),
        "completion_tokens": numeric_stats("completion_tokens"),
        "context_limit": known_context,
        "context_limit_candidates": candidates,
        "context_observability": "observed"
        if known_context is not None
        else "unobservable",
        "passed": passed,
        "results": rows,
        "scope": "Bounded mixed benign traffic at concurrency 1/4/8 with deterministic canaries, growing histories, nominal 32K/128K cold prefills, reasoning tasks, deliberate cancellation followed by a useful-response check, and diagnostic-only repetition heuristics. Actual server prompt-token accounting is authoritative; unavailable values remain null.",
    }
    save_json(args.out_dir / "summary.json", summary)
    print(
        canonical({key: value for key, value in summary.items() if key != "results"}),
        flush=True,
    )
    return 0 if passed else 1


def run_soak(args: argparse.Namespace) -> int:
    return asyncio.run(run_soak_async(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible server origin, for example http://127.0.0.1:5002",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("capability", "api", "soak"))
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help="required for soak mode; total across concurrency 1/4/8",
    )
    args = parser.parse_args()
    try:
        args.base_url = validate_base_url(args.base_url)
    except ValueError as error:
        parser.error(str(error))
    if not args.model.strip() or not args.label.strip():
        parser.error("--model and --label must be nonempty")
    if args.mode == "soak":
        if args.duration_seconds is None or args.duration_seconds < 30:
            parser.error("soak mode requires --duration-seconds >= 30")
    return args


def main() -> int:
    args = parse_args()
    fixture, fixture_hash, capability, capability_hash = load_inputs()
    args.fixture = fixture
    args.fixture_sha256 = fixture_hash
    args.capability_tasks_sha256 = capability_hash
    try:
        args.out_dir = prepare_output(args.out_dir)
    except FileExistsError:
        raise SystemExit(
            f"refusing to reuse an existing output directory: {args.out_dir}"
        )
    if args.mode == "capability":
        return run_capability(args, capability)
    if args.mode == "api":
        return run_api(args)
    return run_soak(args)


if __name__ == "__main__":
    raise SystemExit(main())
