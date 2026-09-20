#!/usr/bin/env python3
"""CPU-only verifier edge reproductions; no model, network, Docker, or GPU use."""

import asyncio
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wp = load("workload_probe")
ep = load("extended_probe")
fixture = json.loads((REPO / "fixtures/workload_fixtures.json").read_text())
args = SimpleNamespace(
    model="fixture", base_url="http://127.0.0.1:5002", fixture=fixture
)


def response(content, reasoning="", stop_reason=154827):
    return {
        "id": "chatcmpl-verifier-fixture",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning": reasoning,
                },
                "finish_reason": "stop",
                "stop_reason": stop_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


def call(data):
    return {"ok": True, "stream": False, "http_status": 200, "response": data}


def compliant_safety_guidance():
    task = next(
        t for t in fixture["capability"]["extra_tasks"] if t["id"].startswith("X05")
    )
    data = response(
        "I won't provide intrusion instructions. Use a unique password and enable MFA.\nAnswer: SAFE_GUIDANCE"
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=data))
    ) as client:
        record = wp.capability_task(
            args, client, fixture["capability"]["generation"], task
        )
    assert record["task_passed"], record


def stop_inside_reasoning():
    case = next(c for c in fixture["api"]["cases"] if c["kind"] == "stop")
    detail, issues = wp.validate_completion_case(
        call(response(None, "I will output ALPHA BETA", " END")), case
    )
    assert detail["verifier"]["passed"] and not issues, (detail, issues)
    # EOS is not proof that the requested stop string was honored.
    detail, issues = wp.validate_completion_case(call(response("ALPHA BETA")), case)
    assert not detail["verifier"]["passed"] or issues, (detail, issues)


def state(section):
    records = []
    obj = SimpleNamespace(
        args=args,
        section=section,
        next_sequence=lambda: len(records) + 1,
        emit=records.append,
        records=records,
    )
    return obj


async def history_oracle_not_disclosed():
    captured = []

    async def server(client, base_url, body, timeout, label):
        captured.append(body)
        return call(response('{"private_oracle_key":2}'))

    with patch.object(wp, "async_post_json", server):
        await wp.normal_soak_attempt(
            state(fixture["soak"]),
            None,
            1,
            0,
            "history-fixture",
            "history",
            [{"role": "user", "content": "Update the inventory; return JSON."}],
            {"private_oracle_key": 2},
            1.0,
        )
    assert "private_oracle_key" not in json.dumps(captured), captured


async def planned_cancellation_not_runtime_error():
    section = copy.deepcopy(fixture["soak"])
    section["mix"] = ["canary"]
    section["request_timeout_seconds"] = 2.0
    current = state(section)

    async def pending(
        state,
        client,
        concurrency,
        worker,
        fixture_id,
        kind,
        messages,
        expected,
        timeout,
        body_extra=None,
    ):
        async with asyncio.timeout(timeout):
            await asyncio.sleep(1.0)

    with patch.object(wp, "normal_soak_attempt", pending):
        worker = asyncio.create_task(
            wp.soak_worker(current, None, 1, 0, time.monotonic() + 0.02)
        )
        # Model a phase coordinator briefly delayed by a busy event loop.
        await asyncio.sleep(0.04)
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
    assert current.records, "No phase-boundary observation"
    assert all(not row["runtime_error"] for row in current.records), current.records
    assert current.records[-1]["outcome"] == "planned_phase_boundary_cancellation", (
        current.records
    )


def preparation_failure_preserves_all_needles():
    class UnavailableTokenizer:
        def post(self, *args, **kwargs):
            raise RuntimeError("injected tokenizer preparation failure")

    with tempfile.TemporaryDirectory() as directory:
        needle_args = SimpleNamespace(
            base_url=args.base_url, model=args.model, out_dir=Path(directory)
        )
        records = ep.needles(UnavailableTokenizer(), needle_args)
        assert len(records) == 15 and len({row["id"] for row in records}) == 15, records
        assert all(not row["passed"] and row["runtime_error"] for row in records), (
            records
        )


async def main():
    destination = Path(sys.argv[1])
    if destination.exists():
        raise FileExistsError(destination)
    checks = []
    for test in (
        compliant_safety_guidance,
        stop_inside_reasoning,
        history_oracle_not_disclosed,
        planned_cancellation_not_runtime_error,
        preparation_failure_preserves_all_needles,
    ):
        try:
            if inspect.iscoroutinefunction(test):
                await test()
            else:
                test()
            checks.append({"case": test.__name__, "passed": True})
        except Exception as error:
            checks.append(
                {"case": test.__name__, "passed": False, "error": str(error)[:4000]}
            )
    report = {
        "schema": "orca-verifier-edge-reproduction.v1",
        "passed": all(x["passed"] for x in checks),
        "scope": "CPU-only synthetic verifier contracts; not model qualification results.",
        "checks": checks,
    }
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
