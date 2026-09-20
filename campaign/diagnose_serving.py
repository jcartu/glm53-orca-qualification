#!/usr/bin/env python3
"""Bounded C8 replay, synchronous-launch diagnosis, and FP8 memory right-sizing."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import httpx

import campaign
import workload_probe

HERE = Path(__file__).resolve().parent
rt = campaign.rt
coordinator = campaign.coordinator
ARMS = {
    "control-blocking": {
        "model": "production",
        "dcp": 4,
        "env": {"CUDA_LAUNCH_BLOCKING": "1"},
        "args": [],
    },
    "control-noasync": {
        "model": "production",
        "dcp": 4,
        "env": {},
        "args": ["--no-async-scheduling"],
    },
    "original-bounded-profile": {
        "model": "original_fp8",
        "dcp": 1,
        "env": {
            "GPU_MEMORY_UTILIZATION": "0.95",
            "MAX_MODEL_LEN": "131072",
            "MAX_NUM_BATCHED_TOKENS": "1024",
            "MAX_NUM_SEQS": "8",
            "MAX_CUDAGRAPH_CAPTURE_SIZE": "16",
            "CUDAGRAPH_CAPTURE_SIZES": "1 2 4 8 16",
        },
        "args": ["--no-async-scheduling"],
    },
    "control-dcp1": {
        "model": "production",
        "dcp": 1,
        "env": {},
        "args": [],
    },
    "control-piecewise": {
        "model": "production",
        "dcp": 4,
        "env": {"CUDAGRAPH_MODE": "PIECEWISE"},
        "args": [],
    },
    "candidate-piecewise": {
        "model": "candidate",
        "dcp": 4,
        "env": {"CUDAGRAPH_MODE": "PIECEWISE"},
        "args": [],
    },
}
REQUEST_FIXTURE = campaign.REPO / "fixtures/capability/serving-diagnosis-requests.json"


def phase(selected):
    rows = []
    tasks = json.loads(workload_probe.CAPABILITY_TASKS_PATH.read_text())["tasks"]
    oracles = {row["id"]: row["verifier"] for row in tasks}
    request_fixture = json.loads(REQUEST_FIXTURE.read_text())
    requests = [
        (row["id"], row["request"]) for row in request_fixture["requests"]
    ]
    original_args = campaign.COMMON_ARGS
    for label in selected:
        arm = ARMS[label]
        row = {
            "label": label,
            "configuration": arm,
            "booted": False,
            "records": [],
            "passed": False,
        }
        try:
            campaign.COMMON_ARGS = [*original_args, *arm["args"]]
            campaign.boot(
                label, model=arm["model"], dcp=arm["dcp"], overrides=arm["env"]
            )
            row["booted"] = True
            with httpx.Client(trust_env=False, timeout=180) as client:

                def request_one(item):
                    ident, payload = item
                    start = time.monotonic()
                    record = {"id": ident, "request": payload, "passed": False}
                    try:
                        response = client.post(
                            rt.BASE_URL + "/v1/chat/completions", json=payload
                        )
                        record["status"] = response.status_code
                        record["response"] = response.json()
                        response.raise_for_status()
                        choice = record["response"]["choices"][0]
                        visible = choice["message"].get("content") or ""
                        score = workload_probe.score_answer(
                            visible, oracles[ident], ident
                        )
                        record["score"] = score
                        record["passed"] = (
                            score.get("passed") is True
                            and choice["finish_reason"] == "stop"
                        )
                    except Exception as error:
                        record["error"] = repr(error)
                    record["elapsed_seconds"] = time.monotonic() - start
                    return record

                with ThreadPoolExecutor(max_workers=8) as pool:
                    row["records"] = list(pool.map(request_one, requests))
                row["server_healthy_after_requests"] = (
                    client.get(rt.BASE_URL + "/health").status_code == 200
                )
            row["passed"] = row["server_healthy_after_requests"] and all(
                record["passed"] for record in row["records"]
            )
        except Exception as error:
            row["error"] = repr(error)
        finally:
            campaign.COMMON_ARGS = original_args
            rt.capture(label)
            rt.stop()
            rows.append(row)
            rt.save_json(
                "serving-diagnosis.json",
                {"complete": len(rows) == len(selected), "rows": rows},
            )
            rt.record_gate(
                "serving-diagnosis:" + label,
                row["passed"],
                {
                    "booted": row["booted"],
                    "passed_requests": sum(
                        record["passed"] for record in row["records"]
                    ),
                },
            )
    return int(not all(row["passed"] for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", action="store_true")
    parser.add_argument("--arms", default=",".join(ARMS))
    args = parser.parse_args()
    selected = args.arms.split(",")
    if any(name not in ARMS for name in selected):
        parser.error("Unknown serving diagnostic arm")
    if args.phase:
        campaign.load_snapshot_receipt()
        for name, expected in json.loads(
            (rt.ROOT / "source-manifest.json").read_text()
        ).items():
            path = campaign.REPO / name
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Frozen diagnostic source changed: " + name)
        return phase(selected)
    campaign.load_snapshot_receipt()
    campaign.verify_runtime_inputs()
    resolved_image_id = campaign.pin_runtime_image()
    if rt.ROOT.exists() and any(rt.ROOT.iterdir()):
        raise RuntimeError("Use a fresh evidence root")
    if rt._inspect() is not None:
        raise RuntimeError("Existing test container must be investigated first")
    rt.ROOT.mkdir(parents=True)
    rt.save_json(
        "selected-runtime-image.json",
        {
            "requested_image": campaign.REQUESTED_IMAGE,
            "resolved_image_id": resolved_image_id,
        },
    )
    sources = [
        Path(__file__),
        HERE / "campaign.py",
        HERE / "workload_probe.py",
        campaign.REPO / "fixtures/workload_fixtures.json",
        workload_probe.CAPABILITY_TASKS_PATH,
        REQUEST_FIXTURE,
    ]
    rt.save_json(
        "source-manifest.json",
        {str(path.resolve().relative_to(campaign.REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
    )
    coordinator.PHASES = [
        (
            "serving-diagnosis",
            str(Path(__file__).resolve()),
            ["--phase", "--arms", args.arms],
            14400,
        )
    ]
    try:
        coordinator.main()
    finally:
        if (rt.ROOT / "production-restored.json").exists():
            campaign.restored_inference()
    progress = json.loads((rt.ROOT / "phase-progress.json").read_text())
    return int(any(row["returncode"] != 0 for row in progress))


if __name__ == "__main__":
    raise SystemExit(main())
