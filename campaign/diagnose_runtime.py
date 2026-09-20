#!/usr/bin/env python3
"""Separate source-FP8 health from compressed-NVFP4 backend compatibility."""

from __future__ import annotations
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
campaign = importlib.import_module("campaign")
rt = campaign.rt
coordinator = campaign.coordinator
campaign.IMAGE = os.environ.get("BATTERY_IMAGE", campaign.IMAGE)

CASES = {
    "original-fp8": ("original_fp8", "marlin"),
    "orca-fp8": ("orca_fp8", "marlin"),
    "orca-nvfp4-marlin": ("candidate", "marlin"),
}
QUESTIONS = [
    ("echo", "Return exactly ORCA_OK_7319 and nothing else.", "ORCA_OK_7319"),
    ("arithmetic", "What is 17 multiplied by 23? Give only the integer.", "391"),
    (
        "translation",
        "Translate the English word hello into French. Give only the French word.",
        "bonjour",
    ),
]


def kernel_probe():
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        rt.NAME,
        "--label",
        "field-lab.battery=r26",
        "--init",
        "--gpus",
        '"device=0"',
        "--network",
        "none",
        "--memory",
        "8g",
        "--memory-swap",
        "8g",
        "--shm-size",
        "2g",
        "-v",
        f"{HERE / 'runtime-repair'}:/probes:ro",
        "-v",
        f"{rt.ROOT}:/evidence",
        "--entrypoint",
        "/opt/venv/bin/python",
        campaign.IMAGE,
        "/probes/scale_kernel_probe.py",
        "--out",
        "/evidence/scale-kernel.json",
    ]
    if rt.run(command, label="kernel-launch", timeout=120):
        raise RuntimeError("Could not launch isolated scale-kernel regression")
    rt._CURRENT_LABEL = "scale-kernel"
    waited = rt.run(["docker", "wait", rt.NAME], label="kernel-wait", timeout=600)
    info = rt._inspect()
    exit_code = info["State"].get("ExitCode") if info else None
    rt.capture("scale-kernel")
    rt.stop()
    if waited or exit_code != 0:
        raise RuntimeError(
            f"Scale-kernel regression failed: wait={waited}, exit={exit_code}"
        )
    receipt = json.loads((rt.ROOT / "scale-kernel.json").read_text())
    if receipt.get("passed") is not True:
        raise RuntimeError("Scale-kernel regression did not produce a passing receipt")


def phase(cases):
    rows = []
    for label in cases:
        model, backend = CASES[label]
        result = {
            "label": label,
            "model": model,
            "moe_backend": backend,
            "calls": [],
            "passed": False,
        }
        try:
            campaign.boot(
                label,
                model=model,
                dcp=1,
                eager=True,
                overrides={
                    "MOE_BACKEND": backend,
                    "LINEAR_BACKEND": "triton",
                    "MAX_MODEL_LEN": "32768",
                    "MAX_NUM_SEQS": "4",
                    "MAX_NUM_BATCHED_TOKENS": "2048",
                },
            )
            result["booted"] = True
            for name, prompt, expected in QUESTIONS:
                payload = {
                    "model": rt.MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 256,
                    "reasoning_effort": "high",
                    "chat_template_kwargs": {
                        "reasoning_effort": "high",
                        "clear_thinking": False,
                    },
                }
                started = time.monotonic()
                response = rt._HTTP.post(
                    rt.BASE_URL + "/v1/chat/completions", json=payload, timeout=180
                )
                record = {
                    "case": name,
                    "request": payload,
                    "status": response.status_code,
                    "seconds": time.monotonic() - started,
                }
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                answer = (message.get("content") or "").strip().strip("`*_.,! ").lower()
                reasoning = (
                    message.get("reasoning") or message.get("reasoning_content") or ""
                )
                record.update(
                    {
                        "response": data,
                        "passed": answer == expected.lower(),
                        "degenerate_lock": reasoning.count("lock") > 32,
                    }
                )
                result["calls"].append(record)
                rt.save_json(f"{label}-{name}.response.json", record)
                if record["degenerate_lock"]:
                    result["early_stop_reason"] = (
                        "Repeated lock tokens in bounded diagnostic; not continuing redundant prompts."
                    )
                    break
            result["passed"] = len(result["calls"]) == len(QUESTIONS) and all(
                row["passed"] for row in result["calls"]
            )
        except Exception as error:
            result["error"] = repr(error)
        finally:
            rt.capture(label)
            rt.stop()
            rows.append(result)
            rt.save_json(
                "runtime-diagnosis.json",
                {"rows": rows, "complete": len(rows) == len(cases)},
            )
            rt.record_gate("runtime-diagnosis:" + label, result["passed"], result)
    return int(not all(row["passed"] for row in rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", action="store_true")
    parser.add_argument("--cases", default=",".join(CASES))
    parser.add_argument("--kernel-probe", action="store_true")
    args = parser.parse_args()
    cases = args.cases.split(",")
    if any(case not in CASES for case in cases):
        parser.error("Unknown diagnostic case")
    if args.phase:
        campaign.load_snapshot_receipt()
        manifest = json.loads((rt.ROOT / "source-manifest.json").read_text())
        for filename, expected in manifest.items():
            path = campaign.REPO / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Frozen diagnostic source changed: {filename}")
        if args.kernel_probe:
            kernel_probe()
        return phase(cases)
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
    rt.save_json(
        "source-manifest.json",
        {
            str(path.resolve().relative_to(campaign.REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                HERE / "campaign.py",
                campaign.R26 / "runtime.py",
                campaign.R26 / "run_qualification.py",
                HERE / "runtime-repair/scale_kernel_probe.py",
                HERE / "runtime-repair/compressed_tensors_moe_w4a4_nvfp4.py",
            )
        },
    )
    coordinator.PHASES = [
        (
            "runtime-diagnosis",
            str(Path(__file__).resolve()),
            ["--phase", "--cases", args.cases]
            + (["--kernel-probe"] if args.kernel_probe else []),
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
