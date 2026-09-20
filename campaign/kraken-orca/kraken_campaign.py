#!/usr/bin/env python3
"""Serial Karmic Kraken Orca campaign; the r26 coordinator owns pause/restore.

Phases boot one server at a time on GPUs 0-3, port 5002, under the guarded
coordinator. The reconciled Orca fast path is explicitly labeled lossy:
ORCA_NVFP4_SCALE_POLICY=reconcile_min unifies independent gate/up NVFP4 global
scales onto the smaller divisor (bounded E4M3 rounding, measured worst 5.70%,
mean 2.21% on this checkpoint). Every phase records what it ran; failed gates
stay failures.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
R26 = REPO / "harness/r26"
R29 = REPO / "harness/r29"
sys.path[:0] = [str(R29), str(R26)]
rt = importlib.import_module("runtime")
coordinator = importlib.import_module("run_qualification")

IMAGE_TAG = os.environ.get("KRAKEN_IMAGE", "glm53-orca-kraken-reconciled:local")
CONTAINER = os.environ.get("BATTERY_CONTAINER", "orca-kraken-test")
PORT = int(os.environ.get("BATTERY_PORT", "5002"))
BASE_URL = f"http://127.0.0.1:{PORT}"
SERVED_MODEL = "GLM-5.3-Flash-NVFP4"
ORCA_MODEL = "/mnt/2king/models/Orca-GLM-5.3-Flash-Uncensored-NVFP4-ec0adf4f"
FP8_MODEL = "/mnt/2king/models/GLM-5.3-Flash-Original-FP8-eb9eb208"
ORCA_FP8_MODEL = "/mnt/2king/models/Orca-GLM-5.3-Flash-Uncensored-FP8-3cec42d6"
NVIDIA_MODEL = "/mnt/2king/models/GLM-5.3-Flash-NVFP4-nvidia-09b04e5e"
HF_CACHE = "/home/josh/.cache/huggingface"
BENCH = REPO / "harness/bench/llm_decode_bench.py"
BENCH_ENV = {
    **os.environ,
    "https_proxy": "127.0.0.1:9",
    "http_proxy": "127.0.0.1:9",
    "HTTPS_PROXY": "127.0.0.1:9",
    "HTTP_PROXY": "127.0.0.1:9",
    "no_proxy": "localhost,127.0.0.1",
    "NO_PROXY": "localhost,127.0.0.1",
}


def resolve_image(tag: str) -> str:
    inspected = subprocess.run(
        ["docker", "image", "inspect", tag, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return inspected.stdout.strip()


def chat(payload: dict, timeout: float = 300):
    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


BOOT_TAG = "boot"


def wait_health(deadline_seconds: float = 1500) -> None:
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        if state != "true":
            logs = subprocess.run(
                ["docker", "logs", CONTAINER],
                capture_output=True,
                text=True,
                timeout=120,
            )
            (Path(rt.ROOT) / f"boot-failure-{BOOT_TAG}.log").write_text(
                logs.stdout + logs.stderr
            )
            subprocess.run(
                ["docker", "rm", "-f", CONTAINER], capture_output=True, timeout=90
            )
            raise RuntimeError("server container exited during boot; log retained")
        try:
            with urllib.request.urlopen(BASE_URL + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception:  # noqa: BLE001 - boot in progress
            time.sleep(5)
    raise RuntimeError("server did not become healthy in time")


def boot(arm: str, mode: str) -> None:
    global BOOT_TAG
    BOOT_TAG = f"{arm}-{mode}"
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, timeout=60)
    cache_dir = Path(os.environ.get("KRAKEN_CACHE_ROOT", str(Path(rt.ROOT) / "cache")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER,
        "--label",
        "field-lab.battery=r26",
        "--init",
        "--network",
        "host",
        "--ipc",
        "host",
        "--shm-size",
        "32g",
        "--gpus",
        '"device=0,1,2,3"',
        "-v",
        f"{HF_CACHE}:/root/.cache/huggingface",
        "-v",
        "/mnt/2king/models:/mnt/2king/models:ro",
        "-v",
        f"{cache_dir}:/cache",
        "-e",
        "PROFILE=glm53-flash",
        "-e",
        "HARDWARE_PROFILE=rtx-pro-6000-pcie",
        "-e",
        "TP=4",
        "-e",
        f"PORT={PORT}",
    ]
    native: list[str] = []
    if arm == "orca":
        command += [
            "-e",
            f"MODEL={ORCA_MODEL}",
            "-e",
            "ORCA_NVFP4_SCALE_POLICY=reconcile_min",
            "-e",
            "ORCA_NVFP4_INPUT_SCALE_MISSING=ones",
        ]
        native += ["--quantization", "compressed-tensors"]
    elif arm in ("fp8", "orca-fp8"):
        fp8_model = ORCA_FP8_MODEL if arm == "orca-fp8" else FP8_MODEL
        command += [
            "-e",
            f"MODEL={fp8_model}",
            "-e",
            "ORCA_FP8_SUFFIX_MATCH=1",
            "-e",
            "GPU_MEMORY_UTILIZATION=0.95",
            "-e",
            "MAX_NUM_BATCHED_TOKENS=1024",
            "-e",
            "MAX_NUM_SEQS=8",
        ]
        native += [
            "--quantization",
            "fp8",
            "--load-format",
            "safetensors",
            "--moe-backend",
            "triton",
        ]
    elif arm == "nvidia":
        # Serve the exact local production checkpoint revision; the profile's
        # hub cache is incomplete and the mount is writable so any hub
        # completion can finish.
        command += ["-e", f"MODEL={NVIDIA_MODEL}"]
    else:
        raise RuntimeError(f"unknown arm {arm}")
    if mode == "off":
        native += ["--mode", "off"]
    elif mode == "mtp3":
        if arm == "orca":
            # The profile's draft moe_backend=marlin suits NVIDIA's NVFP4 MTP
            # draft; Orca's MTP draft MoE is BF16, so marlin is rejected and
            # the draft must use triton. An explicit speculative-config also
            # derives mode and draft width in the launcher.
            native += [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "mtp",
                        "num_speculative_tokens": 3,
                        "draft_sample_method": "probabilistic",
                        "rejection_sample_method": "standard",
                        "moe_backend": "triton",
                        "attention_backend": "B12X",
                    }
                ),
            ]
        else:
            native += ["--mode", "mtp", "--draft-tokens", "3"]
    elif mode == "dflash2":
        if arm == "orca":
            # Resolve the MXFP8 draft from the local mirror instead of the
            # hub, and keep the draft MoE on triton (BF16/MXFP8-safe).
            native += [
                "--speculative-config",
                json.dumps(
                    {
                        "method": "dflash",
                        "num_speculative_tokens": 7,
                        "model": "/mnt/2king/models/GLM-5.3-Flash-DFlash2",
                        "draft_sample_method": "probabilistic",
                        "rejection_sample_method": "standard",
                        "moe_backend": "triton",
                        "attention_backend": "FLASH_ATTN",
                        "kv_cache_dtype": "auto",
                    }
                ),
            ]
        else:
            native += ["--mode", "dflash2", "--draft-tokens", "7"]
    else:
        raise RuntimeError(f"unknown mode {mode}")
    image_id = resolve_image(IMAGE_TAG)
    command.append(image_id)
    command += native
    rt.save_json(f"boot-{arm}-{mode}.command.json", command)
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    wait_health()
    inspected = subprocess.run(
        ["docker", "inspect", CONTAINER],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    rt.save_json(
        f"boot-{arm}-{mode}.json",
        {
            "container_id": json.loads(inspected.stdout)[0]["Id"],
            "image_id": image_id,
            "arm": arm,
            "mode": mode,
            "served_model": SERVED_MODEL,
        },
    )


def sentinels(label: str) -> None:
    out = Path(rt.ROOT) / f"sentinels-{label}"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    status, body = chat(
        {
            "model": SERVED_MODEL,
            "messages": [{"role": "user", "content": "What is 23 multiplied by 19?"}],
            "temperature": 0,
            "top_p": 1,
            "seed": 7,
            "max_tokens": 256,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }
    )
    text = body["choices"][0]["message"]["content"] if status == 200 else str(body)
    results["arithmetic"] = {"status": status, "text": text, "passed": "437" in text}

    status, body = chat(
        {
            "model": SERVED_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Reverse the string 'stressed'. Answer with only the reversed string.",
                }
            ],
            "temperature": 0,
            "top_p": 1,
            "seed": 7,
            "max_tokens": 64,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }
    )
    text = body["choices"][0]["message"]["content"] if status == 200 else str(body)
    results["reverse"] = {
        "status": status,
        "text": text,
        "passed": "desserts" in text,
    }

    schema = {
        "type": "object",
        "properties": {"answer": {"type": "integer"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    def one(index: int):
        left, right = 17 + index, 23 + index
        status, body = chat(
            {
                "model": SERVED_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Return only JSON with an integer answer. What is {left} multiplied by {right}?",
                    }
                ],
                "temperature": 0,
                "top_p": 1,
                "seed": 7,
                "max_tokens": 128,
                "reasoning_effort": "low",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"audit_{index}",
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
        )
        if status != 200:
            return {"index": index, "status": status, "passed": False, "body": body}
        content = body["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            passed = parsed.get("answer") == left * right
        except Exception:  # noqa: BLE001
            passed = False
        return {"index": index, "status": status, "passed": passed, "body": content}

    with ThreadPoolExecutor(max_workers=4) as pool:
        burst = list(pool.map(one, range(20)))
    results["structured_burst"] = {
        "requests": len(burst),
        "http_500s": sum(item["status"] == 500 for item in burst),
        "correct": sum(item["passed"] for item in burst),
        "records": burst,
    }
    results["passed"] = (
        results["arithmetic"]["passed"]
        and results["reverse"]["passed"]
        and results["structured_burst"]["http_500s"] == 0
        and results["structured_burst"]["correct"] == 20
    )
    rt.save_json(f"sentinels-{label}.json", results)
    if not results["passed"]:
        raise SystemExit(f"sentinels failed for {label}")


def speed(label: str) -> None:
    out = Path(rt.ROOT) / f"speed-{label}"
    out.mkdir(parents=True, exist_ok=True)
    for trial in (1, 2):
        destination = out / f"trial{trial}.json"
        with open(os.devnull, "rb") as devnull:
            subprocess.run(
                [
                    sys.executable,
                    str(BENCH),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(PORT),
                    "--model",
                    SERVED_MODEL,
                    "--concurrency",
                    "1,4,8",
                    "--contexts",
                    "0,32k",
                    "--duration",
                    "30",
                    "--max-tokens",
                    "8192",
                    "--output",
                    str(destination),
                ],
                env=BENCH_ENV,
                stdin=devnull,
                check=True,
                timeout=3600,
            )
    rt.note(f"SPEED COMPLETE {label}")


def refusal(label: str) -> None:
    out = Path(rt.ROOT) / f"refusal-{label}"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "refusal_probe.py"),
            "--base-url",
            BASE_URL,
            "--model",
            SERVED_MODEL,
            "--label",
            label,
            "--fixture",
            str(REPO / "fixtures/refusal/audit-prompts.json"),
            "--output-dir",
            str(out),
        ],
        check=True,
        timeout=3600,
    )


def quality(label: str) -> None:
    out = Path(rt.ROOT) / f"quality-{label}"
    out.mkdir(parents=True, exist_ok=True)
    capability = subprocess.run(
        [
            sys.executable,
            str(HERE.parent / "workload_probe.py"),
            "--base-url",
            BASE_URL,
            "--model",
            SERVED_MODEL,
            "--label",
            label,
            "--out-dir",
            str(out / "capability"),
            "--mode",
            "capability",
        ],
        timeout=7200,
    )
    # Failed cases are data, not a phase error; record the probe's verdict.
    rt.record_gate(
        f"capability-probe:{label}",
        capability.returncode == 0,
        {"returncode": capability.returncode, "out": str(out / "capability")},
    )
    behavior = importlib.import_module("behavior_probe")
    # The probe's default sandbox image (R29 digest) is not present on this
    # host; use the campaign image, which provides a plain python3.
    behavior.SANDBOX_IMAGE = os.environ.get(
        "ORCA_BEHAVIOR_SANDBOX_IMAGE", resolve_image(IMAGE_TAG)
    )
    sys.argv = [
        "behavior_probe.py",
        "--base-url",
        BASE_URL,
        "--model",
        SERVED_MODEL,
        "--fixture",
        str(REPO / "fixtures/behavior/behavior-fixtures.json"),
        "--output-dir",
        str(out / "behavior"),
    ]
    behavior.main()


def stop_server() -> None:
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True, timeout=90)


def phase(name: str) -> None:
    if name == "orca-off":
        boot("orca", "off")
        sentinels("orca-off")
        speed("orca-off")
    elif name == "orca-mtp3":
        boot("orca", "mtp3")
        sentinels("orca-mtp3")
        speed("orca-mtp3")
    elif name == "orca-mtp3-refusal":
        # The coordinator stops the shared container between phases, so every
        # phase must boot its own server.
        boot("orca", "mtp3")
        refusal("orca-mtp3")
    elif name == "orca-mtp3-quality":
        boot("orca", "mtp3")
        quality("orca-mtp3")
    elif name == "orca-dflash2":
        boot("orca", "dflash2")
        sentinels("orca-dflash2")
        speed("orca-dflash2")
        stop_server()
    elif name == "orca-prompt-layer":
        boot("orca", "mtp3")
        out = Path(rt.ROOT) / "prompt-layer-orca-mtp3"
        probe = subprocess.run(
            [
                sys.executable,
                str(HERE / "prompt_layer_probe.py"),
                "--base-url",
                BASE_URL,
                "--model",
                SERVED_MODEL,
                "--fixture",
                str(REPO / "fixtures/refusal/audit-prompts.json"),
                "--system-prompt",
                str(REPO / "fixtures/refusal/omp-coding-agent-system-prompt.txt"),
                "--output-dir",
                str(out),
            ],
            timeout=3600,
        )
        rt.record_gate(
            "prompt-layer-probe:orca-mtp3",
            probe.returncode == 0,
            {"returncode": probe.returncode, "out": str(out)},
        )
        stop_server()
    elif name == "nvidia-mtp3-refusal":
        # NVIDIA MTP3 on this image dies in the known layer-45 MTP loader
        # mismatch (packed NVFP4 w2 256 vs BF16 tensor 512); deterministic
        # refusal sampling does not need speculation, so run the control
        # without it and label it accordingly.
        boot("nvidia", "off")
        sentinels("nvidia-off")
        refusal("nvidia-off")
        stop_server()
    elif name == "fp8-off-refusal":
        boot("fp8", "off")
        sentinels("fp8-off")
        refusal("fp8-off")
        stop_server()
        boot("orca-fp8", "off")
        sentinels("orca-fp8")
        refusal("orca-fp8")
        stop_server()
    else:
        raise RuntimeError(f"unknown phase {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase")
    parser.add_argument(
        "--stages",
        default="orca-off,orca-mtp3,orca-mtp3-refusal,orca-mtp3-quality,orca-dflash2,nvidia-mtp3-refusal,fp8-off-refusal",
    )
    args = parser.parse_args()
    if args.phase:
        phase(args.phase)
        return
    root = Path(os.environ.get("BATTERY_ROOT", ""))
    if not str(root) or root.exists() and any(root.iterdir()):
        raise RuntimeError("BATTERY_ROOT must be a fresh empty directory")
    rt.ROOT = root
    root.mkdir(parents=True, exist_ok=True)
    rt.save_json(
        "campaign-config.json",
        {
            "image_tag": IMAGE_TAG,
            "image_id": resolve_image(IMAGE_TAG),
            "container": CONTAINER,
            "port": PORT,
            "orca_model": ORCA_MODEL,
            "fp8_model": FP8_MODEL,
            "scale_policy": "reconcile_min",
            "input_scale_policy": "ones",
            "note": "Reconciled NVFP4 gate/up scales are a bounded lossy approximation; compare quality against the exact Marlin reference before deployment claims.",
        },
    )
    coordinator.PHASES = [
        (stage, str(Path(__file__).resolve()), ["--phase", stage], 10800)
        for stage in args.stages.split(",")
    ]
    try:
        coordinator.main()
    finally:
        stop_server()


if __name__ == "__main__":
    main()
