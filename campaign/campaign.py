#!/usr/bin/env python3
"""Serial Orca qualification; the existing coordinator owns GPU isolation/restore."""

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
REPO = HERE.parent
R26 = REPO / "harness/r26"
R29 = REPO / "harness/r29"
FIXTURE_ROOT = REPO / "fixtures"
STATE_ROOT = Path(os.environ.get("ORCA_STATE_DIR", REPO / ".local")).expanduser().resolve()
MODEL_ROOT = Path(os.environ.get("ORCA_MODEL_ROOT", REPO / ".local/models")).expanduser().resolve()
HISTORY_MANIFEST = FIXTURE_ROOT / "history/manifest.json"
SNAPSHOT_RECEIPT = STATE_ROOT / "snapshot-verification/staging-verification.json"
sys.path[:0] = [str(R29), str(R26)]
rt = importlib.import_module("runtime")
coordinator = importlib.import_module("run_qualification")
steady_metrics = importlib.import_module("steady_metrics")

# Build campaign/runtime-repair/Dockerfile.fp8-exclusions with this tag, or
# provide another locally available image through the established override.
IMAGE = os.environ.get("BATTERY_IMAGE", "glm53-orca-qualification-runtime:local")
REQUESTED_IMAGE = os.environ.get("ORCA_REQUESTED_BATTERY_IMAGE", IMAGE)
MODELS = {
    "production": MODEL_ROOT / "GLM-5.3-Flash-NVFP4-nvidia-09b04e5e",
    "candidate": MODEL_ROOT / "Orca-GLM-5.3-Flash-Uncensored-NVFP4-ec0adf4f",
    "orca_fp8": MODEL_ROOT / "Orca-GLM-5.3-Flash-Uncensored-FP8-3cec42d6",
    "original_fp8": MODEL_ROOT / "GLM-5.3-Flash-Original-FP8-eb9eb208",
}
QUANTS = {
    "production": "modelopt_fp4",
    "candidate": "compressed-tensors",
    "orca_fp8": "fp8",
    "original_fp8": "fp8",
}
REVISIONS = {
    "production": "09b04e5e74bca08ca8549fc736d4cdd8624bfde3",
    "candidate": "ec0adf4f49c9570807cc11a5f650538c1893ae54",
    "orca_fp8": "3cec42d6ed14ec197e328c09650c17fd3660c26a",
    "original_fp8": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
}
BASE_ENV = {
    "LOAD_FORMAT": "safetensors",
    "MAX_NUM_SEQS": "16",
    "MAX_NUM_BATCHED_TOKENS": "4096",
    "GPU_MEMORY_UTILIZATION": "0.90",
    "CUDAGRAPH_MODE": "PIECEWISE",
    "MAX_CUDAGRAPH_CAPTURE_SIZE": "128",
    "CUDAGRAPH_CAPTURE_SIZES": "1 2 4 8 16 32 64 128",
    "MAX_MODEL_LEN": "1048576",
    "GLM53_KDA_PREFILL_BACKEND": "flashkda",
    "VLLM_GLM53_ONLINE_DENSE_MXFP8": "0",
    "VLLM_MXFP8_LM_HEAD": "0",
    "VLLM_MTP_NVFP4_LM_HEAD": "0",
    "VLLM_GLM53_MTP_DRAFT_HEAD": "bf16",
    "MTP_MOE_BACKEND": "auto",
    "MOE_BACKEND": "marlin",
    "LINEAR_BACKEND": "triton",
    "VLLM_LM_HEAD_A16": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TZ": "Europe/Berlin",
}
COMMON_ARGS = [
    "--default-chat-template-kwargs",
    '{"reasoning_effort":"high","clear_thinking":false}',
    "--override-generation-config",
    '{"temperature":1.0,"top_p":0.95}',
    "--enable-prompt-tokens-details",
    "--enable-per-request-metrics",
]


def run_cli(label: str, args: list[str], timeout: int = 7200) -> int:
    return rt.run(args, label=label, timeout=timeout, env=rt.PROXY_ENV)


def pin_runtime_image() -> str:
    resolved_image_id = rt.resolve_image(IMAGE)
    os.environ["ORCA_REQUESTED_BATTERY_IMAGE"] = REQUESTED_IMAGE
    os.environ["BATTERY_IMAGE"] = resolved_image_id
    return resolved_image_id


def verify_history_inputs() -> None:
    manifest = json.loads(HISTORY_MANIFEST.read_text())
    root = HISTORY_MANIFEST.parent.resolve()
    missing = []
    for entry in manifest["cases"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            missing.append(entry["path"])
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"History fixture checksum mismatch: {entry['path']}")
    if missing:
        raise RuntimeError(
            "History fixtures are not materialized; restore evidence then run "
            f"{HISTORY_MANIFEST.parent / 'setup.py'} <restored-evidence-root>. "
            f"Missing {len(missing)} file(s)."
        )


def verify_runtime_inputs() -> None:
    missing = []
    for label, model in MODELS.items():
        for name in ("config.json", "model.safetensors.index.json"):
            if not (model / name).is_file():
                missing.append(f"{label}:{model / name}")
    for name in ("config.json", "conversion_manifest.json", "model.safetensors"):
        if not (rt.DRAFT / name).is_file():
            missing.append(f"draft:{rt.DRAFT / name}")
    if missing:
        raise RuntimeError(
            "Required ORCA_MODEL_ROOT checkpoint inputs are missing: " + ", ".join(missing)
        )


def validate_snapshot_receipt(verification: dict) -> None:
    if verification.get("schema") != "orca-snapshot-integrity.v1":
        raise RuntimeError("Unexpected staged-checkpoint verification schema")
    if verification.get("passed") is not True:
        raise RuntimeError("Fresh snapshot integrity/structural preflight has not passed")
    verifier_hash = hashlib.sha256((HERE / "verify_snapshots.py").read_bytes()).hexdigest()
    if verification.get("verifier_sha256") != verifier_hash:
        raise RuntimeError("Staged-checkpoint receipt was not produced by this verifier")
    snapshots = verification.get("snapshots")
    expected_labels = {"candidate", "orca_fp8", "original_fp8"}
    if not isinstance(snapshots, dict) or set(snapshots) != expected_labels:
        raise RuntimeError("Staged-checkpoint verification covers unexpected snapshots")
    finished_at = verification.get("finished_at")
    if not isinstance(finished_at, (int, float)):
        raise RuntimeError("Staged-checkpoint verification has no completion time")
    auxiliary = verification.get("auxiliary_checkpoints")
    draft = auxiliary.get("draft_mxfp8") if isinstance(auxiliary, dict) else None
    if not isinstance(draft, dict) or draft.get("passed") is not True:
        raise RuntimeError("Fresh draft-checkpoint verification is missing or failed")
    if Path(draft.get("directory", "")).resolve() != rt.DRAFT.resolve():
        raise RuntimeError("Draft receipt path differs from ORCA_MODEL_ROOT")
    if draft.get("source_revision") != "dc77ff1c99eeb2df044ee3d4f0094eb033fee410":
        raise RuntimeError("Draft receipt has an unexpected source revision")
    draft_manifest = rt.DRAFT / "conversion_manifest.json"
    if hashlib.sha256(draft_manifest.read_bytes()).hexdigest() != draft.get(
        "conversion_manifest_sha256"
    ):
        raise RuntimeError("Draft conversion manifest changed after verification")
    if (rt.DRAFT / "model.safetensors").stat().st_mtime > finished_at:
        raise RuntimeError("Draft weights changed after verification")
    for label in sorted(expected_labels):
        row = snapshots[label]
        model = MODELS[label].resolve()
        if Path(row.get("directory", "")).resolve() != model:
            raise RuntimeError(f"Snapshot receipt path differs from ORCA_MODEL_ROOT for {label}")
        if row.get("revision") != REVISIONS[label] or row.get("passed") is not True:
            raise RuntimeError(f"Snapshot receipt identity failed for {label}")
        config = model / "config.json"
        index = model / "model.safetensors.index.json"
        if hashlib.sha256(config.read_bytes()).hexdigest() != row.get("config_sha256"):
            raise RuntimeError(f"Snapshot config changed after verification for {label}")
        if hashlib.sha256(index.read_bytes()).hexdigest() != row.get("index_sha256"):
            raise RuntimeError(f"Snapshot index changed after verification for {label}")
        if any(path.stat().st_mtime > finished_at for path in model.glob("*.safetensors")):
            raise RuntimeError(f"Snapshot weights changed after verification for {label}")


def load_snapshot_receipt() -> dict:
    if not SNAPSHOT_RECEIPT.is_file():
        raise RuntimeError(
            "Fresh staged-checkpoint verification is missing; run "
            "campaign/verify_snapshots.py with this ORCA_STATE_DIR first"
        )
    verification = json.loads(SNAPSHOT_RECEIPT.read_text())
    validate_snapshot_receipt(verification)
    return verification


def boot(
    label: str,
    model: str = "candidate",
    spec: str = "mtp0",
    dcp: int = 4,
    cache: str = "vram",
    overrides: dict | None = None,
    eager: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    env = {**BASE_ENV, **(overrides or {})}
    args = [*COMMON_ARGS, "--quantization", QUANTS[model], *(extra_args or ())]
    if eager:
        args.append("--enforce-eager")
    if not rt.boot(
        label,
        image=IMAGE,
        model=MODELS[model],
        tp=4,
        dcp=dcp,
        spec=spec,
        cache=cache,
        kv="fp8_ds_mla",
        extra_env=env,
        extra_args=args,
    ):
        raise RuntimeError(f"{label}: boot failed; complete logs retained")
    inspected = rt._inspect()
    if inspected is None:
        raise RuntimeError(f"{label}: healthy test container disappeared")
    rt.save_json(
        f"{label}.model-contract.json",
        {
            "model": model,
            "model_path": str(MODELS[model]),
            "config_sha256": hashlib.sha256(
                (MODELS[model] / "config.json").read_bytes()
            ).hexdigest(),
            "index_sha256": hashlib.sha256(
                (MODELS[model] / "model.safetensors.index.json").read_bytes()
            ).hexdigest(),
            "revision": REVISIONS[model],
            "quantization": QUANTS[model],
            "source_weights_modified": False,
            "speculation": spec,
            "dcp": dcp,
            "cache": cache,
            "requested_image": REQUESTED_IMAGE,
            "resolved_image_id": inspected["Image"],
        },
    )


def workload(label: str, mode: str, duration: int = 3600) -> int:
    return run_cli(
        label,
        [
            sys.executable,
            str(HERE / "workload_probe.py"),
            "--base-url",
            rt.BASE_URL,
            "--model",
            rt.MODEL_NAME,
            "--label",
            label,
            "--out-dir",
            str(rt.ROOT / label),
            "--mode",
            mode,
            "--duration-seconds",
            str(duration),
        ],
        max(7200, duration + 1800),
    )


def behavior(label: str) -> int:
    return run_cli(
        label,
        [
            sys.executable,
            str(__file__),
            "--behavior",
            "--base-url",
            rt.BASE_URL,
            "--model",
            rt.MODEL_NAME,
            "--fixture",
            str(FIXTURE_ROOT / "behavior/behavior-fixtures.json"),
            "--output-dir",
            str(rt.ROOT / label),
            "--concurrency",
            "4",
        ],
        14400,
    )


def history(label: str) -> int:
    return run_cli(
        label,
        [
            sys.executable,
            str(R29 / "history_probe.py"),
            "--base-url",
            rt.BASE_URL,
            "--model",
            rt.MODEL_NAME,
            "--manifest",
            str(HISTORY_MANIFEST),
            "--output-dir",
            str(rt.ROOT / label),
        ],
        18000,
    )


def visuals_and_needles(label: str, mode: str) -> int:
    return run_cli(
        label,
        [
            sys.executable,
            str(HERE / "extended_probe.py"),
            "--base-url",
            rt.BASE_URL,
            "--model",
            rt.MODEL_NAME,
            "--mode",
            mode,
            "--out-dir",
            str(rt.ROOT / label),
        ],
        14400,
    )


def matrix(label: str, repeats: int = 2) -> list[dict]:
    rows = []
    for trial in range(1, repeats + 1):
        name = f"{label}-trial{trial}"
        with steady_metrics.Recorder(
            rt.BASE_URL, rt.ROOT / f"{name}.steady.metrics.jsonl"
        ):
            passed = rt.bench(
                name, conc="1,4,8", contexts="0,32k,128k", duration=45, prefill=True
            )
        counter_summary = (
            steady_metrics.summarize(rt.ROOT, name)
            if (rt.ROOT / f"{name}.json").exists()
            else None
        )
        emissions_observed = bool(counter_summary and counter_summary["cells"]) and all(
            "generated_tokens" in row.get("counter_delta", {})
            for row in counter_summary["cells"]
        )
        rows.append(
            {
                "trial": trial,
                "execution_passed": passed,
                "emissions_observed": emissions_observed,
                "counter_summary": counter_summary,
            }
        )
    rt.save_json(f"{label}.matrix-summary.json", rows)
    return rows


def collision(label: str) -> int:
    return run_cli(
        label,
        [
            sys.executable,
            str(HERE / "mixed_traffic.py"),
            "--base-url",
            rt.BASE_URL,
            "--model",
            rt.MODEL_NAME,
            "--policies",
            "off,compute_share",
            "--compute-shares",
            "0.4",
            "--concurrencies",
            "1,4,8",
            "--repeats",
            "2",
            "--prefills-per-cell",
            "4",
            "--min-prefill-tokens",
            "32768",
            "--max-prefill-tokens",
            "131072",
            "--output-dir",
            str(rt.ROOT / label),
            "--name",
            label,
        ],
        14400,
    )


def collision_phase() -> dict:
    boot("candidate-collision-mtp0", spec="mtp0")
    return {"collision": collision("candidate-collision")}


def pilot() -> dict:
    boot(
        "pilot-candidate",
        dcp=1,
        eager=True,
        overrides={
            "MAX_MODEL_LEN": "32768",
            "MAX_NUM_SEQS": "4",
            "MAX_NUM_BATCHED_TOKENS": "2048",
        },
    )
    return {
        "api": workload("pilot-api", "api"),
        "vision": visuals_and_needles("pilot-vision", "vision"),
    }


def control() -> dict:
    boot("production-matched-control", model="production", spec="mtp0")
    return {
        "capability": workload("control-capability", "capability"),
        "api": workload("control-api", "api"),
        "behavior": behavior("control-behavior"),
        "vision": visuals_and_needles("control-vision", "vision"),
        "matrix": matrix("control-mtp0"),
    }


def original_reference() -> dict:
    boot(
        "original-fp8-reference",
        model="original_fp8",
        spec="mtp0",
        dcp=1,
        overrides={
            "GPU_MEMORY_UTILIZATION": "0.95",
            "MAX_NUM_BATCHED_TOKENS": "1024",
            "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
            "MAX_MODEL_LEN": "131072",
            "MAX_NUM_SEQS": "8",
            "MAX_CUDAGRAPH_CAPTURE_SIZE": "16",
            "CUDAGRAPH_CAPTURE_SIZES": "1 2 4 8 16",
        },
        extra_args=["--no-async-scheduling"],
    )
    return {
        "capability": workload("original-fp8-capability", "capability"),
        "api": workload("original-fp8-api", "api"),
        "behavior": behavior("original-fp8-behavior"),
        "vision": visuals_and_needles("original-fp8-vision", "vision"),
    }


def ablated_reference() -> dict:
    boot(
        "orca-fp8-reference",
        model="orca_fp8",
        spec="mtp0",
        dcp=1,
        overrides={
            "GPU_MEMORY_UTILIZATION": "0.95",
            "MAX_NUM_BATCHED_TOKENS": "1024",
            "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
            "MAX_MODEL_LEN": "131072",
            "MAX_NUM_SEQS": "8",
            "MAX_CUDAGRAPH_CAPTURE_SIZE": "16",
            "CUDAGRAPH_CAPTURE_SIZES": "1 2 4 8 16",
        },
        extra_args=["--no-async-scheduling"],
    )
    return {
        "capability": workload("orca-fp8-capability", "capability"),
        "api": workload("orca-fp8-api", "api"),
        "behavior": behavior("orca-fp8-behavior"),
        "vision": visuals_and_needles("orca-fp8-vision", "vision"),
    }


def candidate() -> dict:
    boot("candidate-mtp0", spec="mtp0")
    return {
        "capability": workload("candidate-capability", "capability"),
        "api": workload("candidate-api", "api"),
        "behavior": behavior("candidate-behavior"),
        "vision": visuals_and_needles("candidate-vision", "vision"),
        "history": history("candidate-history"),
        "needles": visuals_and_needles("candidate-needles", "needles"),
        "matrix": matrix("candidate-mtp0"),
        "collision": collision("candidate-collision"),
    }


def speculative() -> dict:
    results = {}
    for spec in ("mtp3", "dflash2"):
        label = f"candidate-{spec}"
        try:
            boot(label, spec=spec)
            results[spec] = {
                "boot": True,
                "api": workload(label + "-api", "api"),
                "capability": workload(label + "-capability", "capability"),
                "matrix": matrix(label),
                "vision": visuals_and_needles(label + "-vision", "vision"),
            }
        except Exception as error:
            results[spec] = {"boot": False, "error": repr(error)}
        finally:
            rt.capture(label)
            rt.stop()
            rt.save_json("speculation-progress.json", results)
    return results


def cache_phase() -> dict:
    label = "candidate-lmcache-mtp3"
    l2 = rt.L2_HOST_ROOT / ("orca-" + rt.ROOT.name) / label
    if l2.exists() and any(l2.iterdir()):
        raise RuntimeError("Cache qualification requires a fresh isolated namespace")
    cache_env = {
        "LMCACHE_L2_HOST_DIR": str(l2),
        "LMCACHE_ENABLED": "1",
        "LMCACHE_TRANSFER_MODE": "engine_driven",
        "LMCACHE_L1_SIZE_GB": "8",
        "LMCACHE_L2_ENABLED": "1",
        "LMCACHE_L2_MAX_CAPACITY_GB": "8",
        "LMCACHE_INSTANCE_ID": "orca-" + rt.ROOT.name,
        "LMCACHE_SHM_NAME": "orca-" + rt.ROOT.name,
        "MAX_MODEL_LEN": "262144",
    }
    boot(label, spec="mtp3", cache="lmcache", overrides=cache_env)
    code = run_cli(
        label,
        [
            sys.executable,
            str(R29 / "cache_lifecycle_probe.py"),
            "--base-url",
            rt.BASE_URL,
            "--model",
            rt.MODEL_NAME,
            "--container",
            rt.NAME,
            "--l2-dir",
            str(l2),
            "--output-dir",
            str(rt.ROOT / label),
            "--context-tokens",
            "32768",
            "--growth-turns",
            "6",
        ],
        18000,
    )
    mixed = run_cli(
        label + "-mixed",
        [
            sys.executable,
            str(R29 / "mixed_restore_probe.py"),
            "--base-url",
            rt.BASE_URL,
            "--model",
            rt.MODEL_NAME,
            "--container",
            rt.NAME,
            "--manifest",
            str(HISTORY_MANIFEST),
            "--output-dir",
            str(rt.ROOT / (label + "-mixed")),
        ],
        7200,
    )
    return {"lifecycle": code, "mixed_restore": mixed}


def fidelity() -> dict:
    results = {}
    snapshots = json.loads(SNAPSHOT_RECEIPT.read_text())["snapshots"]
    resolved_image_id = rt.resolve_image(IMAGE)
    for key in ("original_fp8", "orca_fp8", "candidate"):
        rt.stop()
        label = "fidelity-" + key
        args = {
            "tensor_parallel_size": 4,
            "decode_context_parallel_size": 1,
            "cp_kv_cache_interleave_size": 4,
            "dcp_kv_cache_interleave_size": 4,
            "block_size": 256,
            "mamba_cache_mode": "align",
            "dtype": "bfloat16",
            "kv_cache_dtype": "fp8",
            "quantization": QUANTS[key],
            "max_model_len": 4096,
            "max_num_batched_tokens": 256,
            "max_num_seqs": 1,
            "gpu_memory_utilization": 0.92,
            "kv_cache_memory_bytes": 2147483648,
            "enforce_eager": True,
            "load_format": "safetensors",
            "max_logprobs": -1,
            "enable_prefix_caching": False,
            "attention_backend": "B12X",
            "moe_backend": "marlin",
            "linear_backend": "triton",
            "additional_config": {"kda_prefill_backend": "flashkda"},
        }
        args["language_model_only"] = True
        args_path = rt.save_json(
            f"{label}.llm-args.json",
            {
                "metadata": {
                    "checkpoint_repo": snapshots[key]["repo"],
                    "checkpoint_revision": REVISIONS[key],
                    "runtime_image_digest": resolved_image_id,
                    "config_sha256": snapshots[key]["config_sha256"],
                    "global_scale_audit": snapshots[key][
                        "gate_up_global_scale_mismatches"
                    ],
                },
                "llm_args": args,
            },
        )
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
            '"device=0,1,2,3"',
            "--network",
            "none",
            "--shm-size",
            "64g",
            "--memory",
            "180g",
            "--memory-swap",
            "180g",
            "-v",
            f"{MODELS[key]}:/model:ro",
            "-v",
            f"{HERE}:/campaign:ro",
            "-v",
            f"{rt.ROOT}:/evidence",
            "-v",
            f"{FIXTURE_ROOT}:/fixtures:ro",
            "-v",
            "orca-fidelity-runtime-cache:/cache",
            "--entrypoint",
            "/opt/venv/bin/python",
        ]
        env = {
            **BASE_ENV,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NCCL_IB_DISABLE": "1",
            "CUTE_DSL_ARCH": "sm_120a",
            "VLLM_SSM_CONV_STATE_LAYOUT": "DS",
            "VLLM_SERVER_DEV_MODE": "1",
            "NCCL_MIN_NCHANNELS": "16",
            "NCCL_MAX_NCHANNELS": "16",
            "NCCL_BUFFSIZE": "2097152",
            "VLLM_ENABLE_PCIE_ALLREDUCE": "1",
            "VLLM_PCIE_ALLREDUCE_BACKEND": "b12x",
        }
        for name, value in env.items():
            command += ["-e", f"{name}={value}"]
        command += [
            resolved_image_id,
            "/campaign/fidelity_probe.py",
            "capture",
            "--model",
            "/model",
            "--label",
            key,
            "--out-dir",
            f"/evidence/fidelity/{key}",
            "--panels",
            "/fixtures/fidelity/prepared-panels.json",
            "--llm-args",
            "/evidence/" + args_path.name,
        ]
        launched = run_cli(label + "-launch", command, 180)
        if launched:
            results[key] = {
                "launch_returncode": launched,
                "requested_image": REQUESTED_IMAGE,
                "resolved_image_id": resolved_image_id,
            }
            continue
        rt._CURRENT_LABEL = label
        waited = run_cli(label + "-wait", ["docker", "wait", rt.NAME], 18000)
        info = rt._inspect()
        result = {
            "wait_returncode": waited,
            "exit_code": info["State"].get("ExitCode") if info else None,
            "requested_image": REQUESTED_IMAGE,
            "resolved_image_id": resolved_image_id,
        }
        results[key] = result
        rt.capture(label)
        rt.stop()
        rt.save_json("fidelity-progress.json", results)
    comparisons = {}
    for reference, other in [
        ("original_fp8", "orca_fp8"),
        ("orca_fp8", "candidate"),
        ("original_fp8", "candidate"),
    ]:
        label = f"kld-{reference}-to-{other}"
        comparisons[label] = run_cli(
            label,
            [
                sys.executable,
                str(HERE / "fidelity_probe.py"),
                "compare",
                "--reference",
                str(rt.ROOT / "fidelity" / reference),
                "--candidate",
                str(rt.ROOT / "fidelity" / other),
                "--out-dir",
                str(rt.ROOT / label),
            ],
            3600,
        )
    return {"captures": results, "comparisons": comparisons}


def soak() -> dict:
    boot("candidate-soak-mtp3", spec="mtp3")
    return {"soak": workload("candidate-soak", "soak", 3600)}


PHASE_FUNCTIONS = {
    "pilot": pilot,
    "control": control,
    "candidate": candidate,
    "collision": collision_phase,
    "original-reference": original_reference,
    "ablated-reference": ablated_reference,
    "speculative": speculative,
    "cache": cache_phase,
    "fidelity": fidelity,
    "soak": soak,
}


def execution_failures(value: object, path: str = "") -> list[str]:
    failures = []
    code_keys = {
        "api",
        "capability",
        "behavior",
        "vision",
        "history",
        "needles",
        "collision",
        "lifecycle",
        "mixed_restore",
        "soak",
        "launch_returncode",
        "wait_returncode",
        "exit_code",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            location = f"{path}.{key}" if path else key
            if key in code_keys or path.endswith("comparisons"):
                if not isinstance(item, int) or isinstance(item, bool) or item != 0:
                    failures.append(f"{location}={item!r}")
            elif (
                key
                in (
                    "boot",
                    "execution_passed",
                    "all_windows_valid",
                    "emissions_observed",
                )
                and item is not True
            ):
                failures.append(f"{location}={item!r}")
            elif key == "error" and item:
                failures.append(f"{location}={item!r}")
            else:
                failures.extend(execution_failures(item, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            failures.extend(execution_failures(item, f"{path}[{index}]"))
    return failures


def restored_inference() -> None:
    response = rt._HTTP.post(
        "http://127.0.0.1:5001/v1/chat/completions",
        json={
            "model": rt.MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": "What is 17 multiplied by 23? Give the number.",
                }
            ],
            "temperature": 0,
            "max_tokens": 1024,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"].get("content") or ""
    passed = "391" in content
    rt.save_json(
        "production-restored-inference.json", {"passed": passed, "response": data}
    )
    if not passed:
        raise RuntimeError(
            "Production restoration inference failed its arithmetic canary"
        )


def main() -> int:
    if sys.argv[1:2] == ["--behavior"]:
        import behavior_probe

        behavior_probe.SANDBOX_IMAGE = rt.resolve_image(IMAGE)
        sys.argv = [str(R29 / "behavior_probe.py"), *sys.argv[2:]]
        return behavior_probe.main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=list(PHASE_FUNCTIONS))
    parser.add_argument(
        "--stages",
        default="control,original-reference,ablated-reference,candidate,speculative,cache,fidelity,soak",
    )
    args = parser.parse_args()
    verification = load_snapshot_receipt()
    if args.phase:
        manifest = json.loads((rt.ROOT / "source-manifest.json").read_text())
        for filename, expected in manifest.items():
            path = REPO / filename
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Frozen campaign source changed: {filename}")
        outcome = PHASE_FUNCTIONS[args.phase]()
        failures = execution_failures(outcome)
        rt.save_json(
            args.phase + "-execution.json",
            {
                "phase": args.phase,
                "outcome": outcome,
                "execution_passed": not failures,
                "failures": failures,
                "finished_at": time.time(),
                "note": "Behavior scores and unsupported gates still require explicit review.",
            },
        )
        return int(bool(failures))
    stages = args.stages.split(",")
    if any(stage not in PHASE_FUNCTIONS for stage in stages):
        parser.error("Unknown phase")
    if {"candidate", "cache"} & set(stages):
        verify_history_inputs()
    verify_runtime_inputs()
    resolved_image_id = pin_runtime_image()
    if rt._inspect() is not None:
        raise RuntimeError(
            "An existing test container must be investigated, not overwritten"
        )
    if rt.ROOT.exists() and any(rt.ROOT.iterdir()):
        raise RuntimeError(
            "Use a fresh BATTERY_ROOT for every launch; prior evidence is immutable"
        )
    rt.ROOT.mkdir(parents=True)
    rt.save_json(
        "selected-runtime-image.json",
        {"requested_image": REQUESTED_IMAGE, "resolved_image_id": resolved_image_id},
    )
    source_files = [
        Path(__file__),
        HERE / "verify_snapshots.py",
        HERE / "fidelity_probe.py",
        HERE / "workload_probe.py",
        HERE / "extended_probe.py",
        HERE / "mixed_traffic.py",
        R26 / "run_qualification.py",
        R26 / "runtime.py",
        R26 / "steady_metrics.py",
        R26 / "quality_probes.py",
        R29 / "behavior_probe.py",
        R29 / "history_probe.py",
        R29 / "cache_lifecycle_probe.py",
        R29 / "mixed_restore_probe.py",
        REPO / "harness/bench/llm_decode_bench.py",
        FIXTURE_ROOT / "workload_fixtures.json",
        FIXTURE_ROOT / "acceptance.json",
        FIXTURE_ROOT / "capability/tasks.json",
        FIXTURE_ROOT / "behavior/behavior-fixtures.json",
        FIXTURE_ROOT / "history/manifest.json",
        FIXTURE_ROOT / "fidelity/panels.json",
        FIXTURE_ROOT / "fidelity/prepared-panels.json",
        HERE / "runtime-repair/Dockerfile.fp8-exclusions",
        HERE / "runtime-repair/compressed_tensors_moe_w4a4_nvfp4.py",
        HERE / "runtime-repair/fp8.py",
    ]
    rt.save_json(
        "source-manifest.json",
        {
            str(path.resolve().relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    )
    coordinator.PHASES = [
        (stage, str(Path(__file__).resolve()), ["--phase", stage], 21600)
        for stage in stages
    ]
    try:
        coordinator.main()
    finally:
        if (rt.ROOT / "production-restored.json").exists():
            restored_inference()
    progress = json.loads((rt.ROOT / "phase-progress.json").read_text())
    return int(any(row["returncode"] != 0 for row in progress))


if __name__ == "__main__":
    raise SystemExit(main())
