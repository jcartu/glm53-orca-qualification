#!/usr/bin/env python3
"""Verify approved snapshots and tensor contracts without importing a GPU library."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STATE_ROOT = Path(os.environ.get("ORCA_STATE_DIR", REPO / ".local")).expanduser().resolve()
MODEL_ROOT = Path(os.environ.get("ORCA_MODEL_ROOT", REPO / ".local/models")).expanduser().resolve()
ROOT = STATE_ROOT / "snapshot-verification"
SPECS = {
    "candidate": (
        "orcarouter/GLM-5.3-Flash-Uncensored-NVFP4",
        "ec0adf4f49c9570807cc11a5f650538c1893ae54",
        MODEL_ROOT / "Orca-GLM-5.3-Flash-Uncensored-NVFP4-ec0adf4f",
        "compressed-tensors",
    ),
    "orca_fp8": (
        "orcarouter/GLM-5.3-Flash-Uncensored-FP8",
        "3cec42d6ed14ec197e328c09650c17fd3660c26a",
        MODEL_ROOT / "Orca-GLM-5.3-Flash-Uncensored-FP8-3cec42d6",
        "fp8",
    ),
    "original_fp8": (
        "zai-org/GLM-5.3-Flash",
        "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        MODEL_ROOT / "GLM-5.3-Flash-Original-FP8-eb9eb208",
        "fp8",
    ),
}


def save(name, value):
    ROOT.mkdir(parents=True, exist_ok=True)
    path = ROOT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_tensors(path: Path):
    index = json.loads((path / "model.safetensors.index.json").read_text())
    names = index["weight_map"]
    tensors, scalars, errors = {}, {}, []
    total_size = 0
    for shard in sorted(set(names.values())):
        file = path / shard
        total_size += file.stat().st_size
        with file.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if header_size > 128 * 1024**2:
                raise ValueError(f"Unreasonable safetensors header: {file}")
            header = json.loads(handle.read(header_size))
            data_start = 8 + header_size
            data_size = file.stat().st_size - data_start
            for name, entry in header.items():
                if name == "__metadata__":
                    continue
                if name in tensors:
                    errors.append(f"duplicate tensor {name}")
                offsets = entry["data_offsets"]
                if not 0 <= offsets[0] <= offsets[1] <= data_size:
                    errors.append(f"invalid offsets {name}")
                if names.get(name) != shard:
                    errors.append(f"index disagrees for {name}")
                tensors[name] = {"dtype": entry["dtype"], "shape": entry["shape"]}
                if (
                    name.endswith("weight_global_scale")
                    and entry["dtype"] == "F32"
                    and offsets[1] - offsets[0] == 4
                ):
                    handle.seek(data_start + offsets[0])
                    scalars[name] = struct.unpack("<f", handle.read(4))[0]
    errors.extend(
        f"missing indexed tensor {name}" for name in names.keys() - tensors.keys()
    )
    return {
        "tensor_count": len(tensors),
        "shards": len(set(names.values())),
        "file_bytes": total_size,
        "indexed_tensor_bytes": index.get("metadata", {}).get("total_size"),
        "mtp_tensors": sum(".layers.45." in name for name in tensors),
        "vision_tensors": sum(
            "visual." in name or "vision." in name for name in tensors
        ),
        "fp32_aux_tensors": sum(value["dtype"] == "F32" for value in tensors.values()),
        "errors": errors,
        "tensors": tensors,
        "global_scales": scalars,
    }


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / "staging-verification.json").exists():
        raise RuntimeError(
            "Preserve existing fresh verification; use a new ORCA_STATE_DIR for another audit"
        )
    results, headers = {}, {}
    for label, (repo, revision, directory, expected_quant) in SPECS.items():
        path = Path(directory)
        command = [
            "hf",
            "cache",
            "verify",
            repo,
            "--revision",
            revision,
            "--local-dir",
            str(directory),
            "--fail-on-missing-files",
            "--json",
        ]
        started = time.time()
        with (ROOT / f"{label}.hf-verification.log").open("w") as output:
            run = subprocess.run(
                command,
                stdout=output,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=14400,
            )
        returncode = run.returncode
        config_bytes = (path / "config.json").read_bytes()
        config = json.loads(config_bytes)
        actual_quant = config.get("quantization_config", {}).get("quant_method")
        header = inspect_tensors(path)
        headers[label] = header["tensors"]
        header_path = f"{label}.tensor-schema.json"
        save(header_path, header["tensors"])
        scalars = header.pop("global_scales")
        header.pop("tensors")
        mismatches = []
        for name, value in scalars.items():
            if ".gate_proj." not in name:
                continue
            partner = name.replace(".gate_proj.", ".up_proj.")
            if partner in scalars and abs(value - scalars[partner]) > 1e-8 + 1e-5 * abs(
                scalars[partner]
            ):
                mismatches.append(
                    {"gate": name, "gate_scale": value, "up_scale": scalars[partner]}
                )
        results[label] = {
            "repo": repo,
            "revision": revision,
            "directory": str(directory),
            "verification_returncode": returncode,
            "verified_at": time.time(),
            "elapsed_seconds": time.time() - started,
            "quantization": actual_quant,
            "expected_quantization": expected_quant,
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "index_sha256": hashlib.sha256(
                (path / "model.safetensors.index.json").read_bytes()
            ).hexdigest(),
            "schema_file": header_path,
            "tensor_contract": header,
            "gate_up_global_scale_mismatches": {
                "count": len(mismatches),
                "examples": mismatches[:20],
            },
            "passed": returncode == 0
            and actual_quant == expected_quant
            and not header["errors"]
            and header["mtp_tensors"] > 0,
        }
        save("staging-progress.json", results)
        print(
            json.dumps({k: results[label][k] for k in ("repo", "revision", "passed")}),
            flush=True,
        )
    draft_path = MODEL_ROOT / "GLM-5.3-Flash-DFlash2-MXFP8"
    draft_manifest_path = draft_path / "conversion_manifest.json"
    draft_manifest_bytes = draft_manifest_path.read_bytes()
    draft_manifest = json.loads(draft_manifest_bytes)
    expected_files = draft_manifest["output_files_sha256"]
    observed_files = {
        name: sha256_file(draft_path / name) for name in sorted(expected_files)
    }
    draft = {
        "directory": str(draft_path),
        "conversion_manifest_sha256": hashlib.sha256(draft_manifest_bytes).hexdigest(),
        "source_model": draft_manifest["source"]["model"],
        "source_revision": draft_manifest["source"]["revision"],
        "expected_files_sha256": expected_files,
        "observed_files_sha256": observed_files,
        "passed": observed_files == expected_files,
    }
    original, ablated = headers["original_fp8"], headers["orca_fp8"]
    different = [
        name
        for name in original.keys() | ablated.keys()
        if original.get(name) != ablated.get(name)
    ]
    summary = {
        "schema": "orca-snapshot-integrity.v1",
        "snapshots": results,
        "auxiliary_checkpoints": {"draft_mxfp8": draft},
        "fp8_structural_parity": {
            "passed": not different,
            "different_count": len(different),
            "examples": different[:30],
        },
        "passed": all(row["passed"] for row in results.values()) and not different and draft["passed"],
        "finished_at": time.time(),
        "weights_modified": False,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save("staging-verification.json", summary)
    return int(not summary["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
