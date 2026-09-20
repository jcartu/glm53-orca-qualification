#!/usr/bin/env python3
"""Check that unedited FP8 reader tensors match the selected original exactly."""

import hashlib
import json
import mmap
import os
from pathlib import Path
import struct
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STATE_ROOT = Path(os.environ.get("ORCA_STATE_DIR", REPO / ".local")).expanduser().resolve()
MODEL_ROOT = Path(os.environ.get("ORCA_MODEL_ROOT", REPO / ".local/models")).expanduser().resolve()
ORIGINAL = MODEL_ROOT / "GLM-5.3-Flash-Original-FP8-eb9eb208"
ABLATED = MODEL_ROOT / "Orca-GLM-5.3-Flash-Uncensored-FP8-3cec42d6"


def header(handle):
    size = struct.unpack("<Q", handle.read(8))[0]
    return json.loads(handle.read(size)), 8 + size


def tensor_hash(mapping, start, end):
    digest = hashlib.sha256()
    for offset in range(start, end, 8 * 1024**2):
        view = memoryview(mapping)[offset : min(end, offset + 8 * 1024**2)]
        digest.update(view)
        view.release()
    return digest.hexdigest()


def main():
    destination = STATE_ROOT / "lineage/fp8-lineage.json"
    if destination.exists():
        raise RuntimeError("Refusing to overwrite a completed lineage audit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    original_index = json.loads(
        (ORIGINAL / "model.safetensors.index.json").read_text()
    )["weight_map"]
    ablated_index = json.loads((ABLATED / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    if original_index != ablated_index:
        raise RuntimeError(
            "Reference shard/tensor layout differs; cannot use paired-shard audit"
        )
    started = time.time()
    checked, checked_bytes, excluded = 0, 0, 0
    mismatches = []
    shard_rows = []
    for shard in sorted(set(original_index.values())):
        with (
            (ORIGINAL / shard).open("rb") as original_file,
            (ABLATED / shard).open("rb") as ablated_file,
        ):
            original_header, original_start = header(original_file)
            ablated_header, ablated_start = header(ablated_file)
            with (
                mmap.mmap(
                    original_file.fileno(), 0, access=mmap.ACCESS_READ
                ) as original_map,
                mmap.mmap(
                    ablated_file.fileno(), 0, access=mmap.ACCESS_READ
                ) as ablated_map,
            ):
                shard_checked = 0
                for name, entry in original_header.items():
                    if name == "__metadata__":
                        continue
                    if any(
                        piece in name
                        for piece in (
                            ".down_proj.",
                            ".o_proj.",
                            ".eh_proj.",
                            ".embed_tokens.",
                        )
                    ):
                        excluded += 1
                        continue
                    other = ablated_header[name]
                    if (
                        entry["dtype"] != other["dtype"]
                        or entry["shape"] != other["shape"]
                    ):
                        raise RuntimeError(f"Unexpected structural mismatch: {name}")
                    a0, a1 = entry["data_offsets"]
                    b0, b1 = other["data_offsets"]
                    original_hash = tensor_hash(
                        original_map, original_start + a0, original_start + a1
                    )
                    ablated_hash = tensor_hash(
                        ablated_map, ablated_start + b0, ablated_start + b1
                    )
                    checked += 1
                    shard_checked += 1
                    checked_bytes += a1 - a0
                    if original_hash != ablated_hash:
                        mismatches.append(
                            {
                                "name": name,
                                "original_sha256": original_hash,
                                "ablated_sha256": ablated_hash,
                            }
                        )
                shard_rows.append({"shard": shard, "checked_tensors": shard_checked})
        print(
            json.dumps(
                {"shard": shard, "checked": checked, "mismatches": len(mismatches)}
            ),
            flush=True,
        )
    result = {
        "schema": "orca-fp8-lineage.v1",
        "passed": not mismatches,
        "checked_tensors": checked,
        "checked_bytes_per_checkpoint": checked_bytes,
        "excluded_writer_tensors": excluded,
        "mismatches": mismatches,
        "shards": shard_rows,
        "elapsed_seconds": time.time() - started,
        "original_revision": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        "ablated_revision": "3cec42d6ed14ec197e328c09650c17fd3660c26a",
        "scope": "Byte identity of all tensors outside the broad declared residual-writer families (down_proj, o_proj, eh_proj, embeddings), including their reader scales. Supports source-reference provenance, not semantic equivalence or a complete proof of the ablation algorithm.",
    }
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(destination)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
