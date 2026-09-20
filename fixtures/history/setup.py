#!/usr/bin/env python3
"""Materialize large history fixtures from a downloaded/restored evidence tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVENTORY_PATH = HERE / "inventory.json"
MANIFEST_PATH = HERE / "manifest.json"
SOURCE_RELATIVE = Path("publication-inputs/history")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, expected_size: int, expected_hash: str) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == expected_size
        and sha256(path) == expected_hash
    )


def materialize(source: Path, destination: Path, entry: dict) -> str:
    name = entry["path"]
    if Path(name).name != name:
        raise ValueError(f"unsafe inventory path: {name!r}")
    expected_size = int(entry["bytes"])
    expected_hash = str(entry["sha256"])
    target = destination / name
    if target.exists() or target.is_symlink():
        if checked_file(target, expected_size, expected_hash):
            return "already-present"
        raise RuntimeError(f"refusing to overwrite mismatched fixture: {target}")

    origin = source / name
    if not origin.is_file() or origin.is_symlink():
        raise FileNotFoundError(f"missing regular source fixture: {origin}")
    if origin.stat().st_size != expected_size:
        raise RuntimeError(f"source size mismatch: {origin}")

    temporary = destination / f".{name}.partial"
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"inspect and remove stale partial file first: {temporary}")
    digest = hashlib.sha256()
    copied = 0
    try:
        with origin.open("rb") as reader, temporary.open("xb") as writer:
            while chunk := reader.read(1024 * 1024):
                writer.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if copied != expected_size or digest.hexdigest() != expected_hash:
            raise RuntimeError(f"source checksum mismatch: {origin}")
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return "copied"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the exact history inputs from <restored-evidence>/publication-inputs/history "
            "into fixtures/history. Restore/download evidence first with tools/download_evidence.py."
        )
    )
    parser.add_argument("evidence_root", type=Path)
    args = parser.parse_args()

    inventory_bytes = INVENTORY_PATH.read_bytes()
    inventory = json.loads(inventory_bytes)
    manifest_bytes = MANIFEST_PATH.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != inventory["manifest_sha256"]:
        raise RuntimeError("tracked history manifest checksum mismatch")
    manifest = json.loads(manifest_bytes)
    manifest_hashes = {row["path"]: row["sha256"] for row in manifest["cases"]}
    inventory_hashes = {row["path"]: row["sha256"] for row in inventory["files"]}
    if manifest_hashes != inventory_hashes:
        raise RuntimeError("history manifest and inventory disagree")

    source = (args.evidence_root.expanduser().resolve() / SOURCE_RELATIVE)
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(f"restored history directory is missing: {source}")
    source_manifest = source / "manifest.json"
    if not source_manifest.is_file() or source_manifest.is_symlink():
        raise FileNotFoundError(f"restored history manifest is missing: {source_manifest}")
    if sha256(source_manifest) != inventory["manifest_sha256"]:
        raise RuntimeError("restored history manifest checksum mismatch")

    outcomes = {"copied": 0, "already-present": 0}
    for entry in inventory["files"]:
        outcome = materialize(source, HERE, entry)
        outcomes[outcome] += 1
    print(json.dumps({"destination": str(HERE), "files": len(inventory["files"]), **outcomes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
