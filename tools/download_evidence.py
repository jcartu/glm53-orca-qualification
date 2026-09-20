#!/usr/bin/env python3
"""Download, verify, and atomically unpack qualification evidence assets."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import BinaryIO

SCHEMA = "glm53-orca-evidence.v1"
GITHUB_ASSET_LIMIT = 2 * 1024**3
MAX_RELEASE_ASSETS = 1000
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
CHUNK = 4 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ASSET_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,255}$")
TAR_BLOCK_SIZE = 512
TAR_RECORD_SIZE = TAR_BLOCK_SIZE * 20
MAX_GNU_LONGNAME_BYTES = 16 * 1024
TAR_GNU_MAGIC = b"ustar  \0"
WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
    | {f"COM{suffix}" for suffix in "123456789¹²³"}
    | {f"LPT{suffix}" for suffix in "123456789¹²³"}
)


FILE_KEYS = {
    "asset",
    "exported_bytes",
    "exported_sha256",
    "logical_path",
    "redaction_classification",
    "source_bytes",
    "source_sha256",
}
ASSET_KEYS = {
    "file_count",
    "format",
    "logical_extraction_root",
    "name",
    "sha256",
    "size",
    "unpacked_bytes",
}
COVERAGE_KEYS = {
    "discovered_files",
    "discovered_source_bytes",
    "excluded_files",
    "excluded_source_bytes",
    "exported_bytes",
    "exported_files",
    "exported_source_bytes",
}
TOP_LEVEL_KEYS = {
    "assets",
    "coverage",
    "exclusions",
    "files",
    "redaction_ledger",
    "schema",
    "source_root_label",
}
EXCLUSION_KEYS = {"category", "logical_path", "reason", "source_bytes"}
LEDGER_KEYS = {"category_counts", "files"}
LEDGER_FILE_KEYS = {"category_counts", "logical_path"}
REDACTION_CLASSIFICATIONS = {
    "none",
    "redacted-credentials",
    "sanitized-and-redacted",
    "sanitized-machine-identifiers",
}
MACHINE_CATEGORIES = {
    "cache_path",
    "gzip_metadata",
    "home_path",
    "lan_address",
    "model_path",
    "source_root_path",
}
CREDENTIAL_CATEGORIES = {
    "authorization_header",
    "credential_assignment",
    "credential_field",
    "private_key_block",
    "url_credential",
}
LEDGER_CATEGORIES = (
    MACHINE_CATEGORIES | CREDENTIAL_CATEGORIES | {"synthetic_canary_preserved"}
)
EXCLUSION_REASONS = {
    "generated-cache": "Generated cache or bytecode; reproducible from the exported source.",
    "model-weight": "Model weight; weights are distributed separately and are not evidence artifacts.",
    "private-authorization-material": "Private authorization material is never published.",
    "vendor-runtime-binary": "Compiled vendor runtime binary obtainable from the pinned runtime image.",
}


class DownloadError(RuntimeError):
    pass


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        file_pointer: BinaryIO,
        code: int,
        message: str,
        headers: object,
        new_url: str,
    ) -> urllib.request.Request | None:
        parsed = urllib.parse.urlsplit(new_url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            raise DownloadError(
                "release asset redirect must remain credential-free HTTPS"
            )
        return super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )


def exact_keys(value: dict[str, object], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DownloadError(
            f"{context} has wrong fields (missing={missing}, extra={extra})"
        )


def require_dict(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DownloadError(f"{context} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise DownloadError(f"{context} contains a non-string key")
    return value


def require_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise DownloadError(f"{context} must be an array")
    return value


def require_string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise DownloadError(f"{context} must be a string")
    return value


def require_nonnegative_int(value: object, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DownloadError(f"{context} must be a nonnegative integer")
    return value


def require_positive_int(value: object, context: str) -> int:
    result = require_nonnegative_int(value, context)
    if result == 0:
        raise DownloadError(f"{context} must be positive")
    return result


def require_sha256(value: object, context: str) -> str:
    result = require_string(value, context)
    if not SHA256_RE.fullmatch(result):
        raise DownloadError(f"{context} must be a lowercase SHA-256 digest")
    return result


def is_windows_device_name(component: str) -> bool:
    return component.partition(".")[0].rstrip(" ").upper() in WINDOWS_DEVICE_NAMES


def safe_logical_path(value: object, context: str) -> str:
    result = require_string(value, context)
    if (
        not result
        or result.startswith("/")
        or "\\" in result
        or ":" in result
        or any(ord(character) < 32 or ord(character) == 127 for character in result)
    ):
        raise DownloadError(f"{context} is not a safe portable relative path")
    parts = result.split("/")
    if any(
        part in {"", ".", ".."}
        or part.endswith((".", " "))
        or is_windows_device_name(part)
        for part in parts
    ):
        raise DownloadError(f"{context} is not a safe portable relative path")
    if PurePosixPath(result).as_posix() != result:
        raise DownloadError(f"{context} is not canonical")
    return result


def safe_asset_name(value: object, context: str) -> str:
    result = require_string(value, context)
    if (
        not ASSET_NAME_RE.fullmatch(result)
        or result in {".", ".."}
        or result.endswith((".", " "))
        or is_windows_device_name(result)
    ):
        raise DownloadError(f"{context} is not a safe portable asset filename")
    return result


def validate_counts(value: object, context: str) -> dict[str, int]:
    counts_object = require_dict(value, context)
    unknown = set(counts_object) - LEDGER_CATEGORIES
    if unknown:
        raise DownloadError(f"{context} has unknown categories: {sorted(unknown)}")
    counts: dict[str, int] = {}
    for category, count in counts_object.items():
        counts[category] = require_positive_int(count, f"{context}.{category}")
    return counts


def expected_classification(counts: dict[str, int]) -> str:
    machine = bool(set(counts) & MACHINE_CATEGORIES)
    credential = bool(set(counts) & CREDENTIAL_CATEGORIES)
    if machine and credential:
        return "sanitized-and-redacted"
    if credential:
        return "redacted-credentials"
    if machine:
        return "sanitized-machine-identifiers"
    return "none"


def load_manifest(path: Path) -> dict[str, object]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DownloadError(f"cannot stat manifest: {exc}") from exc
    if size > MAX_MANIFEST_BYTES:
        raise DownloadError(f"manifest exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DownloadError(f"cannot read manifest: {exc}") from exc
    return validate_manifest(value)


def validate_manifest(value: object) -> dict[str, object]:
    manifest = require_dict(value, "manifest")
    exact_keys(manifest, TOP_LEVEL_KEYS, "manifest")
    if manifest["schema"] != SCHEMA:
        raise DownloadError(f"unsupported manifest schema: {manifest['schema']!r}")
    label = require_string(manifest["source_root_label"], "source_root_label")
    if (
        not label
        or "/" in label
        or "\\" in label
        or any(ord(char) < 32 for char in label)
    ):
        raise DownloadError("source_root_label is unsafe")

    raw_assets = require_list(manifest["assets"], "assets")
    if len(raw_assets) > MAX_RELEASE_ASSETS:
        raise DownloadError(
            f"manifest exceeds the {MAX_RELEASE_ASSETS}-asset release limit"
        )
    assets: list[dict[str, object]] = []
    assets_by_name: dict[str, dict[str, object]] = {}
    for index, raw_asset in enumerate(raw_assets):
        asset = require_dict(raw_asset, f"assets[{index}]")
        exact_keys(asset, ASSET_KEYS, f"assets[{index}]")
        name = safe_asset_name(asset["name"], f"assets[{index}].name")
        if name in assets_by_name:
            raise DownloadError(f"duplicate asset name: {name}")
        size = require_positive_int(asset["size"], f"assets[{index}].size")
        if size >= GITHUB_ASSET_LIMIT:
            raise DownloadError(f"asset {name} is not smaller than 2 GiB")
        require_sha256(asset["sha256"], f"assets[{index}].sha256")
        format_name = require_string(asset["format"], f"assets[{index}].format")
        if format_name not in {"tar+zstd", "npy+zstd"}:
            raise DownloadError(f"unsupported asset format for {name}: {format_name}")
        if asset["logical_extraction_root"] != ".":
            raise DownloadError(
                f"asset {name} has an unsupported logical extraction root"
            )
        require_positive_int(asset["file_count"], f"assets[{index}].file_count")
        require_nonnegative_int(
            asset["unpacked_bytes"], f"assets[{index}].unpacked_bytes"
        )
        assets.append(asset)
        assets_by_name[name] = asset

    raw_files = require_list(manifest["files"], "files")
    files: list[dict[str, object]] = []
    files_by_path: dict[str, dict[str, object]] = {}
    files_by_asset: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    for index, raw_file in enumerate(raw_files):
        file_entry = require_dict(raw_file, f"files[{index}]")
        exact_keys(file_entry, FILE_KEYS, f"files[{index}]")
        logical = safe_logical_path(
            file_entry["logical_path"], f"files[{index}].logical_path"
        )
        if logical in files_by_path:
            raise DownloadError(f"duplicate logical path: {logical}")
        source_bytes = require_nonnegative_int(
            file_entry["source_bytes"], f"files[{index}].source_bytes"
        )
        exported_bytes = require_nonnegative_int(
            file_entry["exported_bytes"], f"files[{index}].exported_bytes"
        )
        source_sha = require_sha256(
            file_entry["source_sha256"], f"files[{index}].source_sha256"
        )
        exported_sha = require_sha256(
            file_entry["exported_sha256"], f"files[{index}].exported_sha256"
        )
        asset_name = safe_asset_name(file_entry["asset"], f"files[{index}].asset")
        if asset_name not in assets_by_name:
            raise DownloadError(f"file {logical} refers to unknown asset {asset_name}")
        classification = require_string(
            file_entry["redaction_classification"],
            f"files[{index}].redaction_classification",
        )
        if classification not in REDACTION_CLASSIFICATIONS:
            raise DownloadError(
                f"file {logical} has an unknown redaction classification"
            )
        if classification == "none" and (
            source_bytes != exported_bytes or source_sha != exported_sha
        ):
            raise DownloadError(
                f"unredacted file {logical} does not preserve source bytes exactly"
            )
        files.append(file_entry)
        files_by_path[logical] = file_entry
        files_by_asset[asset_name].append(file_entry)

    for name, asset in assets_by_name.items():
        owned = files_by_asset.get(name, [])
        if len(owned) != asset["file_count"]:
            raise DownloadError(
                f"asset {name} file_count does not match its file entries"
            )
        if (
            sum(int(item["exported_bytes"]) for item in owned)
            != asset["unpacked_bytes"]
        ):
            raise DownloadError(
                f"asset {name} unpacked_bytes does not match its file entries"
            )
        if asset["format"] == "npy+zstd":
            if len(owned) != 1 or not str(owned[0]["logical_path"]).lower().endswith(
                ".npy"
            ):
                raise DownloadError(
                    f"numeric asset {name} must own exactly one .npy file"
                )
            item = owned[0]
            if (
                item["redaction_classification"] != "none"
                or item["source_bytes"] != item["exported_bytes"]
                or item["source_sha256"] != item["exported_sha256"]
            ):
                raise DownloadError(f"numeric asset {name} is not byte-preserving")
        elif any(str(item["logical_path"]).lower().endswith(".npy") for item in owned):
            raise DownloadError(f"numeric .npy files must use npy+zstd assets: {name}")

    raw_exclusions = require_list(manifest["exclusions"], "exclusions")
    exclusions: list[dict[str, object]] = []
    exclusion_paths: set[str] = set()
    for index, raw_exclusion in enumerate(raw_exclusions):
        exclusion = require_dict(raw_exclusion, f"exclusions[{index}]")
        exact_keys(exclusion, EXCLUSION_KEYS, f"exclusions[{index}]")
        logical = safe_logical_path(
            exclusion["logical_path"], f"exclusions[{index}].logical_path"
        )
        if logical in files_by_path or logical in exclusion_paths:
            raise DownloadError(f"duplicate or conflicting excluded path: {logical}")
        category = require_string(
            exclusion["category"], f"exclusions[{index}].category"
        )
        if category not in EXCLUSION_REASONS:
            raise DownloadError(f"unknown exclusion category: {category}")
        if exclusion["reason"] != EXCLUSION_REASONS[category]:
            raise DownloadError(
                f"exclusion reason does not match category for {logical}"
            )
        require_nonnegative_int(
            exclusion["source_bytes"], f"exclusions[{index}].source_bytes"
        )
        exclusion_paths.add(logical)
        exclusions.append(exclusion)

    ledger = require_dict(manifest["redaction_ledger"], "redaction_ledger")
    exact_keys(ledger, LEDGER_KEYS, "redaction_ledger")
    aggregate_declared = validate_counts(
        ledger["category_counts"], "redaction_ledger.category_counts"
    )
    raw_ledger_files = require_list(ledger["files"], "redaction_ledger.files")
    per_file_counts: dict[str, dict[str, int]] = {}
    aggregate_actual: collections.Counter[str] = collections.Counter()
    for index, raw_ledger_file in enumerate(raw_ledger_files):
        ledger_file = require_dict(raw_ledger_file, f"redaction_ledger.files[{index}]")
        exact_keys(ledger_file, LEDGER_FILE_KEYS, f"redaction_ledger.files[{index}]")
        logical = safe_logical_path(
            ledger_file["logical_path"], f"redaction_ledger.files[{index}].logical_path"
        )
        if logical not in files_by_path:
            raise DownloadError(f"redaction ledger refers to unknown file: {logical}")
        if logical in per_file_counts:
            raise DownloadError(f"duplicate redaction ledger path: {logical}")
        counts = validate_counts(
            ledger_file["category_counts"],
            f"redaction_ledger.files[{index}].category_counts",
        )
        if not counts:
            raise DownloadError(f"redaction ledger entry for {logical} is empty")
        per_file_counts[logical] = counts
        aggregate_actual.update(counts)
    if dict(sorted(aggregate_actual.items())) != dict(
        sorted(aggregate_declared.items())
    ):
        raise DownloadError(
            "redaction ledger aggregate does not equal its per-file counts"
        )
    for logical, file_entry in files_by_path.items():
        classification = expected_classification(per_file_counts.get(logical, {}))
        if file_entry["redaction_classification"] != classification:
            raise DownloadError(
                f"redaction classification disagrees with ledger for {logical}"
            )

    coverage = require_dict(manifest["coverage"], "coverage")
    exact_keys(coverage, COVERAGE_KEYS, "coverage")
    coverage_values = {
        key: require_nonnegative_int(value, f"coverage.{key}")
        for key, value in coverage.items()
    }
    expected_coverage = {
        "discovered_files": len(files) + len(exclusions),
        "exported_files": len(files),
        "excluded_files": len(exclusions),
        "discovered_source_bytes": sum(int(item["source_bytes"]) for item in files)
        + sum(int(item["source_bytes"]) for item in exclusions),
        "exported_source_bytes": sum(int(item["source_bytes"]) for item in files),
        "excluded_source_bytes": sum(int(item["source_bytes"]) for item in exclusions),
        "exported_bytes": sum(int(item["exported_bytes"]) for item in files),
    }
    if coverage_values != expected_coverage:
        raise DownloadError("coverage totals do not match files and exclusions")
    if bool(files) != bool(assets):
        raise DownloadError(
            "files and assets must either both be empty or both be nonempty"
        )

    manifest["assets"] = assets
    manifest["files"] = files
    manifest["exclusions"] = exclusions
    return manifest


def validate_release_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme not in {"https", "file"}:
        raise DownloadError("release base URL must use https:// or file://")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise DownloadError(
            "release base URL must not contain credentials, a query, or a fragment"
        )
    if parsed.scheme == "https" and not parsed.netloc:
        raise DownloadError("HTTPS release base URL must include a host")
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise DownloadError("file release base URL must be local")
        if not urllib.parse.unquote(parsed.path).startswith("/"):
            raise DownloadError("file release base URL must be absolute")
    return value.rstrip("/") + "/"


def asset_url(base_url: str, name: str) -> str:
    return base_url + urllib.parse.quote(name, safe="-._~")


def download_asset(
    opener: urllib.request.OpenerDirector,
    url: str,
    destination: Path,
    expected_size: int,
    expected_sha256: str,
    timeout: float,
) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "glm53-orca-evidence-downloader/1"},
        method="GET",
    )
    digest = hashlib.sha256()
    count = 0
    created = False
    try:
        with opener.open(request, timeout=timeout) as response:
            with destination.open("xb") as output:
                created = True
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None:
                    try:
                        declared = int(declared_length)
                    except ValueError as exc:
                        raise DownloadError(
                            "asset response has an invalid Content-Length"
                        ) from exc
                    if declared != expected_size:
                        raise DownloadError(
                            f"asset Content-Length mismatch: expected {expected_size}, "
                            f"received {declared}"
                        )
                while chunk := response.read(CHUNK):
                    count += len(chunk)
                    if count > expected_size:
                        raise DownloadError(
                            "downloaded asset exceeds its manifest size"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        if count != expected_size:
            raise DownloadError(
                f"downloaded asset size mismatch: expected {expected_size}, received {count}"
            )
        if digest.hexdigest() != expected_sha256:
            raise DownloadError("downloaded asset SHA-256 mismatch")
        destination.chmod(0o600)
    except BaseException as exc:
        if created:
            destination.unlink(missing_ok=True)
        if isinstance(exc, DownloadError):
            raise
        if isinstance(exc, (OSError, urllib.error.URLError)):
            raise DownloadError(f"asset download failed: {exc}") from exc
        raise


def target_path(root: Path, logical: str) -> Path:
    safe_logical_path(logical, "archive path")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*logical.split("/")).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise DownloadError(f"archive path escapes extraction root: {logical}") from exc
    return candidate


def write_verified_stream(
    source: BinaryIO, destination: Path, entry: dict[str, object]
) -> None:
    expected_size = int(entry["exported_bytes"])
    expected_sha = str(entry["exported_sha256"])
    digest = hashlib.sha256()
    count = 0
    created = False
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as output:
            created = True
            while count < expected_size:
                chunk = source.read(min(CHUNK, expected_size - count))
                if not chunk:
                    break
                count += len(chunk)
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if count != expected_size:
            raise DownloadError(
                f"unpacked file size mismatch for {entry['logical_path']}: "
                f"expected {expected_size}, received {count}"
            )
        if digest.hexdigest() != expected_sha:
            raise DownloadError(
                f"unpacked file SHA-256 mismatch: {entry['logical_path']}"
            )
        destination.chmod(0o644)
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise


class BoundedTarReader:
    def __init__(self, source: BinaryIO, limit: int, asset_name: str) -> None:
        self.source = source
        self.limit = limit
        self.asset_name = asset_name
        self.count = 0

    def read(self, size: int) -> bytes:
        if size < 0 or size > self.limit - self.count:
            raise DownloadError(
                f"decompressed tar exceeds its bounded expansion: {self.asset_name}"
            )
        chunk = self.source.read(size)
        self.count += len(chunk)
        return chunk


def read_exact(source: BoundedTarReader, size: int, context: str) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = source.read(size - len(result))
        if not chunk:
            raise DownloadError(f"truncated {context}")
        result.extend(chunk)
    return bytes(result)


def tar_padded_size(size: int) -> int:
    return ((size + TAR_BLOCK_SIZE - 1) // TAR_BLOCK_SIZE) * TAR_BLOCK_SIZE


def tar_expansion_limit(entries: list[dict[str, object]]) -> int:
    file_bytes = sum(
        TAR_BLOCK_SIZE + tar_padded_size(int(entry["exported_bytes"]))
        for entry in entries
    )
    metadata_bytes = len(entries) * (
        TAR_BLOCK_SIZE + tar_padded_size(MAX_GNU_LONGNAME_BYTES)
    )
    max_end_bytes = TAR_RECORD_SIZE + TAR_BLOCK_SIZE
    return file_bytes + metadata_bytes + max_end_bytes + 1


def parse_tar_header(block: bytes, asset_name: str) -> tarfile.TarInfo:
    try:
        member = tarfile.TarInfo.frombuf(
            block, encoding="utf-8", errors="surrogateescape"
        )
    except (tarfile.HeaderError, ValueError) as exc:
        raise DownloadError(f"invalid tar header in {asset_name}: {exc}") from exc
    if block[257:265] != TAR_GNU_MAGIC:
        raise DownloadError(f"asset {asset_name} contains a non-GNU tar header")
    return member


def start_zstd(zstd: str, asset_path: Path) -> subprocess.Popen[bytes]:
    process = subprocess.Popen(
        [zstd, "-q", "-d", "-c", "--", str(asset_path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    if process.stdout is None or process.stderr is None:
        process.kill()
        process.wait()
        raise DownloadError("failed to create zstd pipes")
    return process


def finish_zstd(process: subprocess.Popen[bytes], asset_name: str) -> None:
    assert process.stderr is not None
    stderr = process.stderr.read()
    returncode = process.wait()
    if returncode:
        message = stderr.decode("utf-8", "replace").strip()
        raise DownloadError(f"zstd rejected {asset_name} ({returncode}): {message}")


def abort_zstd(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    process.wait()


def extract_numeric_asset(
    zstd: str,
    asset_path: Path,
    tree: Path,
    entry: dict[str, object],
) -> None:
    process = start_zstd(zstd, asset_path)
    assert process.stdout is not None
    try:
        destination = target_path(tree, str(entry["logical_path"]))
        write_verified_stream(process.stdout, destination, entry)
        trailing = process.stdout.read(1)
        if trailing:
            raise DownloadError(
                f"numeric asset has trailing unpacked data: {asset_path.name}"
            )
        finish_zstd(process, asset_path.name)
    except BaseException:
        abort_zstd(process)
        raise
    finally:
        process.stdout.close()


def extract_tar_asset(
    zstd: str,
    asset_path: Path,
    tree: Path,
    expected_entries: list[dict[str, object]],
) -> None:
    expected = {str(entry["logical_path"]): entry for entry in expected_entries}
    seen: set[str] = set()
    pending_long_name: str | None = None
    long_name_headers = 0
    process = start_zstd(zstd, asset_path)
    assert process.stdout is not None
    reader = BoundedTarReader(
        process.stdout,
        tar_expansion_limit(expected_entries),
        asset_path.name,
    )
    try:
        while True:
            block = read_exact(
                reader, TAR_BLOCK_SIZE, f"tar header in {asset_path.name}"
            )
            if block == b"\0" * TAR_BLOCK_SIZE:
                if pending_long_name is not None:
                    raise DownloadError(
                        f"asset {asset_path.name} ends after GNU long-name metadata"
                    )
                bytes_before_end = reader.count - TAR_BLOCK_SIZE
                end_bytes = TAR_BLOCK_SIZE * 2 + (
                    -(bytes_before_end + TAR_BLOCK_SIZE * 2) % TAR_RECORD_SIZE
                )
                remaining_end = read_exact(
                    reader,
                    end_bytes - TAR_BLOCK_SIZE,
                    f"tar end marker in {asset_path.name}",
                )
                if any(remaining_end):
                    raise DownloadError(
                        f"asset {asset_path.name} has nonzero tar end padding"
                    )
                if reader.read(1):
                    raise DownloadError(
                        f"asset {asset_path.name} has data after the tar end marker"
                    )
                break

            member = parse_tar_header(block, asset_path.name)
            if member.type == tarfile.GNUTYPE_LONGNAME:
                if pending_long_name is not None or long_name_headers >= len(expected):
                    raise DownloadError(
                        f"asset {asset_path.name} has unexpected GNU long-name metadata"
                    )
                if member.name != "././@LongLink" or member.linkname:
                    raise DownloadError(
                        f"asset {asset_path.name} has malformed GNU long-name metadata"
                    )
                if member.size <= 1 or member.size > MAX_GNU_LONGNAME_BYTES:
                    raise DownloadError(
                        f"GNU long-name metadata is too large in {asset_path.name}"
                    )
                body = read_exact(
                    reader,
                    member.size,
                    f"GNU long-name metadata in {asset_path.name}",
                )
                padding = read_exact(
                    reader,
                    tar_padded_size(member.size) - member.size,
                    f"GNU long-name padding in {asset_path.name}",
                )
                if any(padding):
                    raise DownloadError(
                        f"asset {asset_path.name} has nonzero GNU long-name padding"
                    )
                if body[-1:] != b"\0" or b"\0" in body[:-1]:
                    raise DownloadError(
                        f"asset {asset_path.name} has malformed GNU long-name data"
                    )
                try:
                    long_name = body[:-1].decode("utf-8", "strict")
                except UnicodeDecodeError as exc:
                    raise DownloadError(
                        f"asset {asset_path.name} has a non-UTF-8 GNU long name"
                    ) from exc
                if len(body) - 1 <= 100:
                    raise DownloadError(
                        f"asset {asset_path.name} has unnecessary GNU long-name metadata"
                    )
                pending_long_name = safe_logical_path(
                    long_name,
                    f"GNU long name in {asset_path.name}",
                )
                if pending_long_name not in expected or pending_long_name in seen:
                    raise DownloadError(
                        f"asset {asset_path.name} has an unexpected GNU long name: "
                        f"{pending_long_name}"
                    )
                long_name_headers += 1
                continue

            if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                raise DownloadError(
                    f"asset {asset_path.name} contains an unsupported member type "
                    f"0x{member.type.hex()}"
                )
            if member.linkname:
                raise DownloadError(
                    f"asset {asset_path.name} has link metadata on a regular member"
                )

            if pending_long_name is None:
                try:
                    member.name.encode("utf-8", "strict")
                except UnicodeEncodeError as exc:
                    raise DownloadError(
                        f"asset {asset_path.name} has a non-UTF-8 member name"
                    ) from exc
                logical = safe_logical_path(member.name, f"member in {asset_path.name}")
            else:
                logical = pending_long_name
                pending_long_name = None

            if logical not in expected:
                raise DownloadError(
                    f"asset {asset_path.name} contains an unmanifested member: {logical}"
                )
            if logical in seen:
                raise DownloadError(
                    f"asset {asset_path.name} contains a duplicate member: {logical}"
                )
            entry = expected[logical]
            if member.size != int(entry["exported_bytes"]):
                raise DownloadError(
                    f"tar header size disagrees with manifest for {logical}"
                )

            write_verified_stream(reader, target_path(tree, logical), entry)
            padding = read_exact(
                reader,
                tar_padded_size(member.size) - member.size,
                f"file padding for {logical}",
            )
            if any(padding):
                raise DownloadError(
                    f"asset {asset_path.name} has nonzero padding for {logical}"
                )
            seen.add(logical)

        if seen != set(expected):
            missing = sorted(set(expected) - seen)
            raise DownloadError(
                f"asset {asset_path.name} is missing manifest files: {missing[:5]}"
            )
        finish_zstd(process, asset_path.name)
    except BaseException:
        abort_zstd(process)
        raise
    finally:
        process.stdout.close()


def download_and_extract(args: argparse.Namespace) -> dict[str, object]:
    manifest_path = Path(args.manifest).expanduser().resolve(strict=True)
    manifest = load_manifest(manifest_path)
    if args.assets not in {"all", "text"}:
        raise DownloadError("--assets must be all or text")
    base_url = validate_release_base_url(args.release_base_url)
    requested_output = Path(args.output).expanduser().absolute()
    if requested_output.exists() or requested_output.is_symlink():
        raise DownloadError(f"output path already exists: {requested_output}")
    destination = requested_output.resolve(strict=False)
    if args.timeout <= 0:
        raise DownloadError("--timeout must be positive")
    zstd = shutil.which(args.zstd)
    if not zstd:
        raise DownloadError(f"zstd executable not found: {args.zstd}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".evidence-download-", dir=destination.parent))
    downloads = work / "assets"
    tree = work / "tree"
    downloads.mkdir()
    tree.mkdir()
    opener = urllib.request.build_opener(SafeRedirectHandler())
    files_by_asset: dict[str, list[dict[str, object]]] = collections.defaultdict(list)
    validated_files: list[dict[str, object]] = []
    for raw_entry in manifest["files"]:
        entry = require_dict(raw_entry, "validated file")
        validated_files.append(entry)
        files_by_asset[str(entry["asset"])].append(entry)
    validated_assets = [
        require_dict(raw_asset, "validated asset") for raw_asset in manifest["assets"]
    ]
    selected_assets = [
        asset
        for asset in validated_assets
        if args.assets == "all" or asset["format"] == "tar+zstd"
    ]
    selected_names = {str(asset["name"]) for asset in selected_assets}
    selected_files = [
        entry for entry in validated_files if str(entry["asset"]) in selected_names
    ]

    try:
        for asset in selected_assets:
            name = str(asset["name"])
            local_asset = downloads / name
            download_asset(
                opener,
                asset_url(base_url, name),
                local_asset,
                int(asset["size"]),
                str(asset["sha256"]),
                args.timeout,
            )
            owned = files_by_asset[name]
            if asset["format"] == "tar+zstd":
                extract_tar_asset(zstd, local_asset, tree, owned)
            else:
                extract_numeric_asset(zstd, local_asset, tree, owned[0])
            local_asset.unlink()

        os.replace(tree, destination)
        complete = args.assets == "all"
        return {
            "schema": SCHEMA,
            "asset_selection": args.assets,
            "restore_status": "complete" if complete else "text-only-subset",
            "complete": complete,
            "restored_files": len(selected_files),
            "restored_assets": len(selected_assets),
            "omitted_files": len(validated_files) - len(selected_files),
            "omitted_assets": len(validated_assets) - len(selected_assets),
            "manifest_files": len(validated_files),
            "manifest_assets": len(validated_assets),
            "output": str(destination),
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download, verify, and atomically restore a glm53-orca-evidence.v1 export."
    )
    parser.add_argument("--manifest", required=True, help="local evidence/index.json")
    parser.add_argument(
        "--release-base-url",
        required=True,
        help="credential-free HTTPS or local file URL containing the named release assets",
    )
    parser.add_argument("--output", required=True, help="new destination directory")
    parser.add_argument(
        "--assets",
        choices=("all", "text"),
        default="all",
        help="restore all assets (default) or only non-array tar assets",
    )
    parser.add_argument("--zstd", default="zstd", help="zstd executable name or path")
    parser.add_argument(
        "--timeout", type=float, default=60.0, help="per-request timeout in seconds"
    )
    return parser


def main() -> int:
    try:
        summary = download_and_extract(build_parser().parse_args())
    except (DownloadError, OSError, subprocess.SubprocessError) as exc:
        print(f"download_evidence: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
