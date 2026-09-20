#!/usr/bin/env python3
"""Export an immutable qualification evidence tree into verified release assets."""

from __future__ import annotations

import ast
import argparse
import collections
import dataclasses
import gzip
import hashlib
import io
import ipaddress
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zlib
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterable

SCHEMA = "glm53-orca-evidence.v1"
GITHUB_ASSET_LIMIT = 2 * 1024**3
MAX_RELEASE_ASSETS = 1000
CHUNK = 4 * 1024 * 1024
MAX_IN_MEMORY_TEXT = 512 * 1024 * 1024
REDACTED = "<REDACTED:CREDENTIAL>"

CACHE_DIRS = {
    "__pycache__",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
PRIVATE_FILENAMES = {
    "access-approved.json",
    ".netrc",
    "authorized_keys",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}
PRIVATE_SUFFIXES = {".jks", ".kdbx", ".key", ".p12", ".pfx", ".pem"}
WEIGHT_SUFFIXES = {".ckpt", ".gguf", ".pth", ".pt", ".safetensors"}
BINARY_EVIDENCE_SUFFIXES = {
    ".bmp",
    ".flac",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".mp4",
    ".npz",
    ".otf",
    ".pdf",
    ".png",
    ".tif",
    ".tiff",
    ".ttf",
    ".wav",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
}

EXCLUSION_REASONS = {
    "generated-cache": "Generated cache or bytecode; reproducible from the exported source.",
    "model-weight": "Model weight; weights are distributed separately and are not evidence artifacts.",
    "private-authorization-material": "Private authorization material is never published.",
    "vendor-runtime-binary": "Compiled vendor runtime binary obtainable from the pinned runtime image.",
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

HOME_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:home/[^/\s\"'<>]+|root)(?=/|\b)")
IPV4_RE = re.compile(
    r"(?<![0-9])(?<![0-9]\.)(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9]|\.[0-9])"
)
IPV6_LAN_RE = re.compile(
    r"(?i)(?<![0-9a-f:])(?:f[cd][0-9a-f]{2}|fe[89ab][0-9a-f])"
    r"(?::[0-9a-f]{0,4}){2,7}(?![0-9a-f:])"
)
PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
AUTH_HEADER_RE = re.compile(
    r"(?im)(\b(?:authorization|proxy-authorization)\s*:\s*(?:(?:bearer|basic)\s+)?)"
    r"([^\s,;\"']+)"
)
COOKIE_HEADER_RE = re.compile(r"(?im)(\b(?:cookie|set-cookie)\s*:\s*)([^\r\n]+)")
QUOTED_CREDENTIAL_RE = re.compile(
    r"""(?im)(?P<prefix>
        (?:
            "(?P<key_double>(?:\\.|[^"\\\r\n])*)"
            |'(?P<key_single>(?:\\.|[^'\\\r\n])*)'
            |(?<![?&A-Za-z0-9_-])(?P<bare_key>[A-Za-z][A-Za-z0-9_-]*)
        )
        \s*(?P<separator>[:=])\s*
    )
    (?:
        "(?P<value_double>(?:\\.|[^"\\\r\n])*)"
        |'(?P<value_single>(?:\\.|[^'\\\r\n])*)'
    )
    (?=\s*(?:[,;#}\]\)\r\n]|$))
    """,
    re.VERBOSE,
)
UNQUOTED_CREDENTIAL_RE = re.compile(
    r"""(?im)(?P<prefix>
        (?:
            "(?P<key_double>(?:\\.|[^"\\\r\n])*)"
            |'(?P<key_single>(?:\\.|[^'\\\r\n])*)'
            |(?<![?&A-Za-z0-9_-])(?P<bare_key>[A-Za-z][A-Za-z0-9_-]*)
        )
        \s*(?P<separator>[:=])\s*
    )
    (?P<value>(?!["'])[^\r\n,;&#}\]]+?)
    (?=\s*(?:[,;#}\]\)\r\n]|$))
    """,
    re.VERBOSE,
)
ENV_ASSIGNMENT_RE = re.compile(
    r"(?im)^(?P<prefix>\s*(?:export\s+)?(?P<key>[A-Z][A-Z0-9_]*)\s*=\s*)"
    r"(?P<value>[^\r\n]*)"
)
CLI_CREDENTIAL_RE = re.compile(
    r"(?i)(?P<prefix>--(?:access[-_]token|api[-_]key|auth[-_]token|authorization|"
    r"bearer[-_]token|client[-_]secret|github[-_]token|gitlab[-_]token|hf[-_]token|"
    r"password|private[-_]key|refresh[-_]token|secret[-_]key|token)(?:=|\s+))"
    r"(?P<value>[^\s]+)"
)
URL_USERINFO_RE = re.compile(
    r"(?i)(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<userinfo>[^/@\s?#]+)@"
)
URL_QUERY_PARAMETER_RE = re.compile(
    r"(?P<prefix>[?&])(?P<key>[^=&#\s]+)=(?P<value>[^&#\s]*)"
)
URL_SECRET_KEYS = {
    "access_token",
    "api_key",
    "auth_token",
    "credential",
    "key",
    "password",
    "secret",
    "sig",
    "signature",
    "token",
    "x_amz_credential",
    "x_amz_security_token",
    "x_amz_signature",
    "x_goog_signature",
}
SUSPICIOUS_PATTERNS = {
    "aws-access-key": re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    "github-token": re.compile(
        r"(?<![A-Za-z0-9_])(?:gh[pousr]_[A-Za-z0-9]{20,}|"
        r"github_pat_[A-Za-z0-9_]{20,})(?![A-Za-z0-9_])"
    ),
    "huggingface-token": re.compile(
        r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{20,}(?![A-Za-z0-9_])"
    ),
    "jwt": re.compile(
        r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
    ),
    "openai-style-key": re.compile(
        r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9_-])"
    ),
    "private-key": re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "slack-token": re.compile(
        r"(?<![A-Za-z0-9-])xox[baprs]-[A-Za-z0-9-]{16,}(?![A-Za-z0-9-])"
    ),
}
BINARY_SUSPICIOUS_PATTERNS = {
    "aws-access-key": re.compile(rb"(?:AKIA|ASIA)[A-Z0-9]{16}"),
    "private-key": re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "github-token": re.compile(
        rb"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"
    ),
    "huggingface-token": re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    "jwt": re.compile(rb"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    "openai-style-key": re.compile(rb"sk-[A-Za-z0-9_-]{16,}"),
    "slack-token": re.compile(rb"xox[baprs]-[A-Za-z0-9-]{16,}"),
}
WINDOWS_RESERVED_NAME_RE = re.compile(r"(?i)^(?:con|prn|aux|nul|com[1-9]|lpt[1-9])$")
SOURCE_CODE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".cxx",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".php",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".scala",
    ".swift",
    ".ts",
    ".tsx",
}
SHELL_SOURCE_SUFFIXES = {".bash", ".ksh", ".ps1", ".sh", ".zsh"}
SCHEMA_DECLARATION_MAP_KEYS = {
    "$defs",
    "definitions",
    "dependentSchemas",
    "patternProperties",
    "properties",
}
SCHEMA_VALUE_KEYS = {"const", "default", "enum", "example", "examples"}
MAX_NPY_HEADER = 64 * 1024
CREDENTIAL_KEYS = {
    "api_key",
    "apikey",
    "auth",
    "auth_token",
    "authorization",
    "aws_secret_access_key",
    "bearer_token",
    "client_secret",
    "cookie",
    "credentials",
    "github_token",
    "gitlab_token",
    "hf_token",
    "hugging_face_hub_token",
    "password",
    "passwd",
    "private_key",
    "proxy_authorization",
    "refresh_token",
    "secret_key",
    "set_cookie",
}
SAFE_ENV_KEYS = {
    "MAX_TOKEN",
    "MAX_TOKENS",
    "TOKENIZERS_PARALLELISM",
    "TOKEN_COUNT",
    "TOKEN_ID",
    "TOKEN_IDS",
}
MODEL_PATH_KEYS = {
    "checkpoint_path",
    "model_dir",
    "model_path",
    "model_root",
    "tokenizer_path",
    "weights_path",
}
CACHE_PATH_KEYS = {"cache_dir", "cache_path", "cache_root", "hf_home"}


class ExportError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Fingerprint:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclasses.dataclass(frozen=True)
class SourceFile:
    logical_path: str
    source_path: Path
    fingerprint: Fingerprint
    link_fingerprint: Fingerprint | None
    exclusion_category: str | None

    @property
    def source_bytes(self) -> int:
        return self.fingerprint.size


@dataclasses.dataclass
class ExportedFile:
    logical_path: str
    source_bytes: int
    source_sha256: str
    exported_bytes: int
    exported_sha256: str
    redaction_counts: collections.Counter[str]
    staged_path: Path | None = None
    asset: str | None = None

    @property
    def redaction_classification(self) -> str:
        categories = set(self.redaction_counts)
        machine = bool(categories & MACHINE_CATEGORIES)
        credential = bool(categories & CREDENTIAL_CATEGORIES)
        if machine and credential:
            return "sanitized-and-redacted"
        if credential:
            return "redacted-credentials"
        if machine:
            return "sanitized-machine-identifiers"
        return "none"


@dataclasses.dataclass(frozen=True)
class Asset:
    name: str
    size: int
    sha256: str
    format: str
    logical_extraction_root: str
    file_count: int
    unpacked_bytes: int


def fingerprint(value: os.stat_result) -> Fingerprint:
    return Fingerprint(
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def validate_logical_path(value: str) -> str:
    if (
        not value
        or value.startswith("/")
        or ":" in value
        or "\\" in value
        or any(ord(char) < 32 or 0x7F <= ord(char) <= 0x9F for char in value)
    ):
        raise ExportError("unsafe evidence path")
    parts = value.split("/")
    for part in parts:
        if part in {"", ".", ".."} or part.endswith((".", " ")):
            raise ExportError("unsafe evidence path component")
        if WINDOWS_RESERVED_NAME_RE.fullmatch(part.split(".", 1)[0]):
            raise ExportError("non-portable evidence path component")
    if PurePosixPath(value).as_posix() != value:
        raise ExportError("non-canonical evidence path")
    return value


def exclusion_for(logical: str) -> str | None:
    path = PurePosixPath(logical)
    lower_parts = tuple(part.lower() for part in path.parts)
    name = lower_parts[-1]
    suffixes = {suffix.lower() for suffix in Path(name).suffixes}

    if any(part in CACHE_DIRS for part in lower_parts) or name.endswith(
        (".pyc", ".pyo")
    ):
        return "generated-cache"
    if name == "pytorch_model.bin" or suffixes & WEIGHT_SUFFIXES:
        return "model-weight"
    if (
        name in PRIVATE_FILENAMES
        or name == ".env"
        or name.startswith(".env.")
        or any(name.endswith(suffix) for suffix in PRIVATE_SUFFIXES)
    ):
        return "private-authorization-material"
    if lower_parts[0] == "runtime-source" and (
        ".so" in suffixes or suffixes & {".a", ".cubin", ".dll", ".dylib", ".exe", ".o"}
    ):
        return "vendor-runtime-binary"
    return None


def scan_source(root: Path, redactor: Redactor) -> list[SourceFile]:
    records: list[SourceFile] = []

    def visit(directory: Path, prefix: tuple[str, ...]) -> None:
        try:
            entries = sorted(
                os.scandir(directory), key=lambda entry: entry.name.encode("utf-8")
            )
        except (OSError, UnicodeError) as exc:
            raise ExportError(f"cannot enumerate source directory: {exc}") from exc
        for entry in entries:
            try:
                entry.name.encode("utf-8", "strict")
            except UnicodeError as exc:
                raise ExportError(
                    "source contains a filename that is not valid UTF-8"
                ) from exc
            logical = validate_logical_path("/".join((*prefix, entry.name)))
            redactor.assert_safe_metadata(logical, "evidence path")
            try:
                link_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ExportError(f"cannot stat {logical}: {exc}") from exc

            if stat.S_ISLNK(link_stat.st_mode):
                try:
                    target = Path(entry.path).resolve(strict=True)
                    target_logical = validate_logical_path(
                        target.relative_to(root).as_posix()
                    )
                except (OSError, ValueError) as exc:
                    raise ExportError(
                        f"symlink escapes evidence root or is broken: {logical}"
                    ) from exc
                redactor.assert_safe_metadata(target_logical, "symlink target")
                target_stat = target.stat()
                if stat.S_ISDIR(target_stat.st_mode):
                    raise ExportError(
                        f"directory symlinks are not exportable safely: {logical}"
                    )
                if not stat.S_ISREG(target_stat.st_mode):
                    raise ExportError(
                        f"symlink does not resolve to a regular file: {logical}"
                    )
                alias_exclusion = exclusion_for(logical)
                target_exclusion = exclusion_for(target_logical)
                exclusion = (
                    "private-authorization-material"
                    if "private-authorization-material"
                    in {alias_exclusion, target_exclusion}
                    else alias_exclusion or target_exclusion
                )
                records.append(
                    SourceFile(
                        logical,
                        target,
                        fingerprint(target_stat),
                        fingerprint(link_stat),
                        exclusion,
                    )
                )
            elif stat.S_ISDIR(link_stat.st_mode):
                visit(Path(entry.path), (*prefix, entry.name))
            elif stat.S_ISREG(link_stat.st_mode):
                records.append(
                    SourceFile(
                        logical,
                        Path(entry.path),
                        fingerprint(link_stat),
                        None,
                        exclusion_for(logical),
                    )
                )
            else:
                raise ExportError(f"special files are not exportable: {logical}")

    visit(root, ())
    for index, record in enumerate(records):
        if record.exclusion_category is None and record.logical_path.startswith(
            "runtime-source/"
        ):
            with open_stable(record) as source:
                is_elf = source.read(4) == b"\x7fELF"
                verify_stable_handle(record, source)
            if is_elf:
                records[index] = dataclasses.replace(
                    record, exclusion_category="vendor-runtime-binary"
                )
    private_identities = {
        (record.fingerprint.device, record.fingerprint.inode)
        for record in records
        if record.exclusion_category == "private-authorization-material"
    }
    return [
        dataclasses.replace(record, exclusion_category="private-authorization-material")
        if (record.fingerprint.device, record.fingerprint.inode) in private_identities
        else record
        for record in records
    ]


def normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def is_credential_key(value: str) -> bool:
    key = normalized_key(value)
    if key in CREDENTIAL_KEYS:
        return True
    return bool(
        re.search(
            r"(?:^|_)(?:(?:access|anthropic|api|auth|aws|azure|github|gitlab|hf|"
            r"huggingface|openai|refresh)_(?:key|password|secret|token)|"
            r"password|passwd|private_key|secret|secret_key)$",
            key,
        )
    )


def is_credential_env_key(value: str) -> bool:
    key = value.upper()
    if key in SAFE_ENV_KEYS:
        return False
    return is_credential_key(key) or bool(
        re.search(
            r"(?:^|_)(?:API_KEY|AUTH_TOKEN|PASSWORD|PRIVATE_KEY|SECRET|TOKEN)$", key
        )
    )


class JsonNumber(str):
    """A validated JSON number retained as its original source lexeme."""


def decode_quoted_literal(raw: str, quote: str) -> str:
    literal = quote + raw + quote
    if quote == '"':
        try:
            decoded = json.loads(literal)
            if isinstance(decoded, str):
                return decoded
        except json.JSONDecodeError:
            pass
    try:
        decoded = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        return raw
    return decoded if isinstance(decoded, str) else raw


def is_nonliteral_expression(value: str, logical: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    if re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", candidate):
        return False
    suffix = Path(logical).suffix.lower()
    if suffix in SHELL_SOURCE_SUFFIXES:
        return candidate.startswith(("$", "`")) or "$(" in candidate
    return suffix in SOURCE_CODE_SUFFIXES


def is_json_schema_document(value: object, logical: str) -> bool:
    name = PurePosixPath(logical).name.lower()
    return isinstance(value, dict) and (
        "$schema" in value or name.endswith("schema.json")
    )


class Redactor:
    def __init__(
        self,
        source_root: Path,
        model_roots: Iterable[Path],
        cache_roots: Iterable[Path],
        approved_canaries: Iterable[str] = (),
    ) -> None:
        replacements: list[tuple[str, str, str]] = [
            (str(source_root), "${EVIDENCE_SOURCE_ROOT}", "source_root_path")
        ]
        replacements.extend(
            (str(path), "${ORCA_MODEL_ROOT}", "model_path") for path in model_roots
        )
        replacements.extend(
            (str(path), "${ORCA_CACHE_ROOT}", "cache_path") for path in cache_roots
        )
        self.replacements = sorted(
            replacements, key=lambda item: len(item[0]), reverse=True
        )
        self.approved_canaries = frozenset(approved_canaries)
        self.approved_binary_canaries = frozenset(
            value.encode("utf-8") for value in self.approved_canaries
        )

    @staticmethod
    def _count(
        counter: collections.Counter[str], category: str, amount: int = 1
    ) -> None:
        if amount:
            counter[category] += amount

    def _is_approved(self, *values: str) -> bool:
        return any(value in self.approved_canaries for value in values)

    def _preserve_approved(
        self,
        counter: collections.Counter[str],
        *values: str,
    ) -> bool:
        if not self._is_approved(*values):
            return False
        self._count(counter, "synthetic_canary_preserved")
        return True

    @staticmethod
    def _assignment_key(match: re.Match[str]) -> str:
        bare = match.group("bare_key")
        if bare is not None:
            return bare
        raw = match.group("key_double")
        if raw is not None:
            return decode_quoted_literal(raw, '"')
        raw = match.group("key_single")
        assert raw is not None
        return decode_quoted_literal(raw, "'")

    @staticmethod
    def _quoted_value(match: re.Match[str]) -> tuple[str, str, str]:
        raw = match.group("value_double")
        if raw is not None:
            return raw, decode_quoted_literal(raw, '"'), '"'
        raw = match.group("value_single")
        assert raw is not None
        return raw, decode_quoted_literal(raw, "'"), "'"

    def sanitize_text(
        self,
        text: str,
        counter: collections.Counter[str],
        logical: str,
    ) -> tuple[str, bool]:
        original = text

        for needle, replacement, category in self.replacements:
            occurrences = text.count(needle)
            if occurrences:
                text = text.replace(needle, replacement)
                self._count(counter, category, occurrences)

        def home_replace(_match: re.Match[str]) -> str:
            self._count(counter, "home_path")
            return "${HOME}"

        text = HOME_RE.sub(home_replace, text)

        def ipv4_replace(match: re.Match[str]) -> str:
            candidate = match.group(0)
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                return candidate
            if any(
                address in network
                for network in (
                    ipaddress.ip_network("10.0.0.0/8"),
                    ipaddress.ip_network("172.16.0.0/12"),
                    ipaddress.ip_network("192.168.0.0/16"),
                )
            ):
                self._count(counter, "lan_address")
                return "<LAN_IPV4>"
            return candidate

        text = IPV4_RE.sub(ipv4_replace, text)

        def ipv6_replace(_match: re.Match[str]) -> str:
            self._count(counter, "lan_address")
            return "<LAN_IPV6>"

        text = IPV6_LAN_RE.sub(ipv6_replace, text)

        def private_key_replace(match: re.Match[str]) -> str:
            value = match.group(0)
            if self._preserve_approved(counter, value):
                return value
            self._count(counter, "private_key_block")
            return "<REDACTED:PRIVATE_KEY>"

        text = PRIVATE_KEY_BLOCK_RE.sub(private_key_replace, text)

        def auth_replace(match: re.Match[str]) -> str:
            value = match.group(2)
            if self._preserve_approved(counter, value):
                return match.group(0)
            self._count(counter, "authorization_header")
            return match.group(1) + REDACTED

        text = AUTH_HEADER_RE.sub(auth_replace, text)

        def cookie_replace(match: re.Match[str]) -> str:
            value = match.group(2)
            if self._preserve_approved(counter, value):
                return match.group(0)
            self._count(counter, "authorization_header")
            return match.group(1) + REDACTED

        text = COOKIE_HEADER_RE.sub(cookie_replace, text)

        def quoted_replace(match: re.Match[str]) -> str:
            if not is_credential_key(self._assignment_key(match)):
                return match.group(0)
            raw, decoded, quote = self._quoted_value(match)
            if decoded in {"", REDACTED} or self._preserve_approved(
                counter, raw, decoded
            ):
                return match.group(0)
            self._count(counter, "credential_assignment")
            return match.group("prefix") + quote + REDACTED + quote

        text = QUOTED_CREDENTIAL_RE.sub(quoted_replace, text)

        def unquoted_replace(match: re.Match[str]) -> str:
            if not is_credential_key(self._assignment_key(match)):
                return match.group(0)
            value = match.group("value").rstrip()
            if value in {"", REDACTED} or self._preserve_approved(counter, value):
                return match.group(0)
            if is_nonliteral_expression(value, logical):
                return match.group(0)
            self._count(counter, "credential_assignment")
            replacement = (
                json.dumps(REDACTED)
                if (
                    match.group("separator") == "="
                    or Path(logical).suffix.lower() in SOURCE_CODE_SUFFIXES
                )
                else REDACTED
            )
            return match.group("prefix") + replacement

        text = UNQUOTED_CREDENTIAL_RE.sub(unquoted_replace, text)

        def env_replace(match: re.Match[str]) -> str:
            if not is_credential_env_key(match.group("key")):
                return match.group(0)
            value = match.group("value")
            candidates = [value]
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                candidates.append(decode_quoted_literal(value[1:-1], value[0]))
            if any(
                candidate in {"", REDACTED} for candidate in candidates
            ) or self._preserve_approved(counter, *candidates):
                return match.group(0)
            if is_nonliteral_expression(value, logical):
                return match.group(0)
            self._count(counter, "credential_assignment")
            return match.group("prefix") + json.dumps(REDACTED)

        text = ENV_ASSIGNMENT_RE.sub(env_replace, text)

        def cli_replace(match: re.Match[str]) -> str:
            value = match.group("value")
            candidates = [value]
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                candidates.append(decode_quoted_literal(value[1:-1], value[0]))
            if self._preserve_approved(counter, *candidates):
                return match.group(0)
            self._count(counter, "credential_assignment")
            return match.group("prefix") + REDACTED

        text = CLI_CREDENTIAL_RE.sub(cli_replace, text)

        def userinfo_replace(match: re.Match[str]) -> str:
            value = match.group("userinfo")
            decoded = urllib.parse.unquote(value)
            if self._preserve_approved(counter, value, decoded):
                return match.group(0)
            self._count(counter, "url_credential")
            return match.group("scheme") + REDACTED + "@"

        text = URL_USERINFO_RE.sub(userinfo_replace, text)

        def query_replace(match: re.Match[str]) -> str:
            key = normalized_key(urllib.parse.unquote_plus(match.group("key")))
            value = match.group("value")
            decoded = urllib.parse.unquote_plus(value)
            if key not in URL_SECRET_KEYS and not is_credential_key(key):
                self.assert_no_unexplained_secrets(decoded, logical)
                return match.group(0)
            if self._preserve_approved(counter, value, decoded):
                return match.group(0)
            self._count(counter, "url_credential")
            return match.group("prefix") + match.group("key") + "=" + REDACTED

        text = URL_QUERY_PARAMETER_RE.sub(query_replace, text)
        self.assert_no_unexplained_secrets(text, logical)
        return text, text != original

    def assert_no_unexplained_secrets(self, text: str, logical: str) -> None:
        for category, pattern in SUSPICIOUS_PATTERNS.items():
            for match in pattern.finditer(text):
                if match.group(0) in self.approved_canaries:
                    continue
                line = text.count("\n", 0, match.start()) + 1
                raise ExportError(
                    f"{logical}:{line}: unexplained {category} pattern; "
                    "extend the explicit redaction policy before exporting"
                )
        home_match = HOME_RE.search(text)
        if home_match:
            line = text.count("\n", 0, home_match.start()) + 1
            raise ExportError(f"{logical}:{line}: home path survived sanitization")
        ipv6_match = IPV6_LAN_RE.search(text)
        if ipv6_match:
            line = text.count("\n", 0, ipv6_match.start()) + 1
            raise ExportError(f"{logical}:{line}: LAN address survived sanitization")
        for match in IPV4_RE.finditer(text):
            try:
                address = ipaddress.ip_address(match.group(0))
            except ValueError:
                continue
            if any(
                address in network
                for network in (
                    ipaddress.ip_network("10.0.0.0/8"),
                    ipaddress.ip_network("172.16.0.0/12"),
                    ipaddress.ip_network("192.168.0.0/16"),
                )
            ):
                line = text.count("\n", 0, match.start()) + 1
                raise ExportError(
                    f"{logical}:{line}: LAN address survived sanitization"
                )

    def assert_safe_metadata(self, value: str, kind: str) -> None:
        try:
            sanitized, changed = self.sanitize_text(
                value, collections.Counter(), "public metadata"
            )
        except ExportError:
            raise ExportError(f"unsafe private value in {kind}") from None
        if changed or sanitized != value:
            raise ExportError(f"unsafe private value in {kind}")

    def scan_binary(self, chunk: bytes, tail: bytes, logical: str) -> bytes:
        combined = tail + chunk
        for category, pattern in BINARY_SUSPICIOUS_PATTERNS.items():
            for match in pattern.finditer(combined):
                if match.group(0) in self.approved_binary_canaries:
                    continue
                raise ExportError(
                    f"{logical}: unexplained {category} pattern in binary evidence"
                )
        return combined[-256:]

    def _redact_credential_value(
        self,
        value: object,
        counter: collections.Counter[str],
    ) -> tuple[object, bool]:
        if isinstance(value, JsonNumber):
            self._count(counter, "credential_field")
            return REDACTED, True
        if isinstance(value, str):
            if value in {"", REDACTED}:
                return value, False
            if self._preserve_approved(counter, value):
                return value, False
            self._count(counter, "credential_field")
            return REDACTED, True
        if isinstance(value, list):
            changed = False
            output: list[object] = []
            for item in value:
                updated, item_changed = self._redact_credential_value(item, counter)
                output.append(updated)
                changed |= item_changed
            return output, changed
        if isinstance(value, dict):
            changed = False
            output: dict[str, object] = {}
            for key, item in value.items():
                updated, item_changed = self._redact_credential_value(item, counter)
                output[str(key)] = updated
                changed |= item_changed
            return output, changed
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            self._count(counter, "credential_field")
            return REDACTED, True
        return value, False

    def sanitize_json(
        self,
        value: object,
        counter: collections.Counter[str],
        logical: str,
        key_context: str | None = None,
    ) -> tuple[object, bool]:
        if isinstance(value, dict):
            output: dict[str, object] = {}
            changed = False
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ExportError(
                        f"{logical}: JSON object contains a non-string key"
                    )
                new_key, key_changed = self.sanitize_text(key, counter, logical)
                if new_key in output:
                    raise ExportError(
                        f"{logical}: sanitization causes a duplicate JSON key"
                    )
                if is_credential_key(key) or (
                    normalized_key(key_context or "")
                    in {"env", "environ", "environment"}
                    and is_credential_env_key(key)
                ):
                    updated, item_changed = self._redact_credential_value(item, counter)
                else:
                    updated, item_changed = self.sanitize_json(
                        item, counter, logical, key
                    )
                output[new_key] = updated
                changed |= key_changed or item_changed
            return output, changed
        if isinstance(value, list):
            output_list: list[object] = []
            changed = False
            environment_list = normalized_key(key_context or "") in {
                "env",
                "environ",
                "environment",
            }
            redact_next = False
            credential_options = {
                "access_token",
                "api_key",
                "auth_token",
                "authorization",
                "client_secret",
                "bearer_token",
                "github_token",
                "gitlab_token",
                "hf_token",
                "password",
                "private_key",
                "secret_key",
                "refresh_token",
                "token",
            }
            for item in value:
                if environment_list and isinstance(item, str) and "=" in item:
                    env_key, env_value = item.split("=", 1)
                    if is_credential_env_key(env_key):
                        redacted, item_changed = self._redact_credential_value(
                            env_value, counter
                        )
                        updated = env_key + "=" + str(redacted)
                    else:
                        updated, item_changed = self.sanitize_json(
                            item, counter, logical, key_context
                        )
                elif redact_next:
                    updated, item_changed = self._redact_credential_value(item, counter)
                else:
                    updated, item_changed = self.sanitize_json(
                        item, counter, logical, key_context
                    )
                output_list.append(updated)
                changed |= item_changed
                redact_next = (
                    isinstance(item, str)
                    and item.startswith("-")
                    and normalized_key(item.lstrip("-")) in credential_options
                    and "=" not in item
                )
            return output_list, changed
        if isinstance(value, JsonNumber):
            return value, False
        if isinstance(value, str):
            key = normalized_key(key_context or "")
            if value.startswith("/") and key in MODEL_PATH_KEYS:
                tail = PurePosixPath(value).name
                self._count(counter, "model_path")
                return "${ORCA_MODEL_ROOT}/" + tail, True
            if value.startswith("/") and key in CACHE_PATH_KEYS:
                tail = PurePosixPath(value).name
                self._count(counter, "cache_path")
                return "${ORCA_CACHE_ROOT}/" + tail, True
            return self.sanitize_text(value, counter, logical)
        return value, False

    def sanitize_json_schema(
        self,
        value: object,
        counter: collections.Counter[str],
        logical: str,
    ) -> tuple[object, bool]:
        return self._sanitize_schema_node(value, counter, logical, False)

    def _sanitize_schema_node(
        self,
        value: object,
        counter: collections.Counter[str],
        logical: str,
        credential_context: bool,
    ) -> tuple[object, bool]:
        if isinstance(value, dict):
            output: dict[str, object] = {}
            changed = False
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ExportError(
                        f"{logical}: JSON object contains a non-string key"
                    )
                self.assert_safe_metadata(key, "JSON Schema declaration name")
                if key in SCHEMA_DECLARATION_MAP_KEYS and isinstance(item, dict):
                    declarations: dict[str, object] = {}
                    item_changed = False
                    for declaration_name, declaration in item.items():
                        if not isinstance(declaration_name, str):
                            raise ExportError(
                                f"{logical}: JSON object contains a non-string key"
                            )
                        self.assert_safe_metadata(
                            declaration_name, "JSON Schema declaration name"
                        )
                        declaration_context = key in {
                            "patternProperties",
                            "properties",
                        } and is_credential_key(declaration_name)
                        updated, declaration_changed = self._sanitize_schema_node(
                            declaration,
                            counter,
                            logical,
                            declaration_context,
                        )
                        declarations[declaration_name] = updated
                        item_changed |= declaration_changed
                    updated_item: object = declarations
                elif key in SCHEMA_VALUE_KEYS:
                    if credential_context:
                        updated_item, item_changed = self._redact_credential_value(
                            item, counter
                        )
                    else:
                        updated_item, item_changed = self.sanitize_json(
                            item, counter, logical
                        )
                else:
                    updated_item, item_changed = self._sanitize_schema_node(
                        item,
                        counter,
                        logical,
                        credential_context,
                    )
                output[key] = updated_item
                changed |= item_changed
            return output, changed
        if isinstance(value, list):
            output_list: list[object] = []
            changed = False
            for item in value:
                updated, item_changed = self._sanitize_schema_node(
                    item,
                    counter,
                    logical,
                    credential_context,
                )
                output_list.append(updated)
                changed |= item_changed
            return output_list, changed
        if isinstance(value, JsonNumber):
            return value, False
        if isinstance(value, str):
            return self.sanitize_text(value, counter, logical)
        return value, False


def open_stable(record: SourceFile) -> BinaryIO:
    handle = record.source_path.open("rb")
    actual = fingerprint(os.fstat(handle.fileno()))
    if actual != record.fingerprint:
        handle.close()
        raise ExportError(f"source changed during export: {record.logical_path}")
    return handle


def verify_stable_handle(record: SourceFile, handle: BinaryIO) -> None:
    if fingerprint(os.fstat(handle.fileno())) != record.fingerprint:
        raise ExportError(f"source changed while being read: {record.logical_path}")


def read_stable(record: SourceFile) -> tuple[bytes, str]:
    if record.source_bytes > MAX_IN_MEMORY_TEXT:
        raise ExportError(
            f"text or compressed-text file exceeds the {MAX_IN_MEMORY_TEXT}-byte in-memory safety bound: "
            f"{record.logical_path}"
        )
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    with open_stable(record) as source:
        while chunk := source.read(CHUNK):
            digest.update(chunk)
            chunks.append(chunk)
        verify_stable_handle(record, source)
    return b"".join(chunks), digest.hexdigest()


def decode_utf8(data: bytes, logical: str) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ExportError(f"expected UTF-8 text in {logical}") from exc


def parse_json_without_duplicates(text: str, logical: str) -> object:
    def object_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in pairs:
            if key in output:
                raise ExportError(f"{logical}: duplicate JSON object key")
            output[key] = value
        return output

    def reject_constant(value: str) -> object:
        raise ExportError(f"{logical}: non-standard JSON numeric constant {value}")

    return json.loads(
        text,
        object_pairs_hook=object_hook,
        parse_float=JsonNumber,
        parse_int=JsonNumber,
        parse_constant=reject_constant,
    )


def encode_json_preserving_numbers(
    value: object,
    source_text: str,
    *,
    indent: int | None = None,
    sort_keys: bool = True,
) -> str:
    strings: list[str] = []

    def collect_strings(item: object) -> None:
        if isinstance(item, JsonNumber):
            return
        if isinstance(item, str):
            strings.append(item)
        elif isinstance(item, list):
            for child in item:
                collect_strings(child)
        elif isinstance(item, dict):
            for key, child in item.items():
                strings.append(key)
                collect_strings(child)

    collect_strings(value)
    seed = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    marker_prefix = f"__glm53_json_number_{seed}_"
    while any(marker_prefix in item for item in strings):
        marker_prefix = "_" + marker_prefix
    numbers: list[str] = []

    def prepare(item: object) -> object:
        if isinstance(item, JsonNumber):
            marker = f"{marker_prefix}{len(numbers)}__"
            numbers.append(str(item))
            return marker
        if isinstance(item, list):
            return [prepare(child) for child in item]
        if isinstance(item, dict):
            return {key: prepare(child) for key, child in item.items()}
        return item

    encoded = json.dumps(
        prepare(value),
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        sort_keys=sort_keys,
    )
    marker_re = re.compile(rf'"{re.escape(marker_prefix)}(?P<index>[0-9]+)__"')
    restored = 0

    def restore(match: re.Match[str]) -> str:
        nonlocal restored
        index = int(match.group("index"))
        if index >= len(numbers):
            raise ExportError("internal error while preserving JSON number text")
        restored += 1
        return numbers[index]

    encoded = marker_re.sub(restore, encoded)
    if restored != len(numbers):
        raise ExportError("internal error while preserving JSON number text")
    return encoded


def transform_json_bytes(
    data: bytes,
    logical: str,
    redactor: Redactor,
    counter: collections.Counter[str],
) -> bytes:
    text = decode_utf8(data, logical)
    try:
        parsed = parse_json_without_duplicates(text, logical)
    except json.JSONDecodeError as exc:
        raise ExportError(
            f"invalid JSON evidence in {logical}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    if is_json_schema_document(parsed, logical):
        updated, changed = redactor.sanitize_json_schema(parsed, counter, logical)
    else:
        updated, changed = redactor.sanitize_json(parsed, counter, logical)
    if changed:
        exported_text = (
            encode_json_preserving_numbers(updated, text, indent=2, sort_keys=True)
            + "\n"
        )
        exported = exported_text.encode("utf-8")
    else:
        exported_text = text
        exported = data
    redactor.assert_no_unexplained_secrets(exported_text, logical)
    return exported


def transform_jsonl_bytes(
    data: bytes,
    logical: str,
    redactor: Redactor,
    counter: collections.Counter[str],
) -> bytes:
    text = decode_utf8(data, logical)
    output: list[str] = []
    changed = False
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), 1):
        line = raw_line.rstrip("\r\n")
        ending = raw_line[len(line) :]
        if not line.strip():
            output.append(raw_line)
            continue
        try:
            parsed = parse_json_without_duplicates(line, f"{logical}:{line_number}")
        except json.JSONDecodeError as exc:
            raise ExportError(
                f"invalid JSONL evidence in {logical}:{line_number}: {exc.msg}"
            ) from exc
        updated, line_changed = redactor.sanitize_json(parsed, counter, logical)
        if line_changed:
            output.append(
                encode_json_preserving_numbers(
                    updated,
                    line,
                    sort_keys=True,
                )
                + ending
            )
        else:
            output.append(raw_line)
        changed |= line_changed
    exported_text = "".join(output)
    redactor.assert_no_unexplained_secrets(exported_text, logical)
    return exported_text.encode("utf-8") if changed else data


def transform_text_bytes(
    data: bytes,
    logical: str,
    redactor: Redactor,
    counter: collections.Counter[str],
) -> bytes:
    text = decode_utf8(data, logical)
    updated, changed = redactor.sanitize_text(text, counter, logical)
    return updated.encode("utf-8") if changed else data


def transform_content(
    data: bytes,
    logical: str,
    redactor: Redactor,
    counter: collections.Counter[str],
) -> bytes:
    # Empty HTTP request bodies are evidence, not malformed JSON documents.
    if not data:
        return data
    lower = logical.lower()
    if lower.endswith(".json"):
        return transform_json_bytes(data, logical, redactor, counter)
    if lower.endswith((".jsonl", ".ndjson")):
        return transform_jsonl_bytes(data, logical, redactor, counter)
    return transform_text_bytes(data, logical, redactor, counter)


def gzip_deterministic(data: bytes) -> bytes:
    compressed = bytearray(gzip.compress(data, compresslevel=9, mtime=0))
    compressed[9] = 255
    return bytes(compressed)


def gzip_decompress_bounded(data: bytes, logical: str) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as compressed:
        unpacked = compressed.read(MAX_IN_MEMORY_TEXT + 1)
    if len(unpacked) > MAX_IN_MEMORY_TEXT:
        raise ExportError(
            f"gzip evidence exceeds the {MAX_IN_MEMORY_TEXT}-byte decompressed safety bound: "
            f"{logical}"
        )
    return unpacked


def hash_file(path: Path) -> tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def copy_binary(
    record: SourceFile,
    destination: Path,
    redactor: Redactor,
) -> tuple[str, int, str]:
    source_digest = hashlib.sha256()
    exported_digest = hashlib.sha256()
    count = 0
    tail = b""
    with open_stable(record) as source, destination.open("xb") as output:
        while chunk := source.read(CHUNK):
            tail = redactor.scan_binary(chunk, tail, record.logical_path)
            source_digest.update(chunk)
            exported_digest.update(chunk)
            output.write(chunk)
            count += len(chunk)
        verify_stable_handle(record, source)
    if count != record.source_bytes:
        raise ExportError(f"source size changed during export: {record.logical_path}")
    return source_digest.hexdigest(), count, exported_digest.hexdigest()


def stage_regular(
    record: SourceFile, staging_root: Path, redactor: Redactor
) -> ExportedFile:
    destination = staging_root.joinpath(*record.logical_path.split("/"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = Path(record.logical_path).suffix.lower()
    counter: collections.Counter[str] = collections.Counter()

    if suffix in BINARY_EVIDENCE_SUFFIXES:
        source_sha, exported_bytes, exported_sha = copy_binary(
            record, destination, redactor
        )
    else:
        raw, source_sha = read_stable(record)
        if record.logical_path.lower().endswith(".gz"):
            try:
                unpacked = gzip_decompress_bounded(raw, record.logical_path)
            except (OSError, EOFError, zlib.error) as exc:
                raise ExportError(
                    f"invalid gzip evidence: {record.logical_path}"
                ) from exc
            inner_logical = record.logical_path[:-3]
            transformed = transform_content(unpacked, inner_logical, redactor, counter)
            exported = gzip_deterministic(transformed)
            counter["gzip_metadata"] += 1
        elif b"\x00" not in raw:
            try:
                raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise ExportError(
                    f"unrecognized binary evidence requires an explicit format policy: {record.logical_path}"
                )
            exported = transform_content(raw, record.logical_path, redactor, counter)
        else:
            raise ExportError(
                f"unrecognized binary evidence requires an explicit format policy: {record.logical_path}"
            )
        destination.write_bytes(exported)
        exported_bytes = len(exported)
        exported_sha = hashlib.sha256(exported).hexdigest()

    return ExportedFile(
        logical_path=record.logical_path,
        source_bytes=record.source_bytes,
        source_sha256=source_sha,
        exported_bytes=exported_bytes,
        exported_sha256=exported_sha,
        redaction_counts=counter,
        staged_path=destination,
    )


def zstd_compress(
    zstd: str,
    output: Path,
    threads: int,
    level: int,
    producer: Callable[[BinaryIO], None],
) -> None:
    with output.open("xb") as compressed:
        process = subprocess.Popen(
            [zstd, "-q", f"-T{threads}", f"-{level}", "-c"],
            stdin=subprocess.PIPE,
            stdout=compressed,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        assert process.stderr is not None
        try:
            producer(process.stdin)
            if not process.stdin.closed:
                process.stdin.close()
            stderr = process.stderr.read()
            returncode = process.wait()
        except BaseException:
            try:
                if not process.stdin.closed:
                    process.stdin.close()
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            raise
        finally:
            process.stderr.close()
    if returncode:
        message = stderr.decode("utf-8", "replace").strip()
        output.unlink(missing_ok=True)
        raise ExportError(f"zstd failed ({returncode}): {message}")


def tar_serialized_size(file_size: int) -> int:
    return 512 + ((file_size + 511) // 512) * 512


def shard_files(
    files: list[ExportedFile], target_bytes: int
) -> list[list[ExportedFile]]:
    shards: list[list[ExportedFile]] = []
    current: list[ExportedFile] = []
    current_bytes = 10_240
    for item in sorted(files, key=lambda value: value.logical_path):
        contribution = tar_serialized_size(item.exported_bytes)
        if current and current_bytes + contribution > target_bytes:
            shards.append(current)
            current = []
            current_bytes = 10_240
        current.append(item)
        current_bytes += contribution
    if current:
        shards.append(current)
    return shards


def create_tar_asset(
    files: list[ExportedFile],
    asset_path: Path,
    zstd: str,
    threads: int,
    level: int,
) -> None:
    def produce(stream: BinaryIO) -> None:
        with tarfile.open(
            fileobj=stream, mode="w|", format=tarfile.GNU_FORMAT
        ) as archive:
            for item in files:
                if item.staged_path is None:
                    raise ExportError(
                        f"internal error: no staged path for {item.logical_path}"
                    )
                info = tarfile.TarInfo(item.logical_path)
                info.size = item.exported_bytes
                info.mode = 0o644
                info.mtime = 0
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                with item.staged_path.open("rb") as source:
                    archive.addfile(info, source)

    zstd_compress(zstd, asset_path, threads, level, produce)


def array_asset_name(index: int, logical: str) -> str:
    basename = re.sub(r"[^A-Za-z0-9._-]+", "-", PurePosixPath(logical).name)[:96]
    path_id = hashlib.sha256(logical.encode("utf-8")).hexdigest()[:12]
    return f"numeric-{index:03d}-{path_id}-{basename}.zst"


def validate_npy_stream(record: SourceFile, source: BinaryIO) -> None:
    prefix = source.read(8)
    if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
        raise ExportError(f"{record.logical_path}: invalid NumPy array magic")
    version = (prefix[6], prefix[7])
    if version == (1, 0):
        length_size = 2
    elif version in {(2, 0), (3, 0)}:
        length_size = 4
    else:
        raise ExportError(f"{record.logical_path}: unsupported NumPy array version")
    length_bytes = source.read(length_size)
    if len(length_bytes) != length_size:
        raise ExportError(f"{record.logical_path}: truncated NumPy array header")
    header_length = int.from_bytes(length_bytes, "little")
    if not 0 < header_length <= MAX_NPY_HEADER:
        raise ExportError(f"{record.logical_path}: unsafe NumPy array header size")
    if source.tell() + header_length > record.source_bytes:
        raise ExportError(f"{record.logical_path}: truncated NumPy array header")
    header_bytes = source.read(header_length)
    if (
        len(header_bytes) != header_length
        or not header_bytes.endswith(b"\n")
        or b"#" in header_bytes
    ):
        raise ExportError(f"{record.logical_path}: malformed NumPy array header")
    try:
        header_text = header_bytes.decode("utf-8" if version == (3, 0) else "latin-1")
        header = ast.literal_eval(header_text)
    except (
        RecursionError,
        SyntaxError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise ExportError(
            f"{record.logical_path}: malformed NumPy array header"
        ) from exc
    if type(header) is not dict or set(header) != {"descr", "fortran_order", "shape"}:
        raise ExportError(f"{record.logical_path}: unsupported NumPy array header")
    descriptor = header["descr"]
    match = (
        re.fullmatch(
            r"(?P<byteorder>[<>=|])?(?P<kind>[biufc])(?P<size>[0-9]+)", descriptor
        )
        if isinstance(descriptor, str)
        else None
    )
    if match is None:
        raise ExportError(f"{record.logical_path}: NumPy array dtype must be numeric")
    size_text = match.group("size")
    if len(size_text) > 2:
        raise ExportError(f"{record.logical_path}: unsupported NumPy numeric dtype")
    item_size = int(size_text)
    valid_sizes = {
        "b": {1},
        "i": {1, 2, 4, 8},
        "u": {1, 2, 4, 8},
        "f": {2, 4, 8, 16},
        "c": {8, 16, 32},
    }
    if item_size not in valid_sizes[match.group("kind")] or (
        match.group("byteorder") == "|" and item_size != 1
    ):
        raise ExportError(f"{record.logical_path}: unsupported NumPy numeric dtype")
    if type(header["fortran_order"]) is not bool:
        raise ExportError(f"{record.logical_path}: invalid NumPy array order")
    shape = header["shape"]
    if not isinstance(shape, tuple) or any(
        type(dimension) is not int or dimension < 0 for dimension in shape
    ):
        raise ExportError(f"{record.logical_path}: invalid NumPy array shape")
    payload_bytes = record.source_bytes - source.tell()
    max_elements = payload_bytes // item_size
    elements = 1
    for dimension in shape:
        if dimension and elements > max_elements // dimension:
            raise ExportError(
                f"{record.logical_path}: NumPy array size does not match its header"
            )
        elements *= dimension
    if elements * item_size != payload_bytes:
        raise ExportError(
            f"{record.logical_path}: NumPy array size does not match its header"
        )
    source.seek(0)


def create_array_asset(
    record: SourceFile,
    asset_path: Path,
    zstd: str,
    threads: int,
    level: int,
    redactor: Redactor,
) -> ExportedFile:
    digest = hashlib.sha256()
    count = 0

    def produce(stream: BinaryIO) -> None:
        nonlocal count
        tail = b""
        with open_stable(record) as source:
            validate_npy_stream(record, source)
            while chunk := source.read(CHUNK):
                tail = redactor.scan_binary(chunk, tail, record.logical_path)
                digest.update(chunk)
                stream.write(chunk)
                count += len(chunk)
            verify_stable_handle(record, source)

    zstd_compress(zstd, asset_path, threads, level, produce)
    if count != record.source_bytes:
        raise ExportError(f"source size changed during export: {record.logical_path}")
    source_sha = digest.hexdigest()
    return ExportedFile(
        logical_path=record.logical_path,
        source_bytes=count,
        source_sha256=source_sha,
        exported_bytes=count,
        exported_sha256=source_sha,
        redaction_counts=collections.Counter(),
    )


def checked_asset(
    path: Path, format_name: str, files: list[ExportedFile], limit: int
) -> Asset:
    size, digest = hash_file(path)
    if size >= limit:
        raise ExportError(
            f"asset {path.name} is {size} bytes; every asset must be smaller than {limit} bytes"
        )
    return Asset(
        name=path.name,
        size=size,
        sha256=digest,
        format=format_name,
        logical_extraction_root=".",
        file_count=len(files),
        unpacked_bytes=sum(item.exported_bytes for item in files),
    )


def path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent)
        return True
    except ValueError:
        return False


def manifest_document(
    source_root: Path,
    records: list[SourceFile],
    exported: list[ExportedFile],
    assets: list[Asset],
) -> dict[str, object]:
    excluded_records = [record for record in records if record.exclusion_category]
    exported_sorted = sorted(exported, key=lambda item: item.logical_path)
    total_ledger: collections.Counter[str] = collections.Counter()
    ledger_files: list[dict[str, object]] = []
    for item in exported_sorted:
        if item.redaction_counts:
            counts = dict(sorted(item.redaction_counts.items()))
            total_ledger.update(item.redaction_counts)
            ledger_files.append(
                {"logical_path": item.logical_path, "category_counts": counts}
            )

    return {
        "schema": SCHEMA,
        "source_root_label": source_root.name,
        "coverage": {
            "discovered_files": len(records),
            "exported_files": len(exported_sorted),
            "excluded_files": len(excluded_records),
            "discovered_source_bytes": sum(record.source_bytes for record in records),
            "exported_source_bytes": sum(item.source_bytes for item in exported_sorted),
            "excluded_source_bytes": sum(
                record.source_bytes for record in excluded_records
            ),
            "exported_bytes": sum(item.exported_bytes for item in exported_sorted),
        },
        "files": [
            {
                "logical_path": item.logical_path,
                "source_bytes": item.source_bytes,
                "exported_bytes": item.exported_bytes,
                "exported_sha256": item.exported_sha256,
                "source_sha256": item.source_sha256,
                "asset": item.asset,
                "redaction_classification": item.redaction_classification,
            }
            for item in exported_sorted
        ],
        "assets": [
            dataclasses.asdict(asset)
            for asset in sorted(assets, key=lambda item: item.name)
        ],
        "exclusions": [
            {
                "logical_path": record.logical_path,
                "source_bytes": record.source_bytes,
                "category": record.exclusion_category,
                "reason": EXCLUSION_REASONS[record.exclusion_category or ""],
            }
            for record in sorted(excluded_records, key=lambda item: item.logical_path)
        ],
        "redaction_ledger": {
            "category_counts": dict(sorted(total_ledger.items())),
            "files": ledger_files,
        },
    }


def write_json_atomic(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise ExportError(f"temporary manifest path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_root_list(values: list[str], label: str) -> list[Path]:
    roots: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if not path.is_absolute() or path == Path("/"):
            raise ExportError(
                f"{label} roots must be absolute and must not be /: {value}"
            )
        roots.append(path)
        resolved = path.resolve(strict=False)
        if resolved != path:
            roots.append(resolved)
    return roots


def parse_approved_canaries(value: str) -> frozenset[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ExportError("--approved-canaries must be a JSON string-list") from exc
    if (
        not isinstance(parsed, list)
        or any(not isinstance(item, str) or not item for item in parsed)
        or len(set(parsed)) != len(parsed)
    ):
        raise ExportError(
            "--approved-canaries must be a duplicate-free JSON list of non-empty strings"
        )
    try:
        for item in parsed:
            item.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ExportError("--approved-canaries values must be valid UTF-8") from exc
    return frozenset(parsed)


def export(args: argparse.Namespace) -> dict[str, object]:
    source_root = Path(args.source_root).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise ExportError("source root must be a directory")
    requested_asset_dir = Path(args.asset_dir).expanduser().absolute()
    requested_manifest = Path(args.manifest).expanduser().absolute()
    if requested_asset_dir.exists() or requested_asset_dir.is_symlink():
        raise ExportError(f"asset directory already exists: {requested_asset_dir}")
    if requested_manifest.exists() or requested_manifest.is_symlink():
        raise ExportError(f"manifest already exists: {requested_manifest}")
    asset_dir = requested_asset_dir.resolve(strict=False)
    manifest_path = requested_manifest.resolve(strict=False)
    if path_is_within(manifest_path, asset_dir):
        raise ExportError("manifest path must be outside the release asset directory")
    if path_is_within(asset_dir, source_root) or path_is_within(
        manifest_path, source_root
    ):
        raise ExportError("outputs must be outside the immutable source root")
    if args.max_asset_bytes > GITHUB_ASSET_LIMIT:
        raise ExportError(f"--max-asset-bytes may not exceed {GITHUB_ASSET_LIMIT}")
    if args.max_asset_bytes < 1024 * 1024:
        raise ExportError("--max-asset-bytes must be at least 1 MiB")
    if not 1 <= args.threads <= 8:
        raise ExportError("--threads must be between 1 and 8")
    if not 1 <= args.compression_level <= 19:
        raise ExportError("--compression-level must be between 1 and 19")

    zstd = shutil.which(args.zstd)
    if not zstd:
        raise ExportError(f"zstd executable not found: {args.zstd}")
    model_roots = parse_root_list(args.model_root, "model")
    cache_roots = parse_root_list(args.cache_root, "cache")
    approved_canaries = parse_approved_canaries(args.approved_canaries)
    redactor = Redactor(
        source_root,
        model_roots,
        cache_roots,
        approved_canaries,
    )
    validate_logical_path(source_root.name)
    redactor.assert_safe_metadata(source_root.name, "source root label")
    raw_target = args.raw_shard_bytes or min(
        1_750_000_000, int(args.max_asset_bytes * 0.85)
    )
    if raw_target >= args.max_asset_bytes or raw_target < 1024 * 1024:
        raise ExportError(
            "--raw-shard-bytes must be at least 1 MiB and below --max-asset-bytes"
        )

    records = scan_source(source_root, redactor)
    exported_records = [record for record in records if not record.exclusion_category]
    array_records = [
        record
        for record in exported_records
        if record.logical_path.lower().endswith(".npy")
    ]
    regular_records = [
        record for record in exported_records if record not in array_records
    ]

    asset_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".evidence-export-", dir=asset_dir.parent))
    staged_files = work / "files"
    staged_assets = work / "assets"
    staged_files.mkdir()
    staged_assets.mkdir()
    files: list[ExportedFile] = []
    assets: list[Asset] = []

    try:
        for record in regular_records:
            files.append(stage_regular(record, staged_files, redactor))

        shards = shard_files(files, raw_target)
        planned_asset_count = len(shards) + len(array_records)
        if planned_asset_count > MAX_RELEASE_ASSETS:
            raise ExportError(
                f"export requires {planned_asset_count} assets; "
                f"GitHub releases permit {MAX_RELEASE_ASSETS}"
            )
        for index, shard in enumerate(shards, 1):
            name = f"evidence-{index:04d}.tar.zst"
            validate_logical_path(name)
            redactor.assert_safe_metadata(name, "release asset name")
            path = staged_assets / name
            create_tar_asset(shard, path, zstd, args.threads, args.compression_level)
            for item in shard:
                item.asset = name
            assets.append(checked_asset(path, "tar+zstd", shard, args.max_asset_bytes))

        for index, record in enumerate(
            sorted(array_records, key=lambda item: item.logical_path), 1
        ):
            name = array_asset_name(index, record.logical_path)
            validate_logical_path(name)
            redactor.assert_safe_metadata(name, "release asset name")
            path = staged_assets / name
            item = create_array_asset(
                record,
                path,
                zstd,
                args.threads,
                args.compression_level,
                redactor,
            )
            item.asset = name
            files.append(item)
            assets.append(checked_asset(path, "npy+zstd", [item], args.max_asset_bytes))

        if len(assets) > MAX_RELEASE_ASSETS:
            raise ExportError(
                f"export requires {len(assets)} assets; GitHub releases permit {MAX_RELEASE_ASSETS}"
            )
        if any(item.asset is None for item in files):
            raise ExportError("internal error: an exported file has no owning asset")

        final_records = scan_source(source_root, redactor)
        if final_records != records:
            raise ExportError(
                "source tree changed during export; discard the staged export and retry"
            )

        document = manifest_document(source_root, records, files, assets)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_temporary = manifest_path.with_name(
            f".{manifest_path.name}.{os.getpid()}.ready"
        )
        with manifest_temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        assets_installed = False
        try:
            os.replace(staged_assets, asset_dir)
            assets_installed = True
            os.replace(manifest_temporary, manifest_path)
        except BaseException:
            if assets_installed:
                shutil.rmtree(asset_dir, ignore_errors=True)
            raise
        finally:
            manifest_temporary.unlink(missing_ok=True)

        return {
            "schema": SCHEMA,
            "exported_files": len(files),
            "excluded_files": len(records) - len(files),
            "assets": len(assets),
            "redaction_category_counts": document["redaction_ledger"][
                "category_counts"
            ],
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a complete evidence tree into deterministic, verified release assets."
    )
    parser.add_argument(
        "--source-root", required=True, help="immutable evidence root to export"
    )
    parser.add_argument(
        "--asset-dir", required=True, help="new directory for release assets"
    )
    parser.add_argument(
        "--manifest", required=True, help="new glm53-orca-evidence.v1 index path"
    )
    parser.add_argument(
        "--model-root",
        action="append",
        default=[],
        help="absolute model-root prefix to replace; repeat as needed",
    )
    parser.add_argument(
        "--cache-root",
        action="append",
        default=[],
        help="absolute cache-root prefix to replace; repeat as needed",
    )
    parser.add_argument(
        "--approved-canaries",
        default="[]",
        help="JSON list of exact credential canary strings permitted to remain",
    )
    parser.add_argument("--zstd", default="zstd", help="zstd executable name or path")
    parser.add_argument(
        "--threads", type=int, default=2, help="bounded zstd worker threads (1-8)"
    )
    parser.add_argument(
        "--compression-level", type=int, default=10, help="zstd level (1-19)"
    )
    parser.add_argument(
        "--max-asset-bytes",
        type=int,
        default=GITHUB_ASSET_LIMIT,
        help="strict exclusive asset-size ceiling; may not exceed 2 GiB",
    )
    parser.add_argument(
        "--raw-shard-bytes",
        type=int,
        help="deterministic uncompressed tar shard target below the asset ceiling",
    )
    return parser


def main() -> int:
    try:
        summary = export(build_parser().parse_args())
    except (ExportError, OSError, subprocess.SubprocessError) as exc:
        print(f"export_evidence: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
