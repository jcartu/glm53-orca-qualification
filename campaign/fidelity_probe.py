#!/usr/bin/env python3
"""Prepare, capture, and compare full-vocabulary GLM fidelity panels."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time
import traceback
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SOURCE_SCHEMA = "orcarouter-fidelity-panel-source.v1"
PREPARED_SCHEMA = "orcarouter-fidelity-panels.v1"
CAPTURE_SCHEMA = "orcarouter-fidelity-capture.v1"
CAPTURE_SUMMARY_SCHEMA = "orcarouter-fidelity-capture-summary.v1"
COMPARE_SCHEMA = "orcarouter-fidelity-comparison.v1"
FAILURE_SCHEMA = "orcarouter-fidelity-failure.v1"
CONTEXT_LENGTH = 2048
SCORED_ROWS = CONTEXT_LENGTH - 1
WIKITEXT_WINDOWS = 12
PRIMARY_SCORED_POSITIONS = WIKITEXT_WINDOWS * SCORED_ROWS
EXPECTED_TOKENIZER_VOCAB_SIZE = 154856
EXPECTED_MODEL_OUTPUT_VOCAB_SIZE = 154880
NORMALIZATION_TOLERANCE = 1.0e-3
EXPECTED_VLLM_VERSION = "0.26.1rc0+glm53.r38.vllm66c29357"
TOKEN_HASH_DOMAIN = b"orcarouter-fidelity-token-ids-le-i32-v1\0"
FIXED_CAPTURE_ARGS: dict[str, Any] = {
    "tensor_parallel_size": 4,
    "decode_context_parallel_size": 1,
    "max_model_len": 4096,
    "max_num_batched_tokens": 256,
    "max_num_seqs": 1,
    "kv_cache_dtype": "fp8",
    "kv_cache_memory_bytes": 2 * 1024**3,
    "enforce_eager": True,
    "enable_prefix_caching": False,
    "skip_tokenizer_init": True,
    "max_logprobs": -1,
    "logprobs_mode": "raw_logprobs",
    "disable_log_stats": True,
}
PINNED_CHECKPOINTS = {
    "zai-org/GLM-5.3-Flash": {
        "revision": "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
        "quantization": "fp8",
    },
    "orcarouter/GLM-5.3-Flash-Uncensored-FP8": {
        "revision": "3cec42d6ed14ec197e328c09650c17fd3660c26a",
        "quantization": "fp8",
    },
    "orcarouter/GLM-5.3-Flash-Uncensored-NVFP4": {
        "revision": "ec0adf4f49c9570807cc11a5f650538c1893ae54",
        "quantization": "compressed-tensors",
    },
}
SPECULATION_KEYS = {
    "speculative_config",
    "speculative_model",
    "num_speculative_tokens",
    "spec_method",
    "spec_model",
    "spec_tokens",
}
MODEL_SPECIFIC_LLM_ARGS = {
    "model",
    "tokenizer",
    "revision",
    "tokenizer_revision",
    "quantization",
    "quantization_config",
    "load_format",
    "trust_remote_code",
}
MODEL_PRECISION_ENV = {
    "VLLM_B12X_DENSE_ACTIVATION_MODE",
    "VLLM_B12X_NVFP4_ACTIVATION_MODE",
    "VLLM_B12X_MXFP8_ACTIVATION_MODE",
    "VLLM_B12X_MOE_FP4_FORCE_A16",
    "VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER",
    "VLLM_DEFAULT_MOE_BACKEND",
}
RUNTIME_SOURCE_FILES = (
    "_version.py",
    "sampling_params.py",
    "logprobs.py",
    "v1/engine/logprobs.py",
    "v1/worker/gpu/sample/logprob.py",
    "v1/worker/gpu/sample/prompt_logprob.py",
    "model_executor/layers/quantization/fp8.py",
    "model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py",
    "model_executor/layers/fused_moe/experts/marlin_moe.py",
)
TOKENIZER_FILENAMES = {
    "added_tokens.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
}
DISTRIBUTION_BOUNDS = (
    0.0,
    1.0e-8,
    1.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    5.0e-2,
    1.0e-1,
    2.5e-1,
    5.0e-1,
    1.0,
    2.0,
    5.0,
    10.0,
)


class FidelityError(RuntimeError):
    status = "fail"


class UnsupportedError(FidelityError):
    status = "unsupported"


class UnobservableError(FidelityError):
    status = "unobservable"


class IncomparableError(FidelityError):
    status = "incomparable"


class IntegrityError(FidelityError):
    status = "fail"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return jsonable(value.to_dict())
        except Exception:
            pass
    return str(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        jsonable(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def token_ids_sha256(token_ids: list[int] | np.ndarray) -> str:
    values = np.asarray(token_ids, dtype="<i4")
    digest = hashlib.sha256()
    digest.update(TOKEN_HASH_DOMAIN)
    digest.update(struct.pack("<Q", int(values.size)))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            jsonable(payload),
            handle,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def save_npy_exclusive(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)


def save_npz_exclusive(path: Path, **values: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **values)


def create_output_dir(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to reuse historical output directory: {path}")
    path.mkdir(parents=True)


def failure_payload(command: str, exc: BaseException) -> dict[str, Any]:
    status = getattr(exc, "status", "fail")
    return {
        "schema": FAILURE_SCHEMA,
        "command": command,
        "status": status,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "timestamp_utc": utc_now(),
        "traceback": traceback.format_exc(),
    }


def record_failure(path: Path, command: str, exc: BaseException) -> None:
    try:
        if not path.exists():
            write_json_exclusive(path, failure_payload(command, exc))
    except Exception as record_exc:
        print(
            f"could not preserve failure artifact {path}: {record_exc}", file=sys.stderr
        )


def require_schema(payload: dict[str, Any], schema: str, path: Path) -> None:
    if payload.get("schema") != schema:
        raise IntegrityError(
            f"Expected schema {schema!r} in {path}; got {payload.get('schema')!r}"
        )


def validate_token_ids(token_ids: Any, *, context: str) -> list[int]:
    if not isinstance(token_ids, list) or len(token_ids) != CONTEXT_LENGTH:
        raise IntegrityError(
            f"{context} must contain exactly {CONTEXT_LENGTH} integer token IDs"
        )
    if not all(
        isinstance(item, int) and not isinstance(item, bool) for item in token_ids
    ):
        raise IntegrityError(f"{context} contains a non-integer token ID")
    if min(token_ids) < 0:
        raise IntegrityError(f"{context} contains a negative token ID")
    return token_ids


def tokenizer_file_hashes(tokenizer_path: Path) -> dict[str, str]:
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(
            f"Tokenizer path is not a local directory: {tokenizer_path}"
        )
    hashes: dict[str, str] = {}
    for name in sorted(TOKENIZER_FILENAMES):
        path = tokenizer_path / name
        if path.is_file():
            hashes[name] = sha256_file(path)
    if "tokenizer.json" not in hashes and not any(
        name.endswith(".model") for name in hashes
    ):
        raise IntegrityError(
            f"No tokenizer.json or tokenizer model file found in {tokenizer_path}"
        )
    return hashes


def quiet_encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(encoded, list) or not all(
        isinstance(item, int) for item in encoded
    ):
        raise IntegrityError("Tokenizer did not return a list of integer token IDs")
    return encoded


def tokenizer_identity(
    tokenizer: Any, tokenizer_path: Path, revision: str
) -> dict[str, Any]:
    probe_texts = (
        "Plain ASCII, punctuation, and 0123456789.",
        "Unicode probe: café, Ελληνικά, 中文, العربية, emoji-free.",
        "def f(x: int) -> int:\n    return x * x\n",
    )
    probes = [quiet_encode(tokenizer, text) for text in probe_texts]
    identity = {
        "source_path": str(tokenizer_path.resolve()),
        "pinned_reference_revision": revision,
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__name__}",
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size_property": int(getattr(tokenizer, "vocab_size", len(tokenizer))),
        "len": int(len(tokenizer)),
        "special_token_ids": {
            name: jsonable(getattr(tokenizer, name, None))
            for name in (
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "unk_token_id",
            )
        },
        "files_sha256": tokenizer_file_hashes(tokenizer_path),
        "probe_token_ids_sha256": [token_ids_sha256(items) for items in probes],
    }
    identity["identity_sha256"] = object_sha256(identity)
    return identity


def obtain_wikitext_parquet(
    source: dict[str, Any], supplied_path: Path | None
) -> tuple[Path, bool]:
    if supplied_path is not None:
        path = supplied_path
        downloaded = False
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise UnsupportedError(
                "prepare requires huggingface_hub when --wikitext-parquet is omitted"
            ) from exc
        path = Path(
            hf_hub_download(
                repo_id=source["dataset"],
                repo_type="dataset",
                filename=source["file"],
                revision=source["revision"],
            )
        )
        downloaded = True
    if not path.is_file():
        raise FileNotFoundError(f"WikiText parquet is missing: {path}")
    observed = sha256_file(path)
    if observed != source["file_sha256"]:
        raise IntegrityError(
            f"WikiText parquet SHA-256 mismatch: expected {source['file_sha256']}, "
            f"observed {observed}"
        )
    return path, downloaded


def read_wikitext_text(path: Path, join_separator: str) -> tuple[str, int, int]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise UnsupportedError(
            "prepare requires pyarrow to read pinned WikiText"
        ) from exc
    table = parquet.read_table(path, columns=["text"])
    rows = table.column("text").to_pylist()
    selected = [value for value in rows if isinstance(value, str) and value.strip()]
    if not selected:
        raise IntegrityError(
            "Pinned WikiText test parquet contains no non-empty text rows"
        )
    return join_separator.join(selected), len(rows), len(selected)


def command_prepare(args: argparse.Namespace) -> int:
    failure_path = args.out.with_suffix(args.out.suffix + ".failure.json")
    try:
        if args.out.exists():
            raise FileExistsError(f"Refusing to overwrite prepared panels: {args.out}")
        source = load_json(args.panels)
        require_schema(source, SOURCE_SCHEMA, args.panels)
        if int(source.get("context_length", -1)) != CONTEXT_LENGTH:
            raise IntegrityError("Panel source has an unexpected context length")
        if int(source.get("wikitext", {}).get("window_count", -1)) != WIKITEXT_WINDOWS:
            raise IntegrityError(
                "Panel source must request exactly 12 WikiText windows"
            )

        tokenizer_source = source.get("tokenizer_source")
        if not isinstance(tokenizer_source, dict):
            raise IntegrityError("Panel source has no tokenizer_source object")
        expected_revision = tokenizer_source.get("revision")
        if args.tokenizer_revision != expected_revision:
            raise IntegrityError(
                f"Tokenizer revision must be the pinned original revision "
                f"{expected_revision}; got {args.tokenizer_revision}"
            )
        tokenizer_path = args.tokenizer.resolve()
        if not tokenizer_path.is_dir():
            raise FileNotFoundError(
                f"Local tokenizer material is not staged: {tokenizer_path}"
            )

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise UnsupportedError("prepare requires transformers") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            local_files_only=True,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer_meta = tokenizer_identity(
            tokenizer, tokenizer_path, args.tokenizer_revision
        )
        expected_tokenizer_files = tokenizer_source.get("required_files_sha256")
        if tokenizer_meta["files_sha256"] != expected_tokenizer_files:
            raise IntegrityError(
                "Local tokenizer files do not match the pinned original tokenizer: "
                + json.dumps(
                    {
                        "expected": expected_tokenizer_files,
                        "observed": tokenizer_meta["files_sha256"],
                    },
                    sort_keys=True,
                )
            )
        if tokenizer_meta["len"] != EXPECTED_TOKENIZER_VOCAB_SIZE:
            raise IntegrityError(
                f"Pinned tokenizer length is {tokenizer_meta['len']}; expected "
                f"{EXPECTED_TOKENIZER_VOCAB_SIZE}"
            )
        tokenizer_meta["model_output_vocab_size"] = EXPECTED_MODEL_OUTPUT_VOCAB_SIZE
        tokenizer_meta["output_rows_beyond_tokenizer_size"] = (
            EXPECTED_MODEL_OUTPUT_VOCAB_SIZE - EXPECTED_TOKENIZER_VOCAB_SIZE
        )

        wikitext = source["wikitext"]
        parquet_path, downloaded = obtain_wikitext_parquet(
            wikitext, args.wikitext_parquet
        )
        joined_text, source_rows, selected_rows = read_wikitext_text(
            parquet_path, wikitext["join_separator"]
        )
        wiki_token_ids = quiet_encode(tokenizer, joined_text)
        stride = int(wikitext["stride_tokens"])
        if stride <= 0 or int(wikitext["window_length_tokens"]) != CONTEXT_LENGTH:
            raise IntegrityError("WikiText window length or stride is invalid")
        required_tokens = CONTEXT_LENGTH + (WIKITEXT_WINDOWS - 1) * stride
        if len(wiki_token_ids) < required_tokens:
            raise IntegrityError(
                f"WikiText token stream has {len(wiki_token_ids)} tokens; "
                f"{required_tokens} are required"
            )

        panels: list[dict[str, Any]] = []
        for index in range(WIKITEXT_WINDOWS):
            start = index * stride
            end = start + CONTEXT_LENGTH
            token_ids = wiki_token_ids[start:end]
            panels.append(
                {
                    "index": index,
                    "id": f"wikitext2-test-{index:02d}",
                    "corpus": "wikitext-2-raw-v1:test",
                    "stratum": "held_out_natural_text",
                    "role": "primary",
                    "input_token_ids": token_ids,
                    "input_token_ids_sha256": token_ids_sha256(token_ids),
                    "source_token_span": [start, end],
                    "prompt_boundary": {
                        "input_positions": [0, CONTEXT_LENGTH],
                        "scored_context_positions": [0, SCORED_ROWS],
                        "gold_target_positions": [1, CONTEXT_LENGTH],
                    },
                    "independent_sequence": True,
                }
            )

        custom_panels = source.get("custom_panels")
        if not isinstance(custom_panels, list) or len(custom_panels) != 4:
            raise IntegrityError("Panel source must contain exactly four custom panels")
        for custom_offset, custom in enumerate(custom_panels):
            text = custom.get("text")
            if not isinstance(text, str) or not text:
                raise IntegrityError("Custom panel text must be non-empty")
            observed_text_hash = sha256_text(text)
            if observed_text_hash != custom.get("text_sha256"):
                raise IntegrityError(
                    f"Custom source hash mismatch for {custom.get('id')}: "
                    f"{observed_text_hash}"
                )
            separator = custom.get("repeat_separator", "\n\n")
            if not isinstance(separator, str):
                raise IntegrityError("repeat_separator must be a string")
            expanded = text
            repeats = 1
            token_ids = quiet_encode(tokenizer, expanded)
            while len(token_ids) < CONTEXT_LENGTH:
                expanded += separator + text
                repeats += 1
                token_ids = quiet_encode(tokenizer, expanded)
                if repeats > 64:
                    raise IntegrityError(
                        f"Custom panel {custom.get('id')} did not reach 2048 tokens"
                    )
            token_ids = token_ids[:CONTEXT_LENGTH]
            index = WIKITEXT_WINDOWS + custom_offset
            panels.append(
                {
                    "index": index,
                    "id": custom["id"],
                    "corpus": "campaign-authored-benign-supplement",
                    "stratum": custom["stratum"],
                    "role": "supplemental",
                    "license": custom.get("license", "CC0-1.0"),
                    "input_token_ids": token_ids,
                    "input_token_ids_sha256": token_ids_sha256(token_ids),
                    "source_text_sha256": observed_text_hash,
                    "expanded_text_sha256": sha256_text(expanded),
                    "source_repetitions": repeats,
                    "source_token_span": [0, CONTEXT_LENGTH],
                    "prompt_boundary": {
                        "input_positions": [0, CONTEXT_LENGTH],
                        "scored_context_positions": [0, SCORED_ROWS],
                        "gold_target_positions": [1, CONTEXT_LENGTH],
                    },
                    "independent_sequence": True,
                }
            )

        primary_positions = sum(
            SCORED_ROWS for panel in panels if panel["role"] == "primary"
        )
        if primary_positions != PRIMARY_SCORED_POSITIONS:
            raise IntegrityError(
                f"Primary scored-position count is {primary_positions}; expected "
                f"{PRIMARY_SCORED_POSITIONS}"
            )
        suite_hash = object_sha256(
            [
                {
                    "index": panel["index"],
                    "id": panel["id"],
                    "token_hash": panel["input_token_ids_sha256"],
                }
                for panel in panels
            ]
        )
        prepared = {
            "schema": PREPARED_SCHEMA,
            "status": "prepared",
            "created_utc": utc_now(),
            "source_spec": str(args.panels.resolve()),
            "source_spec_sha256": sha256_file(args.panels),
            "context_length": CONTEXT_LENGTH,
            "scored_rows_per_panel": SCORED_ROWS,
            "primary_scored_positions": primary_positions,
            "total_scored_positions": len(panels) * SCORED_ROWS,
            "panel_count": len(panels),
            "suite_token_hash_sha256": suite_hash,
            "token_hash_algorithm": "sha256(domain || uint64_le_length || int32_le_bytes)",
            "tokenizer": tokenizer_meta,
            "vocabulary_contract": {
                "tokenizer_size": EXPECTED_TOKENIZER_VOCAB_SIZE,
                "model_output_size": EXPECTED_MODEL_OUTPUT_VOCAB_SIZE,
                "non_token_output_id_range": [
                    EXPECTED_TOKENIZER_VOCAB_SIZE,
                    EXPECTED_MODEL_OUTPUT_VOCAB_SIZE,
                ],
                "policy": (
                    "Gold inputs must be tokenizer IDs; full-distribution capture and "
                    "normalization include every model-output row, including the padded "
                    "non-token tail."
                ),
            },
            "wikitext_source": {
                "dataset": wikitext["dataset"],
                "revision": wikitext["revision"],
                "config": wikitext["config"],
                "split": wikitext["split"],
                "file": wikitext["file"],
                "file_sha256": wikitext["file_sha256"],
                "local_path": str(parquet_path.resolve()),
                "downloaded_by_prepare": downloaded,
                "source_rows": source_rows,
                "selected_nonempty_rows": selected_rows,
                "join_separator": wikitext["join_separator"],
                "joined_text_sha256": sha256_text(joined_text),
                "full_token_stream_count": len(wiki_token_ids),
                "full_token_stream_sha256": token_ids_sha256(wiki_token_ids),
                "windowing": {
                    "count": WIKITEXT_WINDOWS,
                    "length_tokens": CONTEXT_LENGTH,
                    "stride_tokens": stride,
                    "start_offsets": [
                        index * stride for index in range(WIKITEXT_WINDOWS)
                    ],
                    "capture_state": "fresh independent prompt per window",
                },
            },
            "alignment": {
                "add_special_tokens": False,
                "each_panel_is_an_independent_prompt": True,
                "cross_window_kv_or_recurrent_state": False,
                "row_i_distribution_predicts_input_token_i_plus_1": True,
                "final_prompt_distribution_unscored": True,
            },
            "scope": {
                "primary": (
                    "24,564 teacher-forced positions from the pinned WikiText-2 "
                    "raw test split"
                ),
                "supplemental": (
                    "four deterministic campaign-authored benign windows; descriptive "
                    "strata only"
                ),
                "published_replication": False,
                "published_replication_limitation": (
                    "The author's exact token IDs and capture artifact are unavailable; "
                    "this is a matched campaign comparison, not an exact replication of "
                    "a published scalar."
                ),
            },
            "panels": panels,
        }
        write_json_exclusive(args.out, prepared)
        result = {
            "schema": "orcarouter-fidelity-prepare-summary.v1",
            "status": "pass",
            "output": str(args.out.resolve()),
            "output_sha256": sha256_file(args.out),
            "panels": len(panels),
            "primary_scored_positions": primary_positions,
            "total_scored_positions": len(panels) * SCORED_ROWS,
            "suite_token_hash_sha256": suite_hash,
            "tokenizer_identity_sha256": tokenizer_meta["identity_sha256"],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        record_failure(failure_path, "prepare", exc)
        raise


def load_llm_args(path: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    payload = load_json(path)
    if "llm_args" in payload:
        llm_args = payload["llm_args"]
        declared_metadata = payload.get("metadata", {})
        if not isinstance(llm_args, dict) or not isinstance(declared_metadata, dict):
            raise TypeError(
                "llm-args wrapper requires object llm_args and metadata fields"
            )
    else:
        llm_args = payload
        declared_metadata = {}
    for sensitive in ("hf_token", "token", "password", "secret"):
        if sensitive in llm_args:
            raise IntegrityError(
                f"Refusing credential-bearing LLM argument {sensitive!r}; use local weights"
            )
    return dict(llm_args), dict(declared_metadata), sha256_file(path)


def validate_checkpoint_metadata(
    metadata: dict[str, Any], llm_args: dict[str, Any]
) -> None:
    repo = metadata.get("checkpoint_repo")
    revision = metadata.get("checkpoint_revision")
    if repo not in PINNED_CHECKPOINTS:
        raise IntegrityError(
            "llm-args metadata.checkpoint_repo must name one of the three pinned "
            "campaign checkpoints"
        )
    expected = PINNED_CHECKPOINTS[repo]
    if revision != expected["revision"]:
        raise IntegrityError(
            f"Pinned revision mismatch for {repo}: expected {expected['revision']}, "
            f"got {revision}"
        )
    if llm_args.get("quantization") != expected["quantization"]:
        raise IntegrityError(
            f"Checkpoint {repo} requires quantization={expected['quantization']!r}; "
            f"got {llm_args.get('quantization')!r}"
        )


def prepare_capture_llm_args(model: Path, supplied: dict[str, Any]) -> dict[str, Any]:
    if not model.is_dir():
        raise FileNotFoundError(
            f"Capture model must be a staged local directory: {model}"
        )
    result = dict(supplied)
    supplied_model = result.pop("model", None)
    if supplied_model is not None and Path(supplied_model).resolve() != model.resolve():
        raise IntegrityError(
            f"LLM args model {supplied_model!r} differs from --model {str(model)!r}"
        )
    if "tokenizer" in result:
        raise IntegrityError(
            "capture uses prepared token IDs and forbids a tokenizer argument"
        )
    for key in SPECULATION_KEYS:
        value = result.get(key)
        if value not in (None, False, 0, "", "none", "None"):
            raise IntegrityError(
                f"Speculation is forbidden for fidelity capture: {key}={value}"
            )
        result.pop(key, None)
    for key, required in FIXED_CAPTURE_ARGS.items():
        if key in result and result[key] != required:
            raise IntegrityError(
                f"Fidelity contract requires LLM argument {key}={required!r}; "
                f"got {result[key]!r}"
            )
        result[key] = required
    if "quantization" not in result:
        raise IntegrityError(
            "llm-args must explicitly select quantization: fp8 for source checkpoints "
            "or compressed-tensors for NVFP4"
        )
    result["model"] = str(model.resolve())
    return result


def environment_metadata() -> dict[str, str]:
    selected: dict[str, str] = {}
    for key, value in os.environ.items():
        if key.startswith(("VLLM_", "B12X_", "NCCL_")) or key in {
            "CUDA_VISIBLE_DEVICES",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "OMP_NUM_THREADS",
            "PYTORCH_CUDA_ALLOC_CONF",
        }:
            selected[key] = value
    return dict(sorted(selected.items()))


def model_identity(model: Path) -> dict[str, Any]:
    files = {}
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = model / name
        if path.is_file():
            files[name] = sha256_file(path)
    config_path = model / "config.json"
    config = load_json(config_path) if config_path.is_file() else {}
    return {
        "path": str(model.resolve()),
        "directory_name": model.resolve().name,
        "small_files_sha256": files,
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "vocab_size": config.get("vocab_size"),
        "torch_dtype": config.get("torch_dtype", config.get("dtype")),
        "quantization_config": config.get("quantization_config"),
    }


def runtime_source_hashes(vllm_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in RUNTIME_SOURCE_FILES:
        path = vllm_root / relative
        if not path.is_file():
            raise UnobservableError(
                f"Required vLLM runtime source file is missing: {path}"
            )
        hashes[relative] = sha256_file(path)
    return hashes


def extract_runtime_metadata(llm: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import torch
        import vllm

        config = llm.llm_engine.vllm_config
        model = config.model_config
        cache = config.cache_config
        parallel = config.parallel_config
        scheduler = config.scheduler_config
        compilation = config.compilation_config
    except Exception as exc:
        raise UnobservableError(
            f"Cannot inspect active vLLM configuration: {exc}"
        ) from exc

    vllm_root = Path(vllm.__file__).resolve().parent
    device_count = torch.cuda.device_count()
    devices = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [properties.major, properties.minor],
            }
        )
    engine = {
        "tensor_parallel_size": int(parallel.tensor_parallel_size),
        "pipeline_parallel_size": int(parallel.pipeline_parallel_size),
        "decode_context_parallel_size": int(parallel.decode_context_parallel_size),
        "data_parallel_size": int(parallel.data_parallel_size),
        "max_model_len": int(model.max_model_len),
        "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
        "max_num_seqs": int(scheduler.max_num_seqs),
        "enable_chunked_prefill": bool(scheduler.enable_chunked_prefill),
        "dtype": str(model.dtype),
        "quantization": jsonable(model.quantization),
        "max_logprobs": int(model.max_logprobs),
        "logprobs_mode": str(model.logprobs_mode),
        "enforce_eager": bool(model.enforce_eager),
        "vocab_size": int(model.get_vocab_size()),
        "architectures": list(model.architectures or []),
        "cache_dtype": str(cache.cache_dtype),
        "kv_cache_memory_bytes": jsonable(cache.kv_cache_memory_bytes),
        "enable_prefix_caching": bool(cache.enable_prefix_caching),
        "block_size": int(cache.block_size),
        "kv_cache_layout": jsonable(cache.kv_cache_layout),
        "kv_cache_dtype_skip_layers": list(cache.kv_cache_dtype_skip_layers),
        "num_gpu_blocks": jsonable(cache.num_gpu_blocks),
        "kv_cache_size_tokens": jsonable(cache.kv_cache_size_tokens),
        "prefix_match_unit": jsonable(cache.prefix_match_unit),
        "recurrent_checkpoint_policy": str(cache.recurrent_checkpoint_policy),
        "mamba_cache_dtype": str(cache.mamba_cache_dtype),
        "mamba_ssm_cache_dtype": str(cache.mamba_ssm_cache_dtype),
        "mamba_cache_mode": str(cache.mamba_cache_mode),
        "mamba_block_size": jsonable(cache.mamba_block_size),
        "effective_kda_state_dtypes": [
            (
                str(model.dtype)
                if str(cache.mamba_cache_dtype) == "auto"
                else str(cache.mamba_cache_dtype)
            ),
            "torch.float32",
        ],
        "speculative_config": jsonable(config.speculative_config),
        "use_v2_model_runner": bool(config.use_v2_model_runner),
        "compilation_mode": str(compilation.mode),
        "cudagraph_mode": str(compilation.cudagraph_mode),
    }
    runtime = {
        "vllm_version": vllm.__version__,
        "vllm_package_root": str(vllm_root),
        "runtime_source_sha256": runtime_source_hashes(vllm_root),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_devices": devices,
        "environment": environment_metadata(),
    }
    return engine, runtime


def validate_active_capture_contract(engine: dict[str, Any], vocab_size: int) -> None:
    expected = {
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 1,
        "max_model_len": 4096,
        "max_num_batched_tokens": 256,
        "max_num_seqs": 1,
        "max_logprobs": -1,
        "logprobs_mode": "raw_logprobs",
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "vocab_size": vocab_size,
    }
    mismatches = {
        key: {"expected": value, "observed": engine.get(key)}
        for key, value in expected.items()
        if engine.get(key) != value
    }
    if engine.get("speculative_config") is not None:
        mismatches["speculative_config"] = {
            "expected": None,
            "observed": engine["speculative_config"],
        }
    if str(engine.get("cache_dtype")) not in {"fp8", "fp8_e4m3"}:
        mismatches["cache_dtype"] = {
            "expected": "fp8/fp8_e4m3",
            "observed": engine.get("cache_dtype"),
        }
    if engine.get("kv_cache_memory_bytes") != 2 * 1024**3:
        mismatches["kv_cache_memory_bytes"] = {
            "expected": 2 * 1024**3,
            "observed": engine.get("kv_cache_memory_bytes"),
        }
    if mismatches:
        raise IntegrityError(
            "Active vLLM configuration violates the capture contract: "
            + json.dumps(mismatches, sort_keys=True)
        )


def comparison_contract(
    prepared: dict[str, Any],
    engine: dict[str, Any],
    runtime: dict[str, Any],
    llm_args: dict[str, Any],
) -> dict[str, Any]:
    shared_engine_keys = (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "decode_context_parallel_size",
        "data_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "enable_chunked_prefill",
        "dtype",
        "max_logprobs",
        "logprobs_mode",
        "enforce_eager",
        "vocab_size",
        "architectures",
        "cache_dtype",
        "kv_cache_memory_bytes",
        "enable_prefix_caching",
        "block_size",
        "kv_cache_layout",
        "kv_cache_dtype_skip_layers",
        "num_gpu_blocks",
        "kv_cache_size_tokens",
        "prefix_match_unit",
        "recurrent_checkpoint_policy",
        "mamba_cache_dtype",
        "mamba_ssm_cache_dtype",
        "mamba_cache_mode",
        "mamba_block_size",
        "effective_kda_state_dtypes",
        "speculative_config",
        "use_v2_model_runner",
        "compilation_mode",
        "cudagraph_mode",
    )
    environment = runtime["environment"]
    shared_environment = {
        key: value
        for key, value in environment.items()
        if key not in MODEL_PRECISION_ENV
    }
    return {
        "schema": "orcarouter-fidelity-comparison-contract.v1",
        "prepared_panels_canonical_sha256": object_sha256(prepared),
        "suite_token_hash_sha256": prepared["suite_token_hash_sha256"],
        "tokenizer_identity_sha256": prepared["tokenizer"]["identity_sha256"],
        "context_length": CONTEXT_LENGTH,
        "scored_rows_per_panel": SCORED_ROWS,
        "panels": [
            {
                "index": panel["index"],
                "id": panel["id"],
                "corpus": panel["corpus"],
                "stratum": panel["stratum"],
                "role": panel["role"],
                "input_token_ids_sha256": panel["input_token_ids_sha256"],
            }
            for panel in prepared["panels"]
        ],
        "capture_semantics": {
            "teacher_forced": True,
            "semantic_point": "raw_prompt_logprobs_before_sampling_processors",
            "full_vocabulary": True,
            "stored_dtype": "float32",
            "independent_prompts": True,
            "prefix_cache": False,
            "speculation": False,
        },
        "effective_shared_llm_args": {
            key: value
            for key, value in sorted(llm_args.items())
            if key not in MODEL_SPECIFIC_LLM_ARGS
        },
        "engine": {key: engine[key] for key in shared_engine_keys},
        "runtime": {
            "vllm_version": runtime["vllm_version"],
            "runtime_source_sha256": runtime["runtime_source_sha256"],
            "python_version": runtime["python_version"],
            "torch_version": runtime["torch_version"],
            "torch_cuda_version": runtime["torch_cuda_version"],
            "cuda_devices": runtime["cuda_devices"],
            "shared_environment": shared_environment,
        },
    }


def validate_prepared_panels(
    prepared: dict[str, Any], path: Path
) -> list[dict[str, Any]]:
    require_schema(prepared, PREPARED_SCHEMA, path)
    if prepared.get("status") != "prepared":
        raise IntegrityError(f"Prepared panels are not in prepared status: {path}")
    if int(prepared.get("context_length", -1)) != CONTEXT_LENGTH:
        raise IntegrityError("Prepared context length differs from capture contract")
    panels = prepared.get("panels")
    if not isinstance(panels, list) or len(panels) != 16:
        raise IntegrityError("Prepared suite must contain 16 panels")
    expected_indices = list(range(len(panels)))
    if [panel.get("index") for panel in panels] != expected_indices:
        raise IntegrityError("Prepared panel indices are not contiguous and ordered")
    for panel in panels:
        token_ids = validate_token_ids(
            panel.get("input_token_ids"), context=f"panel {panel.get('id')}"
        )
        observed = token_ids_sha256(token_ids)
        if observed != panel.get("input_token_ids_sha256"):
            raise IntegrityError(
                f"Prepared token hash mismatch for panel {panel.get('id')}"
            )
        if max(token_ids) >= EXPECTED_TOKENIZER_VOCAB_SIZE:
            raise IntegrityError(
                f"Out-of-tokenizer-vocabulary token in panel {panel.get('id')}"
            )
        if panel.get("independent_sequence") is not True:
            raise IntegrityError(f"Panel {panel.get('id')} is not marked independent")
    primary = sum(SCORED_ROWS for panel in panels if panel.get("role") == "primary")
    if primary != PRIMARY_SCORED_POSITIONS:
        raise IntegrityError(
            f"Prepared primary position count {primary} != {PRIMARY_SCORED_POSITIONS}"
        )
    suite_hash = object_sha256(
        [
            {
                "index": panel["index"],
                "id": panel["id"],
                "token_hash": panel["input_token_ids_sha256"],
            }
            for panel in panels
        ]
    )
    if suite_hash != prepared.get("suite_token_hash_sha256"):
        raise IntegrityError("Prepared suite token hash is inconsistent")
    return panels


def capture_one_panel(
    llm: Any,
    sampling_params: Any,
    flat_logprobs_type: type,
    panel: dict[str, Any],
    out_dir: Path,
    vocab_size: int,
) -> dict[str, Any]:
    token_ids = panel["input_token_ids"]
    panel_index = int(panel["index"])
    stem = f"panel-{panel_index:02d}-{panel['id']}"
    tokens_path = out_dir / f"{stem}.tokens.npy"
    logprobs_path = out_dir / f"{stem}.logprobs.npy"
    partial_path = out_dir / f"{stem}.logprobs.partial.npy"
    save_npy_exclusive(tokens_path, np.asarray(token_ids, dtype="<i4"))

    started = time.monotonic()
    output = llm.generate(
        [{"prompt_token_ids": token_ids}],
        sampling_params=sampling_params,
        use_tqdm=False,
    )[0]
    if list(output.prompt_token_ids or []) != token_ids:
        raise IntegrityError(
            f"vLLM prompt token IDs differ from prepared panel {panel['id']}"
        )
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None:
        raise UnsupportedError("vLLM returned no prompt log probabilities")
    if not isinstance(prompt_logprobs, flat_logprobs_type):
        raise UnsupportedError(
            f"Expected actual FlatLogprobs, got {type(prompt_logprobs).__name__}; "
            "dictionary/top-k capture is forbidden"
        )
    if len(prompt_logprobs) != CONTEXT_LENGTH:
        raise IntegrityError(
            f"FlatLogprobs has {len(prompt_logprobs)} positions; expected "
            f"{CONTEXT_LENGTH} including the empty first position"
        )
    if prompt_logprobs.start_indices[0] != prompt_logprobs.end_indices[0]:
        raise IntegrityError("First prompt-logprob position is not empty")

    # These two full-size Python lists are not part of the evidence and can be
    # released before constructing the row-major memmap.
    prompt_logprobs.ranks.clear()
    prompt_logprobs.decoded_tokens.clear()

    memmap = np.lib.format.open_memmap(
        partial_path,
        mode="w+",
        dtype="<f4",
        shape=(SCORED_ROWS, vocab_size),
    )
    seen = np.zeros(vocab_size, dtype=np.bool_)
    normalization_abs: list[float] = []
    row_widths: list[int] = []
    duplicate_rows = 0
    raw_id_order_hash = hashlib.sha256()
    top1_ids = np.empty(SCORED_ROWS, dtype="<i4")
    gold_logprobs = np.empty(SCORED_ROWS, dtype="<f4")

    for row_index in range(SCORED_ROWS):
        source_index = row_index + 1
        start = int(prompt_logprobs.start_indices[source_index])
        end = int(prompt_logprobs.end_indices[source_index])
        width = end - start
        if width not in (vocab_size, vocab_size + 1):
            raise IntegrityError(
                f"Panel {panel['id']} row {row_index} has {width} values; "
                f"full vocabulary requires {vocab_size} unique IDs (the R38 API "
                "may include one duplicate gold ID)"
            )
        ids = np.asarray(prompt_logprobs.token_ids[start:end], dtype=np.int64)
        values = np.asarray(prompt_logprobs.logprobs[start:end], dtype=np.float32)
        if ids.size != width or values.size != width:
            raise IntegrityError(
                "FlatLogprobs row bounds do not match flattened values"
            )
        if ids.min(initial=0) < 0 or ids.max(initial=-1) >= vocab_size:
            raise IntegrityError(
                f"Panel {panel['id']} row {row_index} contains an invalid vocabulary ID"
            )
        seen.fill(False)
        seen[ids] = True
        if int(np.count_nonzero(seen)) != vocab_size:
            raise IntegrityError(
                f"Panel {panel['id']} row {row_index} is truncated or has missing IDs"
            )
        gold_id = int(token_ids[row_index + 1])
        gold_mask = ids == gold_id
        expected_gold_copies = 2 if width == vocab_size + 1 else 1
        if int(np.count_nonzero(gold_mask)) != expected_gold_copies:
            raise IntegrityError(
                f"Panel {panel['id']} row {row_index} does not have the expected "
                f"gold-token duplication contract"
            )
        if expected_gold_copies == 2:
            duplicate_values = values[gold_mask].astype(np.float64)
            if float(duplicate_values.max() - duplicate_values.min()) > 1.0e-7:
                raise IntegrityError(
                    f"Duplicate gold log probabilities disagree in panel {panel['id']} "
                    f"row {row_index}"
                )
            duplicate_rows += 1
        row = memmap[row_index]
        row.fill(np.nan)
        row[ids] = values
        if not np.isfinite(row).all():
            raise IntegrityError(
                f"Panel {panel['id']} row {row_index} contains non-finite log probabilities"
            )
        row_max = float(np.max(row))
        log_z = row_max + math.log(
            float(np.exp(row.astype(np.float64) - row_max).sum(dtype=np.float64))
        )
        abs_log_z = abs(log_z)
        if abs_log_z > NORMALIZATION_TOLERANCE:
            raise IntegrityError(
                f"Panel {panel['id']} row {row_index} logsumexp={log_z:.9g}; "
                "full-vocabulary normalization failed"
            )
        normalization_abs.append(abs_log_z)
        row_widths.append(width)
        top1_ids[row_index] = int(np.argmax(row))
        gold_logprobs[row_index] = row[gold_id]
        raw_id_order_hash.update(ids.astype("<i4", copy=False).tobytes())

    memmap.flush()
    del memmap
    if logprobs_path.exists():
        raise FileExistsError(f"Refusing to overwrite {logprobs_path}")
    partial_path.rename(logprobs_path)
    # Drop the enormous Pythonized engine object before hashing the persisted
    # artifact or starting the next prompt.
    prompt_logprobs.start_indices.clear()
    prompt_logprobs.end_indices.clear()
    prompt_logprobs.token_ids.clear()
    prompt_logprobs.logprobs.clear()
    del output, prompt_logprobs
    gc.collect()
    elapsed = time.monotonic() - started
    record = {
        "index": panel_index,
        "id": panel["id"],
        "corpus": panel["corpus"],
        "stratum": panel["stratum"],
        "role": panel["role"],
        "token_file": tokens_path.name,
        "token_file_sha256": sha256_file(tokens_path),
        "input_token_ids_sha256": panel["input_token_ids_sha256"],
        "logprobs_file": logprobs_path.name,
        "logprobs_file_sha256": sha256_file(logprobs_path),
        "logprobs_shape": [SCORED_ROWS, vocab_size],
        "logprobs_dtype": "float32",
        "raw_flat_id_order_sha256": raw_id_order_hash.hexdigest(),
        "full_vocab_unique_ids_per_row": True,
        "finite_values": True,
        "normalization": {
            "method": "float64 logsumexp of stored float32 row",
            "tolerance_abs_logsumexp": NORMALIZATION_TOLERANCE,
            "max_abs_logsumexp": max(normalization_abs),
            "mean_abs_logsumexp": float(np.mean(normalization_abs)),
        },
        "raw_row_width": {
            "min": min(row_widths),
            "max": max(row_widths),
            "rows_with_r38_gold_duplicate": duplicate_rows,
        },
        "gold_next_token_nll_mean": float(-np.mean(gold_logprobs, dtype=np.float64)),
        "gold_next_token_ppl": float(
            math.exp(float(-np.mean(gold_logprobs, dtype=np.float64)))
        ),
        "gold_next_token_top1_accuracy": float(
            np.mean(top1_ids == np.asarray(token_ids[1:], dtype=np.int32))
        ),
        "elapsed_seconds": elapsed,
    }
    return record


def command_capture(args: argparse.Namespace) -> int:
    create_output_dir(args.out_dir)
    failure_path = args.out_dir / "failure.json"
    try:
        prepared = load_json(args.panels)
        panels = validate_prepared_panels(prepared, args.panels)
        llm_args_source, declared_metadata, llm_args_file_hash = load_llm_args(
            args.llm_args
        )
        validate_checkpoint_metadata(declared_metadata, llm_args_source)
        llm_args = prepare_capture_llm_args(args.model.resolve(), llm_args_source)

        # Capture and compare must not fetch code, configs, or tokenizer state.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # The source FP8 target must not gain an online-quantized vocabulary head.
        os.environ["VLLM_MXFP8_LM_HEAD"] = "0"
        os.environ["VLLM_MTP_NVFP4_LM_HEAD"] = "0"

        start_record = {
            "schema": "orcarouter-fidelity-capture-start.v1",
            "status": "running",
            "created_utc": utc_now(),
            "label": args.label,
            "model": str(args.model.resolve()),
            "panels": str(args.panels.resolve()),
            "panels_file_sha256": sha256_file(args.panels),
            "llm_args_file": str(args.llm_args.resolve()),
            "llm_args_file_sha256": llm_args_file_hash,
            "declared_metadata": declared_metadata,
            "effective_llm_args": llm_args,
        }
        write_json_exclusive(args.out_dir / "capture-start.json", start_record)

        try:
            import vllm
            from vllm import LLM, SamplingParams
            from vllm.logprobs import FlatLogprobs
        except ImportError as exc:
            raise UnsupportedError(
                "capture must run inside the pinned target vLLM image"
            ) from exc
        if vllm.__version__ != EXPECTED_VLLM_VERSION:
            raise UnsupportedError(
                f"Capture requires R38 vLLM {EXPECTED_VLLM_VERSION}; "
                f"active version is {vllm.__version__}"
            )
        sampling_fields = set(getattr(SamplingParams, "__struct_fields__", ()))
        if "flat_logprobs" not in sampling_fields and not hasattr(
            SamplingParams, "flat_logprobs"
        ):
            raise UnsupportedError(
                "Active SamplingParams has no flat_logprobs capability"
            )

        llm = LLM(**llm_args)
        engine, runtime = extract_runtime_metadata(llm)
        vocab_size = int(engine["vocab_size"])
        if int(prepared["tokenizer"]["len"]) != EXPECTED_TOKENIZER_VOCAB_SIZE:
            raise IntegrityError(
                "Prepared tokenizer size differs from the pinned tokenizer contract"
            )
        if vocab_size != EXPECTED_MODEL_OUTPUT_VOCAB_SIZE:
            raise IntegrityError(
                f"Model output vocabulary is {vocab_size}; expected "
                f"{EXPECTED_MODEL_OUTPUT_VOCAB_SIZE}"
            )
        validate_active_capture_contract(engine, vocab_size)
        params = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=-1,
            flat_logprobs=True,
            detokenize=False,
        )
        records = []
        capture_started = time.monotonic()
        for panel in panels:
            record = capture_one_panel(
                llm,
                params,
                FlatLogprobs,
                panel,
                args.out_dir,
                vocab_size,
            )
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
        elapsed = time.monotonic() - capture_started

        contract = comparison_contract(prepared, engine, runtime, llm_args)
        manifest = {
            "schema": CAPTURE_SCHEMA,
            "status": "pass",
            "created_utc": utc_now(),
            "label": args.label,
            "model": model_identity(args.model.resolve()),
            "declared_metadata": declared_metadata,
            "panels_file": str(args.panels.resolve()),
            "panels_file_sha256": sha256_file(args.panels),
            "suite_token_hash_sha256": prepared["suite_token_hash_sha256"],
            "tokenizer": prepared["tokenizer"],
            "llm_args_file": str(args.llm_args.resolve()),
            "llm_args_file_sha256": llm_args_file_hash,
            "effective_llm_args": llm_args,
            "model_specific_llm_args": {
                key: value
                for key, value in llm_args.items()
                if key in MODEL_SPECIFIC_LLM_ARGS
            },
            "engine": engine,
            "runtime": runtime,
            "precision_environment": {
                key: runtime["environment"].get(key)
                for key in sorted(MODEL_PRECISION_ENV)
            },
            "capture_method": {
                "api": "offline vllm.LLM.generate",
                "sampling": {
                    "prompt_logprobs": -1,
                    "flat_logprobs": True,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "detokenize": False,
                },
                "semantic_point": "raw_prompt_logprobs_before_sampling_processors",
                "full_vocab_validation": (
                    "every row covers each ID 0..vocab-1 exactly once after allowing "
                    "R38's one equal-valued duplicate gold-token entry"
                ),
                "normalization_check": "independent float64 logsumexp per stored row",
                "cross_window_accumulation": False,
                "prefix_cache": False,
                "speculation": False,
            },
            "comparison_contract": contract,
            "comparison_contract_sha256": object_sha256(contract),
            "panel_count": len(records),
            "scored_positions": len(records) * SCORED_ROWS,
            "primary_scored_positions": PRIMARY_SCORED_POSITIONS,
            "elapsed_seconds": elapsed,
            "panels": records,
            "limitations": [
                (
                    "R38 computes topk(vocab) and Pythonizes the complete prompt-logprob "
                    "tensor before FlatLogprobs is returned; capture is exact but has a "
                    "large per-window host-memory peak."
                ),
                (
                    "This is an end-to-end configured-runtime comparison. It does not "
                    "isolate weight-only dequantization error or replicate an author's "
                    "unpublished token IDs."
                ),
            ],
        }
        write_json_exclusive(args.out_dir / "manifest.json", manifest)
        summary = {
            "schema": CAPTURE_SUMMARY_SCHEMA,
            "status": "pass",
            "label": args.label,
            "manifest": "manifest.json",
            "manifest_sha256": sha256_file(args.out_dir / "manifest.json"),
            "panel_count": len(records),
            "scored_positions": len(records) * SCORED_ROWS,
            "primary_scored_positions": PRIMARY_SCORED_POSITIONS,
            "vocab_size": vocab_size,
            "all_full_vocab_verified": all(
                record["full_vocab_unique_ids_per_row"] for record in records
            ),
            "all_finite": all(record["finite_values"] for record in records),
            "max_abs_logsumexp": max(
                record["normalization"]["max_abs_logsumexp"] for record in records
            ),
            "comparison_contract_sha256": manifest["comparison_contract_sha256"],
            "elapsed_seconds": elapsed,
        }
        write_json_exclusive(args.out_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        record_failure(failure_path, "capture", exc)
        raise


def json_differences(reference: Any, candidate: Any, prefix: str = "") -> list[str]:
    if type(reference) is not type(candidate):
        return [prefix or "$"]
    if isinstance(reference, dict):
        paths: list[str] = []
        for key in sorted(set(reference) | set(candidate)):
            child = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in candidate:
                paths.append(child)
            else:
                paths.extend(json_differences(reference[key], candidate[key], child))
        return paths
    if isinstance(reference, list):
        if len(reference) != len(candidate):
            return [prefix or "$"]
        paths = []
        for index, (left, right) in enumerate(zip(reference, candidate)):
            paths.extend(json_differences(left, right, f"{prefix}[{index}]"))
        return paths
    return [] if reference == candidate else [prefix or "$"]


def validate_capture_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    manifest = load_json(manifest_path)
    require_schema(manifest, CAPTURE_SCHEMA, manifest_path)
    if manifest.get("status") != "pass":
        raise IntegrityError(f"Capture is not successful: {manifest_path}")
    contract = manifest.get("comparison_contract")
    if not isinstance(contract, dict):
        raise IntegrityError(f"Capture has no comparison contract: {manifest_path}")
    if object_sha256(contract) != manifest.get("comparison_contract_sha256"):
        raise IntegrityError(f"Comparison contract hash mismatch: {manifest_path}")
    records = manifest.get("panels")
    if not isinstance(records, list) or len(records) != 16:
        raise IntegrityError(f"Capture must have 16 panel records: {manifest_path}")
    if [record.get("index") for record in records] != list(range(16)):
        raise IntegrityError(f"Capture panel indices are invalid: {manifest_path}")
    return manifest


def verify_capture_artifact(
    directory: Path, record: dict[str, Any], expected_vocab: int
) -> tuple[np.memmap, np.ndarray]:
    logprobs_path = directory / record["logprobs_file"]
    tokens_path = directory / record["token_file"]
    if sha256_file(logprobs_path) != record.get("logprobs_file_sha256"):
        raise IntegrityError(f"Logprob artifact hash mismatch: {logprobs_path}")
    if sha256_file(tokens_path) != record.get("token_file_sha256"):
        raise IntegrityError(f"Token artifact hash mismatch: {tokens_path}")
    tokens = np.load(tokens_path, allow_pickle=False)
    if tokens.dtype.kind not in "iu" or tokens.shape != (CONTEXT_LENGTH,):
        raise IntegrityError(f"Unexpected token artifact shape or dtype: {tokens_path}")
    if token_ids_sha256(tokens) != record.get("input_token_ids_sha256"):
        raise IntegrityError(f"Token content hash mismatch: {tokens_path}")
    logprobs = np.load(logprobs_path, mmap_mode="r", allow_pickle=False)
    if logprobs.dtype.kind != "f" or logprobs.dtype.itemsize != 4:
        raise IntegrityError(f"Expected float32 log probabilities: {logprobs_path}")
    if logprobs.shape != (SCORED_ROWS, expected_vocab):
        raise IntegrityError(
            f"Unexpected logprob shape {logprobs.shape} in {logprobs_path}"
        )
    return logprobs, tokens.astype(np.int64, copy=False)


def streaming_logsumexp_and_top1(
    values: np.memmap,
    row_start: int,
    row_end: int,
    vocab_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = row_end - row_start
    log_z = np.full(rows, -np.inf, dtype=np.float64)
    top_values = np.full(rows, -np.inf, dtype=np.float64)
    top_ids = np.zeros(rows, dtype=np.int32)
    for vocab_start in range(0, values.shape[1], vocab_chunk):
        vocab_end = min(values.shape[1], vocab_start + vocab_chunk)
        chunk = np.asarray(
            values[row_start:row_end, vocab_start:vocab_end], dtype=np.float64
        )
        if not np.isfinite(chunk).all():
            raise IntegrityError(
                f"Non-finite stored log probability at rows [{row_start}, {row_end}) "
                f"vocabulary [{vocab_start}, {vocab_end})"
            )
        local_max = np.max(chunk, axis=1)
        local_lse = local_max + np.log(
            np.exp(chunk - local_max[:, None]).sum(axis=1, dtype=np.float64)
        )
        log_z = np.logaddexp(log_z, local_lse)
        local_ids = np.argmax(chunk, axis=1)
        local_values = chunk[np.arange(rows), local_ids]
        update = local_values > top_values
        top_values[update] = local_values[update]
        top_ids[update] = local_ids[update].astype(np.int32) + vocab_start
    return log_z, top_ids


def compare_panel_arrays(
    reference: np.memmap,
    candidate: np.memmap,
    token_ids: np.ndarray,
    *,
    position_block: int,
    vocab_chunk: int,
    reverse_kl: bool,
) -> dict[str, np.ndarray | int]:
    rows, vocab_size = reference.shape
    kl = np.zeros(rows, dtype=np.float64)
    reverse = np.zeros(rows, dtype=np.float64) if reverse_kl else None
    js = np.zeros(rows, dtype=np.float64)
    reference_nll = np.zeros(rows, dtype=np.float64)
    candidate_nll = np.zeros(rows, dtype=np.float64)
    reference_top1 = np.zeros(rows, dtype=np.int32)
    candidate_top1 = np.zeros(rows, dtype=np.int32)
    reference_log_z = np.zeros(rows, dtype=np.float64)
    candidate_log_z = np.zeros(rows, dtype=np.float64)

    for row_start in range(0, rows, position_block):
        row_end = min(rows, row_start + position_block)
        ref_z, ref_top = streaming_logsumexp_and_top1(
            reference, row_start, row_end, vocab_chunk
        )
        cand_z, cand_top = streaming_logsumexp_and_top1(
            candidate, row_start, row_end, vocab_chunk
        )
        if (
            float(np.max(np.abs(ref_z))) > NORMALIZATION_TOLERANCE
            or float(np.max(np.abs(cand_z))) > NORMALIZATION_TOLERANCE
        ):
            raise IntegrityError(
                f"Stored rows [{row_start}, {row_end}) fail independent "
                "normalization verification"
            )
        reference_log_z[row_start:row_end] = ref_z
        candidate_log_z[row_start:row_end] = cand_z
        reference_top1[row_start:row_end] = ref_top
        candidate_top1[row_start:row_end] = cand_top
        block_kl = np.zeros(row_end - row_start, dtype=np.float64)
        block_reverse = (
            np.zeros(row_end - row_start, dtype=np.float64) if reverse_kl else None
        )
        block_js = np.zeros(row_end - row_start, dtype=np.float64)
        for vocab_start in range(0, vocab_size, vocab_chunk):
            vocab_end = min(vocab_size, vocab_start + vocab_chunk)
            ref_log_p = (
                np.asarray(
                    reference[row_start:row_end, vocab_start:vocab_end],
                    dtype=np.float64,
                )
                - ref_z[:, None]
            )
            cand_log_p = (
                np.asarray(
                    candidate[row_start:row_end, vocab_start:vocab_end],
                    dtype=np.float64,
                )
                - cand_z[:, None]
            )
            ref_p = np.exp(ref_log_p)
            cand_p = np.exp(cand_log_p)
            block_kl += np.sum(
                ref_p * (ref_log_p - cand_log_p), axis=1, dtype=np.float64
            )
            if block_reverse is not None:
                block_reverse += np.sum(
                    cand_p * (cand_log_p - ref_log_p), axis=1, dtype=np.float64
                )
            log_mid = np.logaddexp(ref_log_p, cand_log_p) - math.log(2.0)
            block_js += 0.5 * (
                np.sum(ref_p * (ref_log_p - log_mid), axis=1, dtype=np.float64)
                + np.sum(cand_p * (cand_log_p - log_mid), axis=1, dtype=np.float64)
            )
        if float(np.min(block_kl)) < -1.0e-8 or float(np.min(block_js)) < -1.0e-8:
            raise IntegrityError("Divergence has material negative values")
        np.maximum(block_kl, 0.0, out=block_kl)
        np.maximum(block_js, 0.0, out=block_js)
        kl[row_start:row_end] = block_kl
        js[row_start:row_end] = block_js
        if reverse is not None and block_reverse is not None:
            if float(np.min(block_reverse)) < -1.0e-8:
                raise IntegrityError("Reverse divergence has material negative values")
            np.maximum(block_reverse, 0.0, out=block_reverse)
            reverse[row_start:row_end] = block_reverse

        block_rows = np.arange(row_start, row_end)
        gold_ids = token_ids[row_start + 1 : row_end + 1]
        reference_nll[row_start:row_end] = -(
            np.asarray(reference[block_rows, gold_ids], dtype=np.float64) - ref_z
        )
        candidate_nll[row_start:row_end] = -(
            np.asarray(candidate[block_rows, gold_ids], dtype=np.float64) - cand_z
        )

    result: dict[str, np.ndarray | int] = {
        "kl": kl,
        "js": js,
        "reference_nll": reference_nll,
        "candidate_nll": candidate_nll,
        "reference_top1": reference_top1,
        "candidate_top1": candidate_top1,
        "reference_log_z": reference_log_z,
        "candidate_log_z": candidate_log_z,
        "normalization_tolerance": NORMALIZATION_TOLERANCE,
    }
    if reverse is not None:
        result["reverse_kl"] = reverse
    return result


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise IntegrityError("Cannot summarize empty or non-finite metric values")
    quantiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999)
    summary: dict[str, Any] = {
        "count": int(values.size),
        "mean": float(np.mean(values, dtype=np.float64)),
        "std_population": float(np.std(values, dtype=np.float64)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "quantiles": {
            f"p{quantile * 100:g}": float(np.quantile(values, quantile))
            for quantile in quantiles
        },
    }
    if float(np.min(values)) >= 0.0:
        counts, _ = np.histogram(values, bins=np.asarray(DISTRIBUTION_BOUNDS))
        summary["histogram"] = {
            "bounds": list(DISTRIBUTION_BOUNDS),
            "half_open_bin_counts": counts.astype(int).tolist(),
            "overflow_gt_10": int(np.count_nonzero(values > DISTRIBUTION_BOUNDS[-1])),
        }
    return summary


def aggregate_metrics(
    indices: np.ndarray,
    metrics: dict[str, np.ndarray],
    *,
    include_reverse: bool,
) -> dict[str, Any]:
    if indices.size == 0:
        raise IntegrityError("Cannot aggregate an empty comparison stratum")
    ref_nll = metrics["reference_nll"][indices]
    cand_nll = metrics["candidate_nll"][indices]
    result = {
        "positions": int(indices.size),
        "kl_reference_candidate": distribution_summary(metrics["kl"][indices]),
        "jensen_shannon": distribution_summary(metrics["js"][indices]),
        "reference_nll": distribution_summary(ref_nll),
        "candidate_nll": distribution_summary(cand_nll),
        "nll_delta_candidate_minus_reference": float(
            np.mean(cand_nll - ref_nll, dtype=np.float64)
        ),
        "reference_perplexity": float(
            math.exp(float(np.mean(ref_nll, dtype=np.float64)))
        ),
        "candidate_perplexity": float(
            math.exp(float(np.mean(cand_nll, dtype=np.float64)))
        ),
        "perplexity_ratio_candidate_over_reference": float(
            math.exp(float(np.mean(cand_nll - ref_nll, dtype=np.float64)))
        ),
        "top1_agreement": float(
            np.mean(
                metrics["reference_top1"][indices] == metrics["candidate_top1"][indices]
            )
        ),
        "reference_gold_top1_accuracy": float(
            np.mean(
                metrics["reference_top1"][indices] == metrics["gold_token_ids"][indices]
            )
        ),
        "candidate_gold_top1_accuracy": float(
            np.mean(
                metrics["candidate_top1"][indices] == metrics["gold_token_ids"][indices]
            )
        ),
    }
    if include_reverse:
        result["kl_candidate_reference"] = distribution_summary(
            metrics["reverse_kl"][indices]
        )
    return result


def command_compare(args: argparse.Namespace) -> int:
    create_output_dir(args.out_dir)
    failure_path = args.out_dir / "failure.json"
    try:
        if args.reference.resolve() == args.candidate.resolve():
            raise IncomparableError(
                "Reference and candidate capture directories are identical"
            )
        start = {
            "schema": "orcarouter-fidelity-comparison-start.v1",
            "status": "running",
            "created_utc": utc_now(),
            "reference": str(args.reference.resolve()),
            "candidate": str(args.candidate.resolve()),
            "position_block": args.position_block,
            "vocab_chunk": args.vocab_chunk,
            "reverse_kl": not args.no_reverse_kl,
        }
        write_json_exclusive(args.out_dir / "compare-start.json", start)
        reference_manifest = validate_capture_manifest(args.reference)
        candidate_manifest = validate_capture_manifest(args.candidate)
        reference_contract = reference_manifest["comparison_contract"]
        candidate_contract = candidate_manifest["comparison_contract"]
        differences = json_differences(reference_contract, candidate_contract)
        if differences:
            raise IncomparableError(
                "Capture comparison contracts differ at: " + ", ".join(differences[:64])
            )
        vocab_size = int(reference_contract["engine"]["vocab_size"])
        reference_records = reference_manifest["panels"]
        candidate_records = candidate_manifest["panels"]
        if [record["input_token_ids_sha256"] for record in reference_records] != [
            record["input_token_ids_sha256"] for record in candidate_records
        ]:
            raise IncomparableError("Capture panel token hashes differ")

        arrays: dict[str, list[np.ndarray]] = defaultdict(list)
        panel_reports: list[dict[str, Any]] = []
        panel_index_values: list[np.ndarray] = []
        panel_position_values: list[np.ndarray] = []
        panel_offsets: list[tuple[int, int]] = []
        total_offset = 0
        compare_started = time.monotonic()
        include_reverse = not args.no_reverse_kl
        for reference_record, candidate_record in zip(
            reference_records, candidate_records
        ):
            if (
                reference_record["index"] != candidate_record["index"]
                or reference_record["id"] != candidate_record["id"]
            ):
                raise IncomparableError("Capture panel ordering or identity differs")
            reference_values, reference_tokens = verify_capture_artifact(
                args.reference, reference_record, vocab_size
            )
            candidate_values, candidate_tokens = verify_capture_artifact(
                args.candidate, candidate_record, vocab_size
            )
            if not np.array_equal(reference_tokens, candidate_tokens):
                raise IncomparableError(
                    f"Token arrays differ for panel {reference_record['id']}"
                )
            compared = compare_panel_arrays(
                reference_values,
                candidate_values,
                reference_tokens,
                position_block=args.position_block,
                vocab_chunk=args.vocab_chunk,
                reverse_kl=include_reverse,
            )
            gold_ids = reference_tokens[1:].astype(np.int32, copy=False)
            for key in (
                "kl",
                "js",
                "reference_nll",
                "candidate_nll",
                "reference_top1",
                "candidate_top1",
                "reference_log_z",
                "candidate_log_z",
            ):
                arrays[key].append(np.asarray(compared[key]))
            if include_reverse:
                arrays["reverse_kl"].append(np.asarray(compared["reverse_kl"]))
            arrays["gold_token_ids"].append(gold_ids)
            panel_index = int(reference_record["index"])
            panel_index_values.append(np.full(SCORED_ROWS, panel_index, dtype=np.int16))
            panel_position_values.append(np.arange(SCORED_ROWS, dtype=np.int16))
            panel_offsets.append((total_offset, total_offset + SCORED_ROWS))
            total_offset += SCORED_ROWS
            local_metrics = {
                key: np.asarray(value)
                for key, value in compared.items()
                if isinstance(value, np.ndarray)
            }
            local_metrics["gold_token_ids"] = gold_ids
            local_indices = np.arange(SCORED_ROWS)
            report = {
                "index": panel_index,
                "id": reference_record["id"],
                "corpus": reference_record["corpus"],
                "stratum": reference_record["stratum"],
                "role": reference_record["role"],
                "metrics": aggregate_metrics(
                    local_indices,
                    local_metrics,
                    include_reverse=include_reverse,
                ),
                "normalization": {
                    "reference_max_abs_logsumexp": float(
                        np.max(np.abs(local_metrics["reference_log_z"]))
                    ),
                    "candidate_max_abs_logsumexp": float(
                        np.max(np.abs(local_metrics["candidate_log_z"]))
                    ),
                    "tolerance": NORMALIZATION_TOLERANCE,
                },
            }
            panel_reports.append(report)
            print(json.dumps(report, sort_keys=True), flush=True)
            del reference_values, candidate_values

        flat_metrics = {key: np.concatenate(parts) for key, parts in arrays.items()}
        flat_panel_index = np.concatenate(panel_index_values)
        flat_panel_position = np.concatenate(panel_position_values)
        raw_path = args.out_dir / "per-position-metrics.npz"
        save_npz_exclusive(
            raw_path,
            panel_index=flat_panel_index,
            panel_position=flat_panel_position,
            **flat_metrics,
        )
        all_indices = np.arange(total_offset)
        role_indices: dict[str, list[int]] = defaultdict(list)
        corpus_indices: dict[str, list[int]] = defaultdict(list)
        stratum_indices: dict[str, list[int]] = defaultdict(list)
        primary_window_means: dict[str, list[float]] = defaultdict(list)
        for panel, (start_offset, end_offset), report in zip(
            reference_records, panel_offsets, panel_reports
        ):
            indices = list(range(start_offset, end_offset))
            role_indices[panel["role"]].extend(indices)
            corpus_indices[panel["corpus"]].extend(indices)
            stratum_indices[panel["stratum"]].extend(indices)
            if panel["role"] == "primary":
                primary_window_means["kl_reference_candidate"].append(
                    report["metrics"]["kl_reference_candidate"]["mean"]
                )
                primary_window_means["jensen_shannon"].append(
                    report["metrics"]["jensen_shannon"]["mean"]
                )
                if include_reverse:
                    primary_window_means["kl_candidate_reference"].append(
                        report["metrics"]["kl_candidate_reference"]["mean"]
                    )

        primary_indices = np.asarray(role_indices["primary"], dtype=np.int64)
        if primary_indices.size != PRIMARY_SCORED_POSITIONS:
            raise IntegrityError(
                f"Compared primary positions {primary_indices.size} != "
                f"{PRIMARY_SCORED_POSITIONS}"
            )
        model_precision_differences = json_differences(
            reference_manifest.get("precision_environment", {}),
            candidate_manifest.get("precision_environment", {}),
            "precision_environment",
        ) + json_differences(
            reference_manifest.get("model_specific_llm_args", {}),
            candidate_manifest.get("model_specific_llm_args", {}),
            "model_specific_llm_args",
        )
        elapsed = time.monotonic() - compare_started
        summary = {
            "schema": COMPARE_SCHEMA,
            "status": "pass",
            "created_utc": utc_now(),
            "direction": {
                "reference": reference_manifest["label"],
                "candidate": candidate_manifest["label"],
                "headline": "KL(reference || candidate)",
            },
            "reference_identity": {
                "model": reference_manifest["model"],
                "declared_metadata": reference_manifest["declared_metadata"],
            },
            "candidate_identity": {
                "model": candidate_manifest["model"],
                "declared_metadata": candidate_manifest["declared_metadata"],
            },
            "reference_manifest": str((args.reference / "manifest.json").resolve()),
            "reference_manifest_sha256": sha256_file(args.reference / "manifest.json"),
            "candidate_manifest": str((args.candidate / "manifest.json").resolve()),
            "candidate_manifest_sha256": sha256_file(args.candidate / "manifest.json"),
            "comparison_contract_sha256": reference_manifest[
                "comparison_contract_sha256"
            ],
            "model_precision_differences": model_precision_differences,
            "model_precision_reference": reference_manifest.get(
                "precision_environment", {}
            ),
            "model_precision_candidate": candidate_manifest.get(
                "precision_environment", {}
            ),
            "primary_wikitext": aggregate_metrics(
                primary_indices, flat_metrics, include_reverse=include_reverse
            ),
            "all_panels": aggregate_metrics(
                all_indices, flat_metrics, include_reverse=include_reverse
            ),
            "by_role": {
                key: aggregate_metrics(
                    np.asarray(indices, dtype=np.int64),
                    flat_metrics,
                    include_reverse=include_reverse,
                )
                for key, indices in sorted(role_indices.items())
            },
            "by_corpus": {
                key: aggregate_metrics(
                    np.asarray(indices, dtype=np.int64),
                    flat_metrics,
                    include_reverse=include_reverse,
                )
                for key, indices in sorted(corpus_indices.items())
            },
            "by_stratum": {
                key: aggregate_metrics(
                    np.asarray(indices, dtype=np.int64),
                    flat_metrics,
                    include_reverse=include_reverse,
                )
                for key, indices in sorted(stratum_indices.items())
            },
            "per_window": panel_reports,
            "normalization": {
                "method": "two-pass float64 streaming logsumexp and renormalization",
                "tolerance_abs_logsumexp": NORMALIZATION_TOLERANCE,
                "reference_max_abs_logsumexp": float(
                    np.max(np.abs(flat_metrics["reference_log_z"]))
                ),
                "candidate_max_abs_logsumexp": float(
                    np.max(np.abs(flat_metrics["candidate_log_z"]))
                ),
                "all_values_finite": True,
                "full_vocab_verified_during_capture": True,
            },
            "raw_metrics": {
                "file": raw_path.name,
                "sha256": sha256_file(raw_path),
                "positions": total_offset,
            },
            "uncertainty": {
                "primary_windows": WIKITEXT_WINDOWS,
                "primary_positions": PRIMARY_SCORED_POSITIONS,
                "window_mean_variation": {
                    key: {
                        "mean": float(np.mean(values, dtype=np.float64)),
                        "std_population": float(np.std(values, dtype=np.float64)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                    }
                    for key, values in sorted(primary_window_means.items())
                },
                "confidence_interval_reported": False,
                "reason": (
                    "The 12 WikiText windows are deterministic contiguous windows, not "
                    "an independent random sample; a naive position bootstrap would "
                    "understate corpus and window dependence. Per-window dispersion and "
                    "the complete per-position artifact are reported instead."
                ),
                "published_replication": False,
                "published_replication_limitation": (
                    "No author's exact token-ID panel or raw logits were available. The "
                    "three campaign arms are directly comparable to each other, not an "
                    "exact replication of a published KLD scalar."
                ),
                "supplemental_scope": (
                    "The four campaign-authored benign panels are descriptive strata and "
                    "are excluded from the primary WikiText headline."
                ),
                "runtime_semantics": (
                    "Metrics describe the unmodified pinned checkpoint through the exact "
                    "recorded R38 runtime path. Checkpoint anomalies and runtime warnings "
                    "are not repaired or normalized by this probe and remain part of the "
                    "measured treatment."
                ),
            },
            "elapsed_seconds": elapsed,
        }
        write_json_exclusive(args.out_dir / "summary.json", summary)
        completion = {
            "schema": "orcarouter-fidelity-comparison-summary.v1",
            "status": "pass",
            "reference": reference_manifest["label"],
            "candidate": candidate_manifest["label"],
            "primary_positions": PRIMARY_SCORED_POSITIONS,
            "primary_mean_kl_reference_candidate": summary["primary_wikitext"][
                "kl_reference_candidate"
            ]["mean"],
            "primary_mean_js": summary["primary_wikitext"]["jensen_shannon"]["mean"],
            "primary_top1_agreement": summary["primary_wikitext"]["top1_agreement"],
            "summary_sha256": sha256_file(args.out_dir / "summary.json"),
            "raw_metrics_sha256": summary["raw_metrics"]["sha256"],
        }
        print(json.dumps(completion, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        record_failure(failure_path, "compare", exc)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="Tokenize the pinned source panel on CPU"
    )
    prepare.add_argument("--panels", type=Path, required=True)
    prepare.add_argument("--tokenizer", type=Path, required=True)
    prepare.add_argument(
        "--tokenizer-revision",
        default="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
    )
    prepare.add_argument("--wikitext-parquet", type=Path)
    prepare.add_argument("--trust-remote-code", action="store_true")
    prepare.add_argument("--out", type=Path, required=True)
    prepare.set_defaults(func=command_prepare)

    capture = subparsers.add_parser(
        "capture", help="Capture exact full-vocabulary teacher-forced log probabilities"
    )
    capture.add_argument("--model", type=Path, required=True)
    capture.add_argument("--label", required=True)
    capture.add_argument("--out-dir", type=Path, required=True)
    capture.add_argument("--panels", type=Path, required=True)
    capture.add_argument("--llm-args", type=Path, required=True)
    capture.set_defaults(func=command_capture)

    compare = subparsers.add_parser(
        "compare", help="Compare two compatible full-vocabulary captures on CPU"
    )
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--out-dir", type=Path, required=True)
    compare.add_argument("--position-block", type=int, default=8)
    compare.add_argument("--vocab-chunk", type=int, default=8192)
    compare.add_argument("--no-reverse-kl", action="store_true")
    compare.set_defaults(func=command_compare)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "position_block", 1) <= 0:
        parser.error("--position-block must be positive")
    if getattr(args, "vocab_chunk", 1) <= 0:
        parser.error("--vocab-chunk must be positive")
    try:
        return int(args.func(args))
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": FAILURE_SCHEMA,
                    "status": getattr(exc, "status", "fail"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
