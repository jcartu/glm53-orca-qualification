"""Offline security/integrity checks; fixtures are synthetic, never real credentials."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TOOLS = Path(os.environ.get("EVIDENCE_TOOLS_ROOT", REPO / "tools"))


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exporter = load_tool("export_evidence")
downloader = load_tool("download_evidence")


class EvidenceTools(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.assets = self.root / "assets"
        self.index = self.root / "index.json"

    def export(self, succeeds=True, extra=()):
        result = subprocess.run(
            [
                sys.executable,
                str(TOOLS / "export_evidence.py"),
                "--source-root",
                str(self.source),
                "--asset-dir",
                str(self.assets),
                "--manifest",
                str(self.index),
                "--threads",
                "1",
                "--compression-level",
                "1",
                "--max-asset-bytes",
                "4194304",
                "--raw-shard-bytes",
                "2097152",
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode == 0, succeeds, result.stderr[:500])
        if not succeeds:
            self.assertFalse(self.index.exists())
            self.assertFalse(self.assets.exists())
            return None
        return json.loads(self.index.read_text())

    def restore(self, succeeds=True, assets="all"):
        output = self.root / "restored"
        command = [
            sys.executable,
            str(TOOLS / "download_evidence.py"),
            "--manifest",
            str(self.index),
            "--release-base-url",
            self.assets.as_uri() + "/",
            "--output",
            str(output),
        ]
        if assets != "all":
            command += ["--assets", assets]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode == 0, succeeds, result.stderr[:500])
        self.assertEqual(output.exists(), succeeds)
        return output, result

    def replace_tar(self, raw):
        manifest = json.loads(self.index.read_text())
        asset = next(
            item for item in manifest["assets"] if item["format"] == "tar+zstd"
        )
        compressed = subprocess.run(
            ["zstd", "-q", "-1", "-c"],
            input=raw,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout
        (self.assets / asset["name"]).write_bytes(compressed)
        asset["size"] = len(compressed)
        asset["sha256"] = hashlib.sha256(compressed).hexdigest()
        self.index.write_text(json.dumps(manifest))

    def test_complete_roundtrip_numeric_bytes_and_long_paths(self):
        (self.source / "ok.txt").write_bytes(b"saved failure; never filter it away\n")
        (self.source / ("x" * 120 + ".txt")).write_bytes(b"long path\n")
        (self.source / "empty.txt").write_bytes(b"")
        (self.source / "health.request.json").write_bytes(b"")
        (self.source / "aligned.txt").write_bytes(b"x" * 512)
        np.save(
            self.source / "values.npy", np.arange(24, dtype=np.float32).reshape(4, 6)
        )
        self.export()
        output, _ = self.restore()
        expected = {path.name: path.read_bytes() for path in self.source.iterdir()}
        actual = {path.name: path.read_bytes() for path in output.iterdir()}
        self.assertEqual(actual, expected)

    def test_vendor_elf_without_extension_is_excluded_but_text_is_kept(self):
        runtime = self.source / "runtime-source"
        runtime.mkdir()
        (runtime / "server").write_bytes(b"\x7fELF" + b"\x00" * 64)
        (runtime / "README").write_bytes(b"runtime source provenance\n")
        self.export()
        output, _ = self.restore()
        self.assertEqual(
            (output / "runtime-source/README").read_bytes(),
            b"runtime source provenance\n",
        )
        self.assertFalse((output / "runtime-source/server").exists())
        excluded = json.loads(self.index.read_text())["exclusions"]
        self.assertTrue(
            any(
                item["logical_path"] == "runtime-source/server"
                and item["category"] == "vendor-runtime-binary"
                for item in excluded
            )
        )

    def test_explicit_cache_root_is_normalized_in_ordinary_text(self):
        cache_root = self.root / "private-cache"
        cache_root.mkdir()
        (self.source / "cache.log").write_text(f"storage: {cache_root}/segment\n")
        self.export(extra=("--cache-root", str(cache_root)))
        output, _ = self.restore()
        self.assertEqual(
            (output / "cache.log").read_text(),
            "storage: ${ORCA_CACHE_ROOT}/segment\n",
        )

    def test_reviewed_literal_does_not_exempt_other_credentials(self):
        reviewed = "the laminated drill cards stay in the inactive section"
        token = "ghp_" + "B" * 36
        (self.source / "record.log").write_text(
            f"credentials: {reviewed}\nAuthorization: Bearer {token}\n"
        )
        self.export(extra=("--approved-canaries", json.dumps([reviewed])))
        output, _ = self.restore()
        restored = (output / "record.log").read_text()
        self.assertIn(f"credentials: {reviewed}\n", restored)
        self.assertNotIn(token, restored)

    def test_text_only_restore_is_explicit_subset(self):
        (self.source / "ok.txt").write_text("complete text evidence")
        np.save(self.source / "values.npy", np.arange(4, dtype=np.int32))
        self.export()
        output, result = self.restore(assets="text")
        self.assertTrue((output / "ok.txt").is_file())
        self.assertFalse((output / "values.npy").exists())
        self.assertFalse(json.loads(result.stdout)["complete"])

    def test_sensitive_fields_and_numbers_survive_without_rounding(self):
        (self.source / "data.json").write_text(
            '{"password":123456,"precise":0.12345678901234567890123456789,'
            '"huge":1e400,"expected":"ORCA_SYNTHETIC_7319"}'
        )
        self.export()
        output, _ = self.restore()
        text = (output / "data.json").read_text()
        self.assertNotIn("123456,", text)
        self.assertIn("0.12345678901234567890123456789", text)
        self.assertIn("1e400", text)
        self.assertIn("ORCA_SYNTHETIC_7319", text)

    def test_fixture_words_do_not_authorize_credentials(self):
        token = "ghp_" + "A" * 36
        (self.source / "failure.log").write_text(
            "# fixture setup\nAuthorization: Bearer " + token
        )
        (self.source / "data.json").write_text(
            json.dumps(
                {
                    "password": "live-test-password-482",
                    "test_api_key": "opaque-live-value-482",
                    "Env": ["SERVICE_API_KEY=word one word two"],
                }
            )
        )
        self.export()
        output, _ = self.restore()
        text = (output / "failure.log").read_text() + (output / "data.json").read_text()
        for private_value in (
            token,
            "live-test-password-482",
            "opaque-live-value-482",
            "word one word two",
        ):
            self.assertNotIn(private_value, text)

    def test_private_receipt_and_hardlink_alias_are_excluded(self):
        (self.source / "access-approved.json").write_text(
            '{"approved_by":"Synthetic Person"}'
        )
        (self.source / ".env").write_text("PASSWORD=opaque-private-value")
        os.link(self.source / ".env", self.source / "innocent.txt")
        os.symlink(".env", self.source / "alias.txt")
        (self.source / "failure.log").write_text("a failed measurement is retained")
        manifest = self.export()
        paths = {entry["logical_path"] for entry in manifest["files"]}
        self.assertEqual(paths, {"failure.log"})
        excluded = {entry["logical_path"] for entry in manifest["exclusions"]}
        self.assertEqual(
            excluded, {"access-approved.json", ".env", "innocent.txt", "alias.txt"}
        )

    def test_unlabelled_modern_token_aborts_publication(self):
        (self.source / "failure.log").write_text(
            "unexpected value " + "github_pat_" + "A" * 60
        )
        self.export(succeeds=False)

    def test_source_literals_redact_without_breaking_nonliteral_code(self):
        source = 'import os\napi_key = os.environ.get("API_KEY")\nCFG = {"password": "Q7m8V2n4"}\n'
        (self.source / "snapshot.py").write_text(source)
        self.export()
        output, _ = self.restore()
        text = (output / "snapshot.py").read_text()
        self.assertNotIn("Q7m8V2n4", text)
        self.assertIn('os.environ.get("API_KEY")', text)
        compile(text, "restored-snapshot", "exec")

    def test_json_schema_declarations_are_not_credentials(self):
        document = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "api_key": {"type": "string", "default": "opaque-private-default"}
            },
        }
        (self.source / "schema.json").write_text(json.dumps(document))
        self.export()
        output, _ = self.restore()
        actual = json.loads((output / "schema.json").read_text())
        self.assertEqual(actual["properties"]["api_key"]["type"], "string")
        self.assertNotIn("opaque-private-default", (output / "schema.json").read_text())

    def test_gzip_header_metadata_is_removed(self):
        buffer = io.BytesIO()
        with gzip.GzipFile(
            filename="private-header-name", fileobj=buffer, mode="wb", mtime=123
        ) as compressed:
            compressed.write(b'{"answer":391}')
        (self.source / "request.json.gz").write_bytes(buffer.getvalue())
        self.export()
        output, _ = self.restore()
        content = (output / "request.json.gz").read_bytes()
        self.assertNotIn(b"private-header-name", content)
        self.assertEqual(gzip.decompress(content), b'{"answer":391}')

    def test_failed_exclusive_open_never_deletes_existing_file(self):
        existing = self.root / "existing.txt"
        existing.write_bytes(b"sentinel")
        with self.assertRaises(FileExistsError):
            downloader.write_verified_stream(
                io.BytesIO(b"x"),
                existing,
                {
                    "exported_bytes": 1,
                    "exported_sha256": hashlib.sha256(b"x").hexdigest(),
                    "logical_path": "existing.txt",
                },
            )
        self.assertEqual(existing.read_bytes(), b"sentinel")

    def test_portable_paths_reject_drive_device_and_alias_forms(self):
        for value in (
            "C:../C:../victim.txt",
            "nested/C:../victim",
            "file:stream",
            "CON",
            "nested/NUL.txt",
            "trail./x",
            "space /x",
            "../x",
            "/x",
            "a\\b",
        ):
            with self.subTest(path=value):
                with self.assertRaises((ValueError, RuntimeError)):
                    downloader.safe_logical_path(value, "synthetic path")

    def test_nonzero_tar_tail_inside_read_ahead_is_rejected(self):
        (self.source / "ok.txt").write_bytes(b"x")
        self.export()
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as archive:
            info = tarfile.TarInfo("ok.txt")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        payload = bytearray(raw.getvalue())
        payload[2048] = 1
        self.replace_tar(payload)
        self.restore(succeeds=False)

    def test_extension_metadata_cannot_hide_large_expansion(self):
        (self.source / "ok.txt").write_bytes(b"x")
        self.export()
        extension = tarfile.TarInfo("././@LongLink")
        extension.type = tarfile.GNUTYPE_LONGNAME
        extension.size = 65536
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as archive:
            archive.addfile(extension, io.BytesIO(b"ok.txt\0" + b"\0" * (65536 - 7)))
            info = tarfile.TarInfo("ok.txt")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
        self.replace_tar(raw.getvalue())
        self.restore(succeeds=False)

    def test_compressed_tamper_never_publishes_output(self):
        (self.source / "ok.txt").write_bytes(b"x")
        manifest = self.export()
        asset = self.assets / manifest["assets"][0]["name"]
        body = bytearray(asset.read_bytes())
        body[-1] ^= 1
        asset.write_bytes(body)
        self.restore(succeeds=False)

    def test_userinfo_and_signed_url_secrets_are_removed(self):
        (self.source / "urls.json").write_text(
            json.dumps(
                {
                    "repository": "https://opaque-userinfo-value@github.com/example/repo",
                    "download": "https://example.invalid/data?sig=opaque-signed-value",
                }
            )
        )
        self.export()
        output, _ = self.restore()
        text = (output / "urls.json").read_text()
        self.assertNotIn("opaque-userinfo-value", text)
        self.assertNotIn("opaque-signed-value", text)

    def test_private_logical_filename_fails_closed(self):
        (self.source / "host-192.168.8.42.log").write_text("ordinary measurement")
        self.export(succeeds=False)

    def test_npy_suffix_does_not_authorize_arbitrary_payloads(self):
        (self.source / "not-numeric.npy").write_bytes(b"not a NumPy array")
        self.export(succeeds=False)

    def test_object_npy_is_not_a_numeric_capture(self):
        np.save(self.source / "object.npy", np.array(["synthetic text"], dtype=object))
        self.export(succeeds=False)

    def test_gzip_content_redaction_preserves_numeric_lexemes(self):
        payload = b'{"api_key":"opaque-private-value","metric":0.123456789012345678901}'
        (self.source / "request.json.gz").write_bytes(gzip.compress(payload, mtime=0))
        self.export()
        output, _ = self.restore()
        restored = gzip.decompress((output / "request.json.gz").read_bytes())
        self.assertNotIn(b"opaque-private-value", restored)
        self.assertIn(b"0.123456789012345678901", restored)

    def test_same_size_mtime_restored_mutation_prevents_publication(self):
        source = self.source / "result.txt"
        source.write_bytes(b"AAAA")
        original_stat = source.stat()
        original_create = exporter.create_tar_asset

        def mutate_before_compression(*args, **kwargs):
            source.write_bytes(b"BBBB")
            os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
            return original_create(*args, **kwargs)

        arguments = exporter.build_parser().parse_args(
            [
                "--source-root",
                str(self.source),
                "--asset-dir",
                str(self.assets),
                "--manifest",
                str(self.index),
                "--threads",
                "1",
                "--compression-level",
                "1",
            ]
        )
        with patch.object(exporter, "create_tar_asset", mutate_before_compression):
            with self.assertRaises(exporter.ExportError):
                exporter.export(arguments)
        self.assertFalse(self.assets.exists())
        self.assertFalse(self.index.exists())


if __name__ == "__main__":
    unittest.main()
