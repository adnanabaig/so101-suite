from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from support import SCRIPTS  # noqa: F401

from artifact_manifest import (
    MANIFEST_NAME,
    ManifestError,
    canonical_json_bytes,
    create_bundle_manifest,
    stream_verified_file,
    verify_bundle,
)


class ArtifactManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "bundle"
        (self.root / "nested").mkdir(parents=True)
        (self.root / "weights.bin").write_bytes(b"weights\x00\x01")
        (self.root / "nested" / "metadata.json").write_text('{"fps":30}\n', encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def create(self) -> dict[str, object]:
        return create_bundle_manifest(
            self.root,
            kind="checkpoint",
            name="reach-v1",
            source_commit="a" * 40,
            created_utc="2026-10-07T12:00:00Z",
        )

    def rewrite_manifest(self, value: object, *, canonical: bool = True) -> None:
        raw = (
            canonical_json_bytes(value)
            if canonical
            else json.dumps(value, indent=2).encode("utf-8")
        )
        (self.root / MANIFEST_NAME).write_bytes(raw)

    def test_create_is_canonical_and_verifies(self) -> None:
        manifest = self.create()
        raw = (self.root / MANIFEST_NAME).read_bytes()
        self.assertEqual(raw, canonical_json_bytes(manifest))
        self.assertEqual(verify_bundle(self.root), manifest)
        self.assertEqual((self.root / MANIFEST_NAME).stat().st_mode & 0o777, 0o600)

    def test_modified_payload_fails(self) -> None:
        self.create()
        (self.root / "weights.bin").write_bytes(b"different")
        with self.assertRaisesRegex(ManifestError, "mismatch"):
            verify_bundle(self.root)

    def test_atomic_payload_replacement_while_hashing_fails(self) -> None:
        self.create()
        replacement = Path(self.temporary.name) / "replacement.bin"
        replacement.write_bytes(b"weights\x00\x01")
        original_read = os.read
        replaced = False

        def replacing_read(descriptor: int, size: int) -> bytes:
            nonlocal replaced
            chunk = original_read(descriptor, size)
            opened_path = os.readlink(f"/proc/self/fd/{descriptor}")
            if chunk and not replaced and opened_path.endswith("/weights.bin"):
                replaced = True
                os.replace(replacement, self.root / "weights.bin")
            return chunk

        with mock.patch("artifact_manifest.os.read", side_effect=replacing_read):
            with self.assertRaisesRegex(ManifestError, "replaced|identity|changed"):
                verify_bundle(self.root)

    def test_atomic_manifest_replacement_during_payload_hashing_fails(self) -> None:
        self.create()
        replacement = Path(self.temporary.name) / "replacement-manifest.json"
        replacement.write_bytes((self.root / MANIFEST_NAME).read_bytes())
        original_read = os.read
        replaced = False

        def replacing_read(descriptor: int, size: int) -> bytes:
            nonlocal replaced
            chunk = original_read(descriptor, size)
            opened_path = os.readlink(f"/proc/self/fd/{descriptor}")
            if chunk and not replaced and opened_path.endswith("/weights.bin"):
                replaced = True
                os.replace(replacement, self.root / MANIFEST_NAME)
            return chunk

        with mock.patch("artifact_manifest.os.read", side_effect=replacing_read):
            with self.assertRaisesRegex(ManifestError, "replaced|identity"):
                verify_bundle(self.root)

    def test_undeclared_payload_fails(self) -> None:
        self.create()
        (self.root / "late.bin").write_bytes(b"late")
        with self.assertRaisesRegex(ManifestError, "undeclared"):
            verify_bundle(self.root)

    def test_traversal_entry_fails_before_file_access(self) -> None:
        manifest = self.create()
        manifest["files"][0]["path"] = "../outside"
        self.rewrite_manifest(manifest)
        with self.assertRaisesRegex(ManifestError, r"'\.\.'"):
            verify_bundle(self.root)

    def test_absolute_and_backslash_entries_fail(self) -> None:
        for unsafe in ("/etc/passwd", "nested\\file"):
            manifest = self.create()
            manifest["files"][0]["path"] = unsafe
            self.rewrite_manifest(manifest)
            with self.assertRaisesRegex(ManifestError, "relative POSIX"):
                verify_bundle(self.root)
            (self.root / MANIFEST_NAME).unlink()

    def test_noncanonical_json_fails(self) -> None:
        manifest = self.create()
        self.rewrite_manifest(manifest, canonical=False)
        with self.assertRaisesRegex(ManifestError, "canonical"):
            verify_bundle(self.root)

    def test_duplicate_json_key_fails(self) -> None:
        self.create()
        raw = (self.root / MANIFEST_NAME).read_text(encoding="utf-8")
        raw = raw.replace('{"artifact":', '{"schema_version":1,"artifact":', 1)
        (self.root / MANIFEST_NAME).write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(ManifestError, "duplicate JSON key"):
            verify_bundle(self.root)

    def test_lock_and_incomplete_markers_fail_creation(self) -> None:
        for marker in ("transfer.lock", ".incomplete", "episode.partial", "write.tmp"):
            path = self.root / marker
            path.write_bytes(b"marker")
            with self.assertRaisesRegex(ManifestError, "prohibited"):
                self.create()
            path.unlink()

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are not supported")
    def test_payload_symlink_fails(self) -> None:
        os.symlink(self.root / "weights.bin", self.root / "alias.bin")
        with self.assertRaisesRegex(ManifestError, "symlinks are prohibited"):
            self.create()

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are not supported")
    def test_manifest_symlink_fails(self) -> None:
        outside = Path(self.temporary.name) / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        os.symlink(outside, self.root / MANIFEST_NAME)
        with self.assertRaisesRegex(ManifestError, "refusing to replace"):
            self.create()

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are not supported")
    def test_descriptor_walk_rejects_nested_ancestor_symlink(self) -> None:
        nested_root = Path(self.temporary.name) / "nested-bundle"
        payload = nested_root / "a" / "b" / "payload.bin"
        payload.parent.mkdir(parents=True)
        payload.write_bytes(b"same verified inode")
        manifest = create_bundle_manifest(
            nested_root,
            kind="dataset",
            name="ancestor-test",
            source_commit="a" * 40,
            created_utc="2026-10-07T12:00:00Z",
        )
        old_a = Path(self.temporary.name) / "old-a"
        (nested_root / "a").rename(old_a)
        external_a = Path(self.temporary.name) / "external-a"
        (external_a / "b").mkdir(parents=True)
        os.link(old_a / "b" / "payload.bin", external_a / "b" / "payload.bin")
        os.symlink(external_a, nested_root / "a")
        entry = next(item for item in manifest["files"] if item["path"].endswith("payload.bin"))
        with self.assertRaisesRegex(ManifestError, "ancestor|symlink|securely"):
            stream_verified_file(nested_root, entry, lambda _chunk: None)

    def test_existing_manifest_is_never_overwritten(self) -> None:
        self.create()
        before = (self.root / MANIFEST_NAME).read_bytes()
        with self.assertRaisesRegex(ManifestError, "refusing to replace"):
            self.create()
        self.assertEqual((self.root / MANIFEST_NAME).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
