import hashlib
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = runpy.run_path(str(ROOT / "scripts/adapt_so101_g15_model.py"))
MODEL = ROOT / "src/so101/simulation/assets/robotstudio_so101/so101.xml"


def test_g15_model_reproduces_only_the_pinned_range_changes():
    adapted = MODEL.read_bytes()
    upstream = adapted
    for original, replacement in ADAPTER["REPLACEMENTS"]:
        upstream = upstream.replace(replacement, original, 1)
    assert hashlib.sha256(upstream).hexdigest() == ADAPTER["UPSTREAM_SHA256"]
    assert ADAPTER["adapt"](upstream) == adapted
    assert ADAPTER["adapt"](adapted) == adapted


def test_g15_model_adapter_refuses_unrecognized_geometry_or_ranges():
    with pytest.raises(ValueError, match="refusing"):
        ADAPTER["adapt"](MODEL.read_bytes().replace(b'axis="0 0 1"', b'axis="0 1 0"', 1))


def test_adapter_updates_manifest_with_model_and_checks_both(tmp_path, monkeypatch):
    upstream = MODEL.read_bytes()
    for original, replacement in ADAPTER["REPLACEMENTS"]:
        upstream = upstream.replace(replacement, original, 1)
    path = tmp_path / "so101.xml"
    path.write_bytes(upstream)
    provenance = tmp_path / "PROVENANCE.md"
    provenance.write_text(
        "# Provenance: `robotstudio_so101`\n\n| Field | Value |\n| --- | --- |\n"
        "| Commit | pinned |\n\n## Modifications\n\n"
        "1. `so101.xml`, other files are byte-exact copies.\n2. Mesh conversion.\n\n"
        "Total size: 17,440,569 bytes upstream, 2,695,969 bytes vendored.\n\n"
        "| File | Role | Faces | Bytes | SHA-256 in | SHA-256 out |\n"
        f"| `so101.xml` | copied | - | 17,206 -> 17,206 | `{ADAPTER['UPSTREAM_SHA256']}` | "
        f"`{ADAPTER['UPSTREAM_SHA256']}` |\n\n"
        "## Dynamics equivalence\n\nBase conversion proof.\n\n## Tools\n"
    )
    monkeypatch.setattr(sys, "argv", ["adapter", "--model", str(path), "--check"])
    with pytest.raises(SystemExit):
        ADAPTER["main"]()
    assert path.read_bytes() == upstream
    monkeypatch.setattr(sys, "argv", ["adapter", "--model", str(path)])
    assert ADAPTER["main"]() == 0
    assert path.read_bytes() == MODEL.read_bytes()
    manifest = provenance.read_text()
    assert "2,696,017 bytes vendored" in manifest
    assert "| adapted |" in manifest and ADAPTER["ADAPTED_SHA256"] in manifest
    assert "Dynamics equivalence (base mesh conversion)" in manifest
    assert "Base conversion proof." in manifest
    assert ADAPTER["adapt_provenance"](manifest) == manifest
    monkeypatch.setattr(sys, "argv", ["adapter", "--model", str(path), "--check"])
    assert ADAPTER["main"]() == 0
    provenance.write_text(manifest.replace(ADAPTER["ADAPTED_SHA256"], "0" * 64))
    with pytest.raises(ValueError, match="output hash"):
        ADAPTER["main"]()


def test_checked_in_manifest_is_the_canonical_adapted_manifest():
    manifest = MODEL.with_name("PROVENANCE.md").read_text()
    assert ADAPTER["adapt_provenance"](manifest) == manifest
