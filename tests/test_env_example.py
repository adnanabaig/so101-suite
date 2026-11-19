""".env.example documents exactly the settings so101 reads (audit appendix B)."""

from __future__ import annotations

from pathlib import Path

from so101.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"
SETTINGS = {f"SO101_{name.upper()}" for name in Settings.model_fields}


def _entries() -> dict[str, str]:
    entries = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, separator, value = line.partition("=")
            assert separator, f"not KEY=value: {line!r}"
            entries[key] = value
    return entries


def test_every_entry_is_a_setting():
    # Settings ignores unknown keys, so a misspelt entry would silently do nothing.
    assert set(_entries()) - SETTINGS == set()


def test_every_setting_has_an_entry():
    assert SETTINGS - set(_entries()) == set()


def test_ports_are_obvious_placeholders_and_one_camera_is_configured():
    entries = _entries()
    assert entries["SO101_FOLLOWER_PORT"] == "/dev/tty.usbmodemXXXX"
    assert entries["SO101_LEADER_PORT"] == "/dev/tty.usbmodemYYYY"
    assert entries["SO101_CAMERAS"] == Settings.model_fields["cameras"].default == "0"
