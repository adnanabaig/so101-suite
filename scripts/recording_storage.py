"""Resolve recorder storage and refuse unavailable volumes before creating output."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path


class StorageUnavailable(RuntimeError):
    pass


class RecordingStorage:
    def __init__(
        self,
        output: Path,
        state_dir: Path,
        required_mount: Path | None = None,
        required_volume_uuid: str | None = None,
        min_free_gib: float = 20,
    ) -> None:
        self.output = output
        self.state_dir = state_dir
        self.required_mount = required_mount
        self.required_volume_uuid = required_volume_uuid
        self.min_free_gib = min_free_gib
        if isinstance(min_free_gib, bool) or not math.isfinite(min_free_gib) or min_free_gib <= 0:
            raise ValueError("min_free_gib must be a positive finite number")
        if required_volume_uuid and required_mount is None:
            raise ValueError("required_volume_uuid needs required_mount")

    def check(self) -> None:
        if self.required_mount is not None:
            mount = self.required_mount
            if not mount.is_mount():
                raise StorageUnavailable(
                    f"Recording paused: required drive is not mounted at {mount}"
                )
            if not self.output.resolve().is_relative_to(mount.resolve()):
                raise StorageUnavailable("Recording output resolves outside the required drive")
            if self.required_volume_uuid:
                try:
                    result = subprocess.run(
                        ["findmnt", "-n", "-o", "UUID", "--mountpoint", str(mount)],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=3,
                    )
                except (OSError, subprocess.SubprocessError) as exc:
                    raise StorageUnavailable(f"Cannot verify recording volume: {exc}") from exc
                if result.stdout.strip().casefold() != self.required_volume_uuid.casefold():
                    raise StorageUnavailable(
                        "Recording paused: the mounted volume UUID is different"
                    )
        existing = self.output
        while not existing.exists():
            existing = existing.parent
        free = shutil.disk_usage(existing).free
        if free < self.min_free_gib * 1024**3:
            raise StorageUnavailable(
                f"Recording paused: {free / 1024**3:.1f} GiB free; "
                f"{self.min_free_gib:g} GiB reserve required. Existing files retained."
            )

    def prepare(self) -> None:
        self.check()
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)


def load_storage(
    root: Path,
    *,
    output: Path | None = None,
    state_dir: Path | None = None,
    profile: Path | None = None,
    min_free_gib: float | None = None,
) -> RecordingStorage:
    default_profile = root / "config/rigs/mcap-storage.local.json"
    if profile is None and output is None and default_profile.exists():
        profile = default_profile
    values: dict = {}
    if profile is not None:
        try:
            values = json.loads(profile.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot load recording storage profile {profile}: {exc}") from exc
        allowed = {"output", "state_dir", "required_mount", "required_volume_uuid", "min_free_gib"}
        if not isinstance(values, dict) or set(values) - allowed:
            raise ValueError("Invalid recording storage profile fields")
        if not isinstance(values.get("output"), str) or not values["output"]:
            raise ValueError("Recording storage profile must specify output")
        for key in ("state_dir", "required_mount", "required_volume_uuid"):
            if key in values and (not isinstance(values[key], str) or not values[key]):
                raise ValueError(f"Invalid recording storage profile {key}")
    destination = output or Path(values.get("output", root / "data/real/mcap"))
    metadata = state_dir or Path(values.get("state_dir", destination))
    reserve = min_free_gib if min_free_gib is not None else values.get("min_free_gib", 20)
    if not isinstance(reserve, (int, float)):
        raise ValueError("min_free_gib must be numeric")
    return RecordingStorage(
        output=destination,
        state_dir=metadata,
        required_mount=Path(values["required_mount"]) if values.get("required_mount") else None,
        required_volume_uuid=values.get("required_volume_uuid"),
        min_free_gib=reserve,
    )
