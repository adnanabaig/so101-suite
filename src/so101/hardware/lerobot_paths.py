"""Where LeRobot 0.6.1 keeps SO-101 calibration files.

Mirrors ``lerobot/utils/constants.py:68-77``::

    HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", Path(HF_HOME) / "lerobot")).expanduser()
    HF_LEROBOT_CALIBRATION = Path(
        os.getenv("HF_LEROBOT_CALIBRATION", HF_LEROBOT_HOME / "calibration")
    ).expanduser()

Robots store ``<calibration>/robots/<robot.name>/<id>.json`` (``robots/robot.py:46-56``)
and teleoperators ``<calibration>/teleoperators/<name>/<id>.json``
(``teleoperators/teleoperator.py:45-56``). In 0.6.1 the SO-101 follower is
``SOFollower.name = "so_follower"`` (``robots/so_follower/so_follower.py:44``) and
the leader ``SOLeader.name = "so_leader"`` (``teleoperators/so_leader/so_leader.py:37``).

When LeRobot is importable its own constant is used. Otherwise ``HF_HOME`` is
derived the way huggingface_hub does (``$HF_HOME`` or
``$XDG_CACHE_HOME/huggingface`` or ``~/.cache/huggingface``); that fallback is
unverified here because huggingface_hub is not installed in the dev venv.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

LEROBOT_ROBOTS_DIR = "robots"
LEROBOT_TELEOPERATORS_DIR = "teleoperators"
FOLLOWER_NAME = "so_follower"
LEADER_NAME = "so_leader"
#: Directory names used by LeRobot releases before 0.6 (unverified; for hints only).
LEGACY_NAMES = {"follower": "so101_follower", "leader": "so101_leader"}


def _fallback_hf_home() -> Path:
    default = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.join("~", ".cache")), "huggingface")
    return Path(os.path.expandvars(os.path.expanduser(os.getenv("HF_HOME", default))))


def lerobot_calibration_root() -> tuple[Path, str]:
    """Return the calibration root and how it was determined."""
    try:
        from lerobot.utils.constants import (  # type: ignore[import-not-found]
            HF_LEROBOT_CALIBRATION,
        )

        return Path(HF_LEROBOT_CALIBRATION), "lerobot.utils.constants.HF_LEROBOT_CALIBRATION"
    except Exception:
        pass
    home = Path(os.getenv("HF_LEROBOT_HOME", str(_fallback_hf_home() / "lerobot"))).expanduser()
    root = Path(os.getenv("HF_LEROBOT_CALIBRATION", str(home / "calibration"))).expanduser()
    return root, "mirrored from lerobot/utils/constants.py (LeRobot not importable)"


def lerobot_calibration_file(arm: str, device_id: str, root: Path | None = None) -> Path:
    """Path LeRobot 0.6.1 reads for this arm's calibration."""
    base = root if root is not None else lerobot_calibration_root()[0]
    if arm == "leader":
        return base / LEROBOT_TELEOPERATORS_DIR / LEADER_NAME / f"{device_id}.json"
    return base / LEROBOT_ROBOTS_DIR / FOLLOWER_NAME / f"{device_id}.json"


def legacy_calibration_file(arm: str, device_id: str, root: Path | None = None) -> Path:
    base = root if root is not None else lerobot_calibration_root()[0]
    group = LEROBOT_TELEOPERATORS_DIR if arm == "leader" else LEROBOT_ROBOTS_DIR
    return base / group / LEGACY_NAMES[arm] / f"{device_id}.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_digest(arm: str, device_id: str, root: Path | None = None) -> str | None:
    """sha256 of the arm's LeRobot calibration file, or None when it is missing."""
    path = lerobot_calibration_file(arm, device_id, root)
    try:
        return file_sha256(path)
    except FileNotFoundError:
        return None
